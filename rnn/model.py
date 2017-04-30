import sys, os, json, pickle
import chainer.functions as F
from chainer import Chain, serializers
sys.path.append(os.path.split(os.getcwd())[0])
import qrnn as L

def save_vocab(dirname, vocab, vocab_inv):
	vocab_filename = dirname + "/vocab.pickle"
	inv_filename = dirname + "/inv.pickle"
	
	with open(vocab_filename, mode="wb") as f:
		pickle.dump(vocab, f)
	
	with open(inv_filename, mode="wb") as f:
		pickle.dump(vocab_inv, f)

def load_vocab(dirname):
	vocab = None
	vocab_inv = None
	vocab_filename = dirname + "/vocab.pickle"
	inv_filename = dirname + "/inv.pickle"
	
	if os.path.isfile(vocab_filename):
		with open(vocab_filename, mode="rb") as f:
			vocab = pickle.load(f)
	
	if os.path.isfile(inv_filename):
		with open(inv_filename, mode="rb") as f:
			vocab_inv = pickle.load(f)

	return vocab, vocab_inv

def save_model(dirname, qrnn):
	model_filename = dirname + "/model.hdf5"
	param_filename = dirname + "/params.json"

	try:
		os.mkdir(dirname)
	except:
		pass

	if os.path.isfile(model_filename):
		os.remove(model_filename)
	serializers.save_hdf5(model_filename, qrnn)

	params = {
		"vocab_size": qrnn.vocab_size,
		"ndim_embedding": qrnn.ndim_embedding,
		"ndim_h": qrnn.ndim_h,
		"pooling": qrnn.pooling,
		"zoneout": qrnn.zoneout,
		"wstd": qrnn.wstd,
	}
	with open(param_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

def load_model(dirname):
	model_filename = dirname + "/model.hdf5"
	param_filename = dirname + "/params.json"

	if os.path.isfile(param_filename):
		print("loading {} ...".format(param_filename))
		with open(param_filename, "r") as f:
			try:
				params = json.load(f)
			except Exception as e:
				raise Exception("could not load {}".format(param_filename))

		qrnn = QRNN(params["vocab_size"], params["ndim_embedding"], params["ndim_h"], params["pooling"], params["zoneout"], params["wstd"])

		if os.path.isfile(model_filename):
			print("loading {} ...".format(model_filename))
			serializers.load_hdf5(model_filename, qrnn)

		return qrnn
	else:
		return None

class QRNN(Chain):
	def __init__(self, vocab_size, ndim_embedding, ndim_h, pooling="fo", zoneout=False, wstd=1):
		super(QRNN, self).__init__(
			embed=L.EmbedID(vocab_size, ndim_embedding, ignore_label=0),
			l1=L.QRNN(ndim_embedding, ndim_h, kernel_size=4, pooling=pooling, zoneout=zoneout, wstd=wstd),
			l2=L.QRNN(ndim_h, ndim_h, kernel_size=4, pooling=pooling, zoneout=zoneout, wstd=wstd),
			l3=L.Linear(ndim_h, vocab_size),
		)
		self.vocab_size = vocab_size
		self.ndim_embedding = ndim_embedding
		self.ndim_h = ndim_h
		self.pooling = pooling
		self.zoneout = zoneout
		self.wstd = wstd

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()

	# we use "dense convolution"
	# https://arxiv.org/abs/1608.06993
	def __call__(self, X, test=False):
		batchsize = X.shape[0]
		seq_length = X.shape[1]
		H0 = self.embed(X)
		H0 = F.swapaxes(H0, 1, 2)
		self.l1(H0, test=test)
		H1 = self.l1.get_all_hidden_states()
		self.l2(H1, test=test)
		H2 = self.l2.get_all_hidden_states() + H1
		H2 = F.reshape(F.swapaxes(H2, 1, 2), (batchsize * seq_length, -1))
		Y = self.l3(H2)
		return Y