import sys, os, json, pickle
import chainer.functions as F
from chainer import Chain, serializers
sys.path.append(os.path.split(os.getcwd())[0])
import qrnn as L

def save_vocab(dirname, vocab, vocab_inv):
	vocab_filename = dirname + "/vocab.pickle"
	inv_filename = dirname + "/inv.pickle"
	
	try:
		os.mkdir(dirname)
	except:
		pass

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
		"num_layers": qrnn.num_layers,
		"kernel_size": qrnn.kernel_size,
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

		qrnn = QRNN(params["vocab_size"], params["ndim_embedding"], params["num_layers"], params["ndim_h"], params["kernel_size"], params["pooling"], params["zoneout"], params["wstd"])

		if os.path.isfile(model_filename):
			print("loading {} ...".format(model_filename))
			serializers.load_hdf5(model_filename, qrnn)

		return qrnn
	else:
		return None

class Seq2Seq(Chain):
	def __init__(self, vocab_size, ndim_embedding, num_layers, ndim_h, kernel_size=4, pooling="fo", zoneout=False, wstd=1):
		super(QRNN, self).__init__(
			embed=L.EmbedID(vocab_size, ndim_embedding, ignore_label=0),
			dense=L.Linear(ndim_h, vocab_size),
		)
		assert num_layers > 0
		self.vocab_size = vocab_size
		self.ndim_embedding = ndim_embedding
		self.num_layers = num_layers
		self.ndim_h = ndim_h
		self.kernel_size = kernel_size
		self.pooling = pooling
		self.zoneout = zoneout
		self.wstd = wstd

		self.add_link("enc0", L.QRNNEncoder(ndim_embedding, ndim_h, kernel_size=kernel_size, pooling=pooling, zoneout=zoneout, wstd=wstd))
		for i in xrange(num_layers - 1):
			self.add_link("enc{}".format(i + 1), L.QRNNEncoder(ndim_h, ndim_h, kernel_size=kernel_size, pooling=pooling, zoneout=zoneout, wstd=wstd))

		self.add_link("dec0", L.QRNNDecoder(ndim_embedding, ndim_h, kernel_size=kernel_size, pooling=pooling, zoneout=zoneout, wstd=wstd))
		for i in xrange(num_layers - 1):
			self.add_link("dec{}".format(i + 1), L.QRNNDecoder(ndim_h, ndim_h, kernel_size=kernel_size, pooling=pooling, zoneout=zoneout, wstd=wstd))

	def encoder(self, index):
		return getattr(self, "enc{}".format(index))

	def decoder(self, index):
		return getattr(self, "dec{}".format(index))

	def reset_state(self):
		for i in xrange(self.num_layers):
			self.encoder(i).reset_state()
			self.decoder(i).reset_state()

	def _forward_encoder_one_layer(self, layer_index, in_data, skip_mask=None, test=False):
		if test:
			in_data.unchain_backward()
		encoder = self.encoder(layer_index, skip_mask=skip_mask)
		out_data = encoder(in_data, test=test)
		if test:
			out_data.unchain_backward()
		return out_data

	def _forward_decoder_one_layer(self, layer_index, in_data, encoder_hidden_state, test=False):
		if test:
			in_data.unchain_backward()
		decoder = self.decoder(layer_index, skip_mask=skip_mask)
		out_data = decoder(in_data, encoder_hidden_state, test=test)
		if test:
			out_data.unchain_backward()
		return out_data

	def encode(self, X, skip_mask=None, test=False):
		batchsize = X.shape[0]
		seq_length = X.shape[1]
		enmbedding = self.embed(X)
		enmbedding = F.swapaxes(enmbedding, 1, 2)

		out_data = self._forward_encoder_one_layer(0, enmbedding, skip_mask=skip_mask, test=test)
		for layer_index in xrange(1, self.num_layers):
			out_data = self._forward_encoder_one_layer(layer_index, out_data, skip_mask=skip_mask, test=test)

		if test:
			out_data.unchain_backward()

		last_hidden_states = []
		for layer_index in xrange(0, self.num_layers):
			encoder = self.encoder(layer_index)
			last_hidden_states.append(encoder.get_last_hidden_state())

		return last_hidden_states

	def decode(self, X, last_hidden_states, test=False):
		assert len(last_hidden_states) == self.num_layers

		out_data = self._forward_decoder_one_layer(0, enmbedding, last_hidden_states[0], test=test)
		for layer_index in xrange(1, self.num_layers):
			out_data = self._forward_decoder_one_layer(layer_index, out_data, last_hidden_states[layer_index], test=test)

		out_data = F.reshape(F.swapaxes(out_data, 1, 2), (batchsize * seq_length, -1))
		Y = self.dense(out_data)

		if test:
			Y.unchain_backward()

		return Y

class AttentiveSeq2Seq(Chain):
	def __init__(self, vocab_size, ndim_embedding, num_layers, ndim_h, kernel_size=4, pooling="fo", zoneout=False, wstd=1):
		super(QRNN, self).__init__(
			embed=L.EmbedID(vocab_size, ndim_embedding, ignore_label=0)
		)
		assert num_layers > 0
		self.vocab_size = vocab_size
		self.ndim_embedding = ndim_embedding
		self.num_layers = num_layers
		self.ndim_h = ndim_h
		self.kernel_size = kernel_size
		self.pooling = pooling
		self.zoneout = zoneout
		self.wstd = wstd

		self.add_link("enc0", L.QRNNEncoder(ndim_embedding, ndim_h, kernel_size=kernel_size, pooling=pooling, zoneout=zoneout, wstd=wstd))
		for i in xrange(num_layers - 1):
			self.add_link("enc{}".format(i + 1), L.QRNNEncoder(ndim_h, ndim_h, kernel_size=kernel_size, pooling=pooling, zoneout=zoneout, wstd=wstd))

		if num_layers == 1:
			self.add_link("dec0", L.QRNNGlobalAttentiveDecoder(ndim_embedding, ndim_h, kernel_size=kernel_size, zoneout=zoneout, wstd=wstd))
		else:
			self.add_link("dec0", L.QRNNDecoder(ndim_embedding, ndim_h, kernel_size=kernel_size, pooling=pooling, zoneout=zoneout, wstd=wstd))
			for i in xrange(num_layers - 2):
				self.add_link("dec{}".format(i + 1), L.QRNNDecoder(ndim_h, ndim_h, kernel_size=kernel_size, pooling=pooling, zoneout=zoneout, wstd=wstd))
			self.add_link("dec{}".format(num_layers - 1), L.QRNNGlobalAttentiveDecoder(ndim_embedding, ndim_h, kernel_size=kernel_size, zoneout=zoneout, wstd=wstd))

	def encoder(self, index):
		return getattr(self, "enc{}".format(index))

	def decoder(self, index):
		return getattr(self, "dec{}".format(index))

	def reset_state(self):
		for i in xrange(self.num_layers):
			self.encoder(i).reset_state()
			self.decoder(i).reset_state()

	def _forward_encoder_one_layer(self, layer_index, in_data, skip_mask=None, test=False):
		if test:
			in_data.unchain_backward()
		encoder = self.encoder(layer_index, skip_mask=skip_mask)
		out_data = encoder(in_data, test=test)
		if test:
			out_data.unchain_backward()
		return out_data

	def encode(self, X, skip_mask=None, test=False):
		batchsize = X.shape[0]
		seq_length = X.shape[1]
		enmbedding = self.embed(X)
		enmbedding = F.swapaxes(enmbedding, 1, 2)

		in_data = self._forward_encoder_one_layer(0, enmbedding, skip_mask=skip_mask, test=test)
		for layer_index in xrange(1, self.num_layers):
			in_data = self._forward_encoder_one_layer(layer_index, in_data, skip_mask=skip_mask, test=test)

		out_data = in_data

		if test:
			out_data.unchain_backward()

		out_data = F.reshape(F.swapaxes(out_data, 1, 2), (batchsize * seq_length, -1))
		Y = self.dense(out_data)

		if test:
			Y.unchain_backward()

	def __call__(self, X, test=False):
		batchsize = X.shape[0]
		seq_length = X.shape[1]
		enmbedding = self.embed(X)
		enmbedding = F.swapaxes(enmbedding, 1, 2)

		out_data = self._forward_encoder_one_layer(0, enmbedding, test=test)
		in_data = [out_data]
		for layer_index in xrange(1, self.num_layers):
			out_data = self._forward_encoder_one_layer(layer_index, sum(in_data), test=test)	# dense conv
			in_data.append(out_data)

		out_data = sum(in_data)	# dense conv

		if test:
			out_data.unchain_backward()

		out_data = F.reshape(F.swapaxes(out_data, 1, 2), (batchsize * seq_length, -1))
		Y = self.dense(out_data)

		if test:
			Y.unchain_backward()

		return Y

	def _forward_rnn_one_layer_one_step(self, layer_index, in_data, test=False):
		if test:
			in_data.unchain_backward()
		rnn = self.rnn(layer_index)
		out_data = rnn.forward_one_step(in_data, test=test)
		if test:
			out_data.unchain_backward()
		return out_data

	def forward_one_step(self, X, test=False):
		batchsize = X.shape[0]
		seq_length = X.shape[1]
		ksize = self.kernel_size

		if seq_length < ksize:
			self.reset_state()
			return self.__call__(X, test=test)

		xt = X[:, -ksize:]
		enmbedding = self.embed(xt)
		enmbedding = F.swapaxes(enmbedding, 1, 2)

		out_data = self._forward_rnn_one_layer_one_step(0, enmbedding, test=test)
		in_data = [out_data]
		for layer_index in xrange(1, self.num_layers):
			out_data = self._forward_rnn_one_layer_one_step(layer_index, sum(in_data), test=test)
			in_data.append(out_data)

		out_data = sum(in_data)	# dense conv

		if test:
			out_data.unchain_backward()
			
		out_data = F.reshape(F.swapaxes(out_data, 1, 2), (batchsize * seq_length, -1))
		Y = self.dense(out_data)

		if test:
			Y.unchain_backward()

		return Y
