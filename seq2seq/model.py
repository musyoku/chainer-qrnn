import sys, os, json, pickle
import chainer.functions as F
from chainer import Chain, serializers
sys.path.append(os.path.split(os.getcwd())[0])
import qrnn as L

def save_vocab(dirname, vocab, vocab_inv):
	vocab_filename = dirname + "/vocab.pickle"
	
	try:
		os.mkdir(dirname)
	except:
		pass

	with open(vocab_filename, mode="wb") as f:
		source, target = vocab
		pickle.dump(source, f)
		pickle.dump(target, f)
		source, target = vocab_inv
		pickle.dump(source, f)
		pickle.dump(target, f)

def load_vocab(dirname):
	vocab_source, vocab_target = None, None
	vocab_source_inv, vocab_target_inv = None, None
	vocab_filename = dirname + "/vocab.pickle"
	
	if os.path.isfile(vocab_filename):
		with open(vocab_filename, mode="rb") as f:
			vocab_source = pickle.load(f)
			vocab_target = pickle.load(f)
			vocab_source_inv = pickle.load(f)
			vocab_target_inv = pickle.load(f)

	return (vocab_source, vocab_target), (vocab_source_inv, vocab_target_inv)

def save_model(dirname, model):
	model_filename = dirname + "/model.hdf5"
	param_filename = dirname + "/params.json"

	try:
		os.mkdir(dirname)
	except:
		pass

	if os.path.isfile(model_filename):
		os.remove(model_filename)
	serializers.save_hdf5(model_filename, model)

	params = {
		"vocab_size_enc": model.vocab_size_enc,
		"vocab_size_dec": model.vocab_size_dec,
		"ndim_embedding": model.ndim_embedding,
		"ndim_h": model.ndim_h,
		"num_layers": model.num_layers,
		"pooling": model.pooling,
		"zoneout": model.zoneout,
		"wstd": model.wstd,
		"attention": isinstance(model, AttentiveSeq2SeqModel),
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

		model = seq2seq(params["vocab_size_enc"], params["vocab_size_dec"], params["ndim_embedding"], params["num_layers"], params["ndim_h"], params["pooling"], params["zoneout"], params["wstd"], params["attention"])

		if os.path.isfile(model_filename):
			print("loading {} ...".format(model_filename))
			serializers.load_hdf5(model_filename, model)

		return model
	else:
		return None

def seq2seq(vocab_size_enc, vocab_size_dec, ndim_embedding, num_layers, ndim_h, pooling="fo", zoneout=False, wstd=1, attention=False):
	if attention:
		pass
	return Seq2SeqModel(vocab_size_enc, vocab_size_dec, ndim_embedding, num_layers, ndim_h, pooling, zoneout, wstd)

class Seq2SeqModel(Chain):
	def __init__(self, vocab_size_enc, vocab_size_dec, ndim_embedding, num_layers, ndim_h, pooling="fo", zoneout=False, wstd=1):
		super(Seq2SeqModel, self).__init__(
			encoder_embed=L.EmbedID(vocab_size_enc, ndim_embedding, ignore_label=0),
			decoder_embed=L.EmbedID(vocab_size_dec, ndim_embedding, ignore_label=0),
			dense=L.Linear(ndim_h, vocab_size_dec),
		)
		assert num_layers > 0
		self.vocab_size_enc = vocab_size_enc
		self.vocab_size_dec = vocab_size_dec
		self.ndim_embedding = ndim_embedding
		self.num_layers = num_layers
		self.ndim_h = ndim_h
		self.kernel_size_encoder_first = 6
		self.kernel_size_encoder_other = 4
		self.kernel_size_decoder_first = 6
		self.kernel_size_decoder_other = 4
		self.pooling = pooling
		self.zoneout = zoneout
		self.wstd = wstd

		self.add_link("enc0", L.QRNNEncoder(ndim_embedding, ndim_h, kernel_size=self.kernel_size_encoder_first, pooling=pooling, zoneout=zoneout, wstd=wstd))
		for i in xrange(num_layers - 1):
			self.add_link("enc{}".format(i + 1), L.QRNNEncoder(ndim_h, ndim_h, kernel_size=self.kernel_size_encoder_other, pooling=pooling, zoneout=zoneout, wstd=wstd))

		self.add_link("dec0", L.QRNNDecoder(ndim_embedding, ndim_h, kernel_size=self.kernel_size_decoder_first, pooling=pooling, zoneout=zoneout, wstd=wstd))
		for i in xrange(num_layers - 1):
			self.add_link("dec{}".format(i + 1), L.QRNNDecoder(ndim_h, ndim_h, kernel_size=self.kernel_size_decoder_other, pooling=pooling, zoneout=zoneout, wstd=wstd))

	def get_encoder(self, index):
		return getattr(self, "enc{}".format(index))

	def get_decoder(self, index):
		return getattr(self, "dec{}".format(index))

	def reset_state(self):
		for i in xrange(self.num_layers):
			self.get_encoder(i).reset_state()
			self.get_decoder(i).reset_state()

	def reset_encoder_state(self):
		for i in xrange(self.num_layers):
			self.get_encoder(i).reset_state()

	def reset_decoder_state(self):
		for i in xrange(self.num_layers):
			self.get_decoder(i).reset_state()

	def _forward_encoder_one_layer(self, layer_index, in_data, skip_mask=None, test=False):
		if test:
			in_data.unchain_backward()
		encoder = self.get_encoder(layer_index)
		out_data = encoder(in_data, skip_mask=skip_mask, test=test)
		if test:
			out_data.unchain_backward()
		return out_data

	def _forward_decoder_one_layer(self, layer_index, in_data, encoder_hidden_state, test=False):
		if test:
			in_data.unchain_backward()
		decoder = self.get_decoder(layer_index)
		out_data = decoder(in_data, encoder_hidden_state, test=test)
		if test:
			out_data.unchain_backward()
		return out_data

	def encode(self, X, skip_mask=None, test=False):
		batchsize = X.shape[0]
		seq_length = X.shape[1]
		enmbedding = self.encoder_embed(X)
		enmbedding = F.swapaxes(enmbedding, 1, 2)

		out_data = self._forward_encoder_one_layer(0, enmbedding, skip_mask=skip_mask, test=test)
		for layer_index in xrange(1, self.num_layers):
			out_data = self._forward_encoder_one_layer(layer_index, out_data, skip_mask=skip_mask, test=test)

		if test:
			out_data.unchain_backward()

		last_hidden_states = []
		for layer_index in xrange(0, self.num_layers):
			encoder = self.get_encoder(layer_index)
			last_hidden_states.append(encoder.get_last_hidden_state())

		return last_hidden_states

	def decode(self, X, last_hidden_states, test=False):
		assert len(last_hidden_states) == self.num_layers
		batchsize = X.shape[0]
		seq_length = X.shape[1]
		enmbedding = self.decoder_embed(X)
		enmbedding = F.swapaxes(enmbedding, 1, 2)

		out_data = self._forward_decoder_one_layer(0, enmbedding, last_hidden_states[0], test=test)
		for layer_index in xrange(1, self.num_layers):
			out_data = self._forward_decoder_one_layer(layer_index, out_data, last_hidden_states[layer_index], test=test)

		out_data = F.reshape(F.swapaxes(out_data, 1, 2), (-1, self.ndim_h))
		Y = self.dense(out_data)

		if test:
			Y.unchain_backward()

		return Y

	def _forward_decoder_one_layer_one_step(self, layer_index, in_data, encoder_hidden_state, test=False):
		if test:
			in_data.unchain_backward()
		decoder = self.get_decoder(layer_index)
		out_data = decoder.forward_one_step(in_data, encoder_hidden_state, test=test)
		if test:
			out_data.unchain_backward()
		return out_data

	def decode_one_step(self, X, last_hidden_states, test=False):
		assert len(last_hidden_states) == self.num_layers
		batchsize = X.shape[0]
		seq_length = X.shape[1]
		ksize = self.kernel_size_decoder_first

		if seq_length < ksize:
			self.reset_state()
			return self.decode(X, last_hidden_states, test=test)

		xt = X[:, -ksize:]
		enmbedding = self.decoder_embed(xt)
		enmbedding = F.swapaxes(enmbedding, 1, 2)

		ksize = self.kernel_size_decoder_other
		out_data = self._forward_decoder_one_layer_one_step(0, enmbedding, last_hidden_states[0], test=test)
		if ksize != self.kernel_size_decoder_first:
			out_data = out_data[:, :, -ksize:]
		for layer_index in xrange(1, self.num_layers):
			out_data = self._forward_decoder_one_layer_one_step(layer_index, out_data, last_hidden_states[layer_index], test=test)

		out_data = F.reshape(F.swapaxes(out_data, 1, 2), (batchsize * seq_length, -1))
		Y = self.dense(out_data)

		if test:
			Y.unchain_backward()

		return Y

class AttentiveSeq2SeqModel(Chain):
	def __init__(self, vocab_size, ndim_embedding, num_layers, ndim_h, kernel_size=4, pooling="fo", zoneout=False, wstd=1):
		super(AttentiveSeq2SeqModel, self).__init__(
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

	def get_encoder(self, index):
		return getattr(self, "enc{}".format(index))

	def get_decoder(self, index):
		return getattr(self, "dec{}".format(index))

	def reset_state(self):
		for i in xrange(self.num_layers):
			self.get_encoder(i).reset_state()
			self.get_decoder(i).reset_state()

	def _forward_encoder_one_layer(self, layer_index, in_data, skip_mask=None, test=False):
		if test:
			in_data.unchain_backward()
		encoder = self.get_encoder(layer_index, skip_mask=skip_mask)
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
