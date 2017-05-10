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
		"wgain": model.wgain,
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

		model = seq2seq(vocab_size_enc=params["vocab_size_enc"], vocab_size_dec=params["vocab_size_dec"], ndim_embedding=params["ndim_embedding"], num_layers=params["num_layers"], ndim_h=params["ndim_h"], pooling=params["pooling"], zoneout=params["zoneout"], wgain=params["wgain"], densely_connected=params["densely_connected"], attention=params["attention"])

		if os.path.isfile(model_filename):
			print("loading {} ...".format(model_filename))
			serializers.load_hdf5(model_filename, model)

		return model
	else:
		return None

def seq2seq(vocab_size_enc, vocab_size_dec, ndim_embedding, num_layers, ndim_h, pooling="fo", zoneout=False, wgain=1, densely_connected=False, attention=False):
	if attention:
		return AttentiveSeq2SeqModel(vocab_size_enc, vocab_size_dec, ndim_embedding, num_layers, ndim_h, pooling, zoneout, wgain, densely_connected)
	return Seq2SeqModel(vocab_size_enc, vocab_size_dec, ndim_embedding, num_layers, ndim_h, pooling, zoneout, wgain, densely_connected)

class Seq2SeqModel(Chain):
	def __init__(self, vocab_size_enc, vocab_size_dec, ndim_embedding, num_layers, ndim_h, pooling="fo", zoneout=False, wgain=1, densely_connected=False):
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
		self.kernel_size_first = 6
		self.kernel_size_other = 4
		self.pooling = pooling
		self.zoneout = zoneout
		self.densely_connected = densely_connected
		self.wgain = wgain

		self.add_link("enc0", L.QRNNEncoder(ndim_embedding, ndim_h, kernel_size=self.kernel_size_first, pooling=pooling, zoneout=zoneout, wgain=wgain))
		for i in xrange(num_layers - 1):
			self.add_link("enc{}".format(i + 1), L.QRNNEncoder(ndim_h, ndim_h, kernel_size=self.kernel_size_other, pooling=pooling, zoneout=zoneout, wgain=wgain))

		self.add_link("dec0", L.QRNNDecoder(ndim_embedding, ndim_h, pooling=pooling, zoneout=zoneout, wgain=wgain))
		for i in xrange(num_layers - 1):
			self.add_link("dec{}".format(i + 1), L.QRNNDecoder(ndim_h, ndim_h, pooling=pooling, zoneout=zoneout, wgain=wgain))

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

	def _forward_encoder_layer(self, layer_index, in_data, skip_mask=None, test=False):
		if test:
			in_data.unchain_backward()
		encoder = self.get_encoder(layer_index)
		out_data = encoder(in_data, skip_mask=skip_mask, test=test)
		if test:
			out_data.unchain_backward()
		return out_data

	def _forward_decoder_layer(self, layer_index, in_data, encoder_last_hidden_states, test=False):
		if test:
			in_data.unchain_backward()
		decoder = self.get_decoder(layer_index)
		out_data = decoder(in_data, encoder_last_hidden_states, test=test)
		if test:
			out_data.unchain_backward()
		return out_data

	def encode(self, X, skip_mask=None, test=False):
		batchsize = X.shape[0]
		seq_length = X.shape[1]
		enmbedding = self.encoder_embed(X)
		enmbedding = F.swapaxes(enmbedding, 1, 2)

		out_data = self._forward_encoder_layer(0, enmbedding, skip_mask=skip_mask, test=test)
		in_data = [out_data]

		for layer_index in xrange(1, self.num_layers):
			out_data = self._forward_encoder_layer(layer_index, sum(in_data) if self.densely_connected else in_data[-1], skip_mask=skip_mask, test=test)
			in_data.append(out_data)

		out_data = sum(in_data) if self.densely_connected else out_data	# dense conv

		if test:
			out_data.unchain_backward()

		last_hidden_states = []
		for layer_index in xrange(0, self.num_layers):
			encoder = self.get_encoder(layer_index)
			last_hidden_states.append(encoder.get_last_hidden_state())

		return last_hidden_states

	def decode(self, X, encoder_last_hidden_states, test=False, return_last=False):
		assert len(encoder_last_hidden_states) == self.num_layers
		batchsize = X.shape[0]
		seq_length = X.shape[1]
		enmbedding = self.decoder_embed(X)
		enmbedding = F.swapaxes(enmbedding, 1, 2)


		out_data = self._forward_decoder_layer(0, enmbedding, encoder_last_hidden_states[0], test=test)
		in_data = [out_data]

		for layer_index in xrange(1, self.num_layers):
			out_data = self._forward_decoder_layer(layer_index, sum(in_data) if self.densely_connected else in_data[-1], encoder_last_hidden_states[layer_index], test=test)
			in_data.append(out_data)

		out_data = sum(in_data) if self.densely_connected else out_data	# dense conv

		if return_last:
			out_data = out_data[:, :, -1, None]

		out_data = F.reshape(F.swapaxes(out_data, 1, 2), (-1, self.ndim_h))
		Y = self.dense(out_data)

		if test:
			Y.unchain_backward()

		return Y

	def _forward_decoder_layer_one_step(self, layer_index, in_data, encoder_last_hidden_states, test=False):
		if test:
			in_data.unchain_backward()
		decoder = self.get_decoder(layer_index)
		out_data = decoder.forward_one_step(in_data, encoder_last_hidden_states, test=test)
		if test:
			out_data.unchain_backward()
		return out_data

	def decode_one_step(self, X, encoder_last_hidden_states, test=False):
		assert len(encoder_last_hidden_states) == self.num_layers
		batchsize = X.shape[0]
		seq_length = X.shape[1]

		xt = X[:, -1, None]
		enmbedding = self.decoder_embed(xt)
		enmbedding = F.swapaxes(enmbedding, 1, 2)

		out_data = self._forward_decoder_layer_one_step(0, enmbedding, encoder_last_hidden_states[0], test=test)
		in_data = [out_data]

		for layer_index in xrange(1, self.num_layers):
			out_data = self._forward_decoder_layer_one_step(layer_index, sum(in_data) if self.densely_connected else in_data[-1], encoder_last_hidden_states[layer_index], test=test)
			in_data.append(out_data)

		out_data = sum(in_data) if self.densely_connected else out_data	# dense conv
		out_data = out_data[:, :, -1, None]
		out_data = F.reshape(F.swapaxes(out_data, 1, 2), (-1, self.ndim_h))
		Y = self.dense(out_data)

		if test:
			Y.unchain_backward()

		return Y

class AttentiveSeq2SeqModel(Chain):
	def __init__(self, vocab_size_enc, vocab_size_dec, ndim_embedding, num_layers, ndim_h, pooling="fo", zoneout=False, wgain=1, densely_connected=False):
		super(AttentiveSeq2SeqModel, self).__init__(
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
		self.kernel_size_first = 6
		self.kernel_size_other = 4
		self.densely_connected = densely_connected
		self.pooling = pooling
		self.zoneout = zoneout
		self.wgain = wgain

		self.add_link("enc0", L.QRNNEncoder(ndim_embedding, ndim_h, kernel_size=self.kernel_size_first, pooling=pooling, zoneout=zoneout, wgain=wgain))
		for i in xrange(num_layers - 1):
			self.add_link("enc{}".format(i + 1), L.QRNNEncoder(ndim_h, ndim_h, kernel_size=self.kernel_size_other, pooling=pooling, zoneout=zoneout, wgain=wgain))

		if num_layers == 1:
			self.add_link("dec0", L.QRNNGlobalAttentiveDecoder(ndim_embedding, ndim_h, zoneout=zoneout, wgain=wgain))
		else:
			self.add_link("dec0", L.QRNNDecoder(ndim_embedding, ndim_h, pooling=pooling, zoneout=zoneout, wgain=wgain))
			for i in xrange(num_layers - 2):
				self.add_link("dec{}".format(i + 1), L.QRNNDecoder(ndim_h, ndim_h, pooling=pooling, zoneout=zoneout, wgain=wgain))
			self.add_link("dec{}".format(num_layers - 1), L.QRNNGlobalAttentiveDecoder(ndim_h, ndim_h, zoneout=zoneout, wgain=wgain))

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

	def _forward_encoder_layer(self, layer_index, in_data, skip_mask=None, test=False):
		if test:
			in_data.unchain_backward()
		encoder = self.get_encoder(layer_index)
		out_data = encoder(in_data, skip_mask=skip_mask, test=test)
		if test:
			out_data.unchain_backward()
		return out_data

	def encode(self, X, skip_mask=None, test=False):
		batchsize = X.shape[0]
		seq_length = X.shape[1]
		enmbedding = self.encoder_embed(X)
		enmbedding = F.swapaxes(enmbedding, 1, 2)

		out_data = self._forward_encoder_layer(0, enmbedding, skip_mask=skip_mask, test=test)
		for layer_index in xrange(1, self.num_layers):
			out_data = self._forward_encoder_layer(layer_index, out_data, skip_mask=skip_mask, test=test)

		if test:
			out_data.unchain_backward()

		last_hidden_states = []
		last_layer_outputs = None
		for layer_index in xrange(0, self.num_layers):
			encoder = self.get_encoder(layer_index)
			last_hidden_states.append(encoder.get_last_hidden_state())
			last_layer_outputs = encoder.get_all_hidden_states()

		return last_hidden_states, last_layer_outputs

	def _forward_decoder_layer(self, layer_index, in_data, encoder_last_hidden_states, encoder_last_layer_outputs, encoder_skip_mask=None, test=False):
		if test:
			in_data.unchain_backward()

		decoder = self.get_decoder(layer_index)
		if isinstance(decoder, L.QRNNGlobalAttentiveDecoder):
			out_data = decoder(in_data, encoder_last_hidden_states, encoder_last_layer_outputs, encoder_skip_mask, test=test)
		elif isinstance(decoder, L.QRNNDecoder):
			out_data = decoder(in_data, encoder_last_hidden_states, test=test)
		else:
			raise Exception()

		if test:
			out_data.unchain_backward()

		return out_data

	def decode(self, X, encoder_last_hidden_states, encoder_last_layer_outputs, encoder_skip_mask=None, test=False, return_last=False):
		assert len(encoder_last_hidden_states) == self.num_layers
		batchsize = X.shape[0]
		seq_length = X.shape[1]
		enmbedding = self.decoder_embed(X)
		enmbedding = F.swapaxes(enmbedding, 1, 2)

		out_data = self._forward_decoder_layer(0, enmbedding, encoder_last_hidden_states[0], encoder_last_layer_outputs, encoder_skip_mask, test=test)
		in_data = [out_data]

		for layer_index in xrange(1, self.num_layers):
			out_data = self._forward_decoder_layer(layer_index, sum(in_data) if self.densely_connected else in_data[-1], encoder_last_hidden_states[layer_index], encoder_last_layer_outputs, encoder_skip_mask, test=test)
			in_data.append(out_data)

		out_data = sum(in_data) if self.densely_connected else out_data	# dense conv

		if return_last:
			out_data = out_data[:, :, -1, None]

		out_data = F.reshape(F.swapaxes(out_data, 1, 2), (-1, self.ndim_h))
		Y = self.dense(out_data)

		if test:
			Y.unchain_backward()

		return Y

	def _forward_decoder_layer_one_step(self, layer_index, in_data, encoder_last_hidden_states, encoder_last_layer_outputs, encoder_skip_mask=None, test=False):
		if test:
			in_data.unchain_backward()

		decoder = self.get_decoder(layer_index)
		if isinstance(decoder, L.QRNNGlobalAttentiveDecoder):
			out_data = decoder.forward_one_step(in_data, encoder_last_hidden_states, encoder_last_layer_outputs, encoder_skip_mask, test=test)
		elif isinstance(decoder, L.QRNNDecoder):
			out_data = decoder.forward_one_step(in_data, encoder_last_hidden_states, test=test)
		else:
			raise Exception()

		if test:
			out_data.unchain_backward()

		return out_data

	def decode_one_step(self, X, encoder_last_hidden_states, encoder_last_layer_outputs, encoder_skip_mask=None, test=False):
		assert len(encoder_last_hidden_states) == self.num_layers
		batchsize = X.shape[0]
		seq_length = X.shape[1]

		xt = X[:, -1, None]
		enmbedding = self.decoder_embed(xt)
		enmbedding = F.swapaxes(enmbedding, 1, 2)

		out_data = self._forward_decoder_layer_one_step(0, enmbedding, encoder_last_hidden_states[0], encoder_last_layer_outputs, encoder_skip_mask, test=test)
		in_data = [out_data]
		
		for layer_index in xrange(1, self.num_layers):
			out_data = self._forward_decoder_layer_one_step(layer_index, sum(in_data) if self.densely_connected else in_data[-1], encoder_last_hidden_states[layer_index], encoder_last_layer_outputs, encoder_skip_mask, test=test)
			in_data.append(out_data)

		out_data = sum(in_data) if self.densely_connected else out_data	# dense conv
		out_data = out_data[:, :, -1, None]
		out_data = F.reshape(F.swapaxes(out_data, 1, 2), (batchsize, self.ndim_h))
		Y = self.dense(out_data)

		if test:
			Y.unchain_backward()

		return Y
