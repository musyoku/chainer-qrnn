# encoding: utf-8
import chainer
from chainer import Variable, Chain
import numpy as np
from qrnn import QRNN, QRNNDecoder, QRNNAttentiveEncoder

shape = (2, 3, 10)
prod = shape[0] * shape[1] * shape[2]
data = np.arange(0, prod, dtype=np.float32).reshape(shape) / prod
data = Variable(data)

qrnn = QRNN(shape[1], 4, kernel_size=3, pooling="fo", zoneout=True)
y = qrnn(data)

dec = QRNNDecoder(shape[1], 4, kernel_size=3, pooling="fo", zoneout=True)
c, h = qrnn.get_state()
y = dec(data, h)

enc = QRNNAttentiveEncoder(shape[1], 4, kernel_size=3, pooling="ifo", zoneout=True)
y = enc(data)
print enc.get_state()[1].data

# Vanilla QRNN layers
class QRNNLayers(Chain):
	def __init__(self, num_vocab, ndim_embedding):
		super(RNNForLM, self).__init__(
			embed=L.EmbedID(num_vocab, n_units),
			l1=QRNN(ndim_embedding, 128, kernel_size=3, pooling="f"),
			l2=QRNN(128, 128, kernel_size=3, pooling="f"),
			l3=L.Linear(128, num_vocab),
		)
		for param in self.params():
			param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()

	def __call__(self, x):
		h0 = self.embed(x)
		h1 = self.l1(h0)
		h2 = self.l2(h1)
		y = self.l3(h2)
		return y


# QRNN Encoder-Decoder
class QRNNEncoderDecoder(Chain):
	def __init__(self, num_vocab, ndim_embedding):
		super(RNNForLM, self).__init__(
			embed=L.EmbedID(num_vocab, n_units),
			enc1=QRNN(ndim_embedding, 128, kernel_size=3, pooling="f"),
			enc2=QRNN(128, 128, kernel_size=3, pooling="f"),
			l3=L.Linear(128, num_vocab),
		)
		for param in self.params():
			param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()

	def __call__(self, x):
		h0 = self.embed(x)
		h1 = self.l1(h0)
		h2 = self.l2(h1)
		y = self.l3(h2)
		return y
