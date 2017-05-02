# encoding: utf-8
from __future__ import division
from __future__ import print_function
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable, Chain
from qrnn import QRNN, QRNNEncoder, QRNNDecoder, QRNNGlobalAttentiveDecoder

class Encoder(Chain):
	def __init__(self, num_vocab, ndim_embedding):
		super(Encoder, self).__init__(
			embed=L.EmbedID(num_vocab, ndim_embedding),
			l1=QRNNEncoder(ndim_embedding, 128, kernel_size=3, pooling="fo"),
			l2=QRNNEncoder(128, 128, kernel_size=3, pooling="fo"),
			l3=L.Linear(128, num_vocab),
		)
		for param in self.params():
			param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()

	def __call__(self, x):
		h0 = self.embed(x)
		h0 = F.swapaxes(h0, 1, 2)
		self.l1(h0)
		h1 = self.l1.get_all_hidden_states()
		self.l2(h1)
		h2 = self.l2.get_last_hidden_state()
		y = self.l3(h2)
		return y


class EncoderDecoder(Chain):
	def __init__(self, num_vocab, ndim_embedding):
		super(EncoderDecoder, self).__init__(
			embed=L.EmbedID(num_vocab, n_units),
			enc1=QRNNEncoder(ndim_embedding, 128, kernel_size=3, pooling="f"),
			enc2=QRNNEncoder(128, 128, kernel_size=3, pooling="f"),
			dec1=QRNNDecoder(128, 128, kernel_size=3, pooling="f"),
			dec2=QRNNDecoder(128, 128, kernel_size=3, pooling="f"),
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

np.random.seed(0)
shape = (2, 3, 6)
prod = shape[0] * shape[1] * shape[2]
data = np.arange(0, prod, dtype=np.float32).reshape(shape) / prod
skip_mask = np.ones((data.shape[0], data.shape[2]), dtype=np.float32)
skip_mask[:, :1] = 0

data[:, :, 4] = "inf"
qrnn1 = QRNN(shape[1], 4, kernel_size=4, pooling="fo", zoneout=True, zoneout_ratio=0.9)
qrnn2 = QRNN(4, 4, kernel_size=4, pooling="fo", zoneout=True)
dense = L.Linear(4, 2)
y = qrnn1(data, skip_mask=skip_mask)
print(y.data)
# qrnn1.reset_state()
# y = qrnn1(data[:, :, 2:], skip_mask=None)
# print(y.data)
y = qrnn2(y)
print(y.data)
y = F.reshape(F.swapaxes(y, 1, 2), (2 * 6, -1))
print(y.data)
y = dense(y)
print(y.data)


# qrnn = QRNN(shape[1], 4, kernel_size=3, pooling="fo", zoneout=True)
# y = qrnn(data)

# qrnn = QRNN(shape[1], 4, kernel_size=3, pooling="ifo", zoneout=True)
# y = qrnn(data)

# dec = QRNNDecoder(shape[1], 4, kernel_size=3, pooling="fo", zoneout=True)
# h = qrnn.get_last_hidden_state()
# y = dec(data, h)

# dec = QRNNGlobalAttentiveDecoder(shape[1], 4, kernel_size=3, zoneout=True)
# h = qrnn.get_last_hidden_state()
# H = qrnn.get_all_hidden_states()
# y = dec(data, h, H)

# num_vocab = 10
# ndim_embedding = 200
# shape = (16, 100)
# prod = shape[0] * shape[1]
# data = np.arange(0, prod, dtype=np.int32).reshape(shape) / prod
# data = Variable(data)
# enc = Encoder(num_vocab, ndim_embedding)
# y = enc(data)
# print(y.data)