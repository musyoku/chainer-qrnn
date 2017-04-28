# encoding: utf-8
import chainer
from chainer import Variable, Chain
import numpy as np
from qrnn import QRNN

shape = (1, 3, 4)
data = np.arange(0, shape[0] * shape[1] * shape[2], dtype=np.float32).reshape(shape)
data = Variable(data)

np.random.seed(0)
qrnn = QRNN(shape[1], 3, kernel_size=3, pooling="f")
y = qrnn(data)
y = qrnn(data)
np.random.seed(0)
qrnn = QRNN(shape[1], 3, kernel_size=3, pooling="f", zoneout=True)
y = qrnn(data)
y = qrnn(data)

# Vanilla QRNN layers
class QRNNLayers(Chain):
	def __init__(self, num_vocab, ndim_embedding):
		super(RNNForLM, self).__init__(
			embed=L.EmbedID(num_vocab, n_units),
			l1=QRNN(ndim_embedding, 64, kernel_size=3, pooling="f"),
			l2=QRNN(64, 128, kernel_size=3, pooling="f"),
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
