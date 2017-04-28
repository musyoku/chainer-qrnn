import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer import cuda, Function, Variable, Chain, function, link, functions
from chainer.utils import type_check
from chainer.functions import sigmoid, tanh, expand_dims, concat


def attention_sum(encoding, query):
	alpha = F.softmax(F.batch_matmul(encoding, query, transb=True))
	alpha, encoding = F.broadcast(alpha[:, :, :, None],
								  encoding[:, :, None, :])
	return F.sum(alpha * encoding, axis=1)


class Linear(L.Linear):

	def __call__(self, x):
		shape = x.shape
		if len(shape) == 3:
			x = F.reshape(x, (-1, shape[2]))
		y = super().__call__(self, x)
		if len(shape) == 3:
			y = F.reshape(y, shape)
		return y


class Zoneout(function.Function):
	def __init__(self, zoneout_ratio):
		self.zoneout_ratio = zoneout_ratio

	def check_type_forward(self, in_types):
		type_check.expect(in_types.size() == 1)
		type_check.expect(in_types[0].dtype.kind == 'f')

	def forward(self, x):
		if not hasattr(self, 'mask'):
			xp = cuda.get_array_module(*x)
			if xp == np:
				flag = xp.random.rand(*x[0].shape) >= self.zoneout_ratio
			else:
				flag = (xp.random.rand(*x[0].shape, dtype=np.float32) >= self.zoneout_ratio)
			self.mask = flag
		return x[0] * self.mask,

	def backward(self, x, gy):
		return gy[0] * self.mask,

def zoneout(x, ratio=.5, train=True):
	if train:
		return Zoneout(ratio)(x)
	return x

class QRNN(link.Chain):
	def __init__(self, in_channels, out_channels, kernel_size=2, pooling="f", zoneout=False, zoneout_ratio=0.5):
		self.num_split = len(pooling) + 1
		super(QRNN, self).__init__(W=L.ConvolutionND(1, in_channels, self.num_split * out_channels, kernel_size, stride=1, pad=kernel_size - 1))
		self._in_channels, self._out_channels, self._kernel_size, self._pooling, self._zoneout, self._zoneout_ratio = in_channels, out_channels, kernel_size, pooling, zoneout, zoneout_ratio
		self.reset_state()

	def __call__(self, x, test=False):
		assert isinstance(x, Variable)
		self._test = test
		# remove right paddings
		# e.g.
		# kernel_size = 3
		# pad = 2
		# input sequence with paddings:
		# [0, 0, x1, x2, x3, 0, 0]
		# |< t1 >|
		#     |< t2 >|
		#         |< t3 >|
		pad = self._kernel_size - 1
		Wx = self.W(x)[:, :, :-pad]
		return self.pool(F.split_axis(Wx, self.num_split, axis=1))

	def zoneout(self, u):
		if self._zoneout:
			return 1 - zoneout(F.sigmoid(-u), ratio=self._zoneout_ratio, train=not self._test)
		return F.sigmoid(u)

	def pool(self, Wx):
		# f-pooling
		if len(self._pooling) == 1:
			assert len(Wx) == 2
			Z, F = Wx
			Z = tanh(Z)
			F = self.zoneout(F)
			for t in xrange(Z.shape[2]):
				z = Z[:, :, t]
				f = F[:, :, t]
				if t == 0:
					self.h = (1 - f) * z
				else:
					self.h = f * self.h + (1 - f) * z
			return self.h

		# fo-pooling
		if len(self._pooling) == 2:
			assert len(Wx) == 3
			Z, F, O = Wx
			Z = tanh(Z)
			F = self.zoneout(F)
			O = sigmoid(O)
			for t in xrange(Z.shape[2]):
				z = Z[:, :, t]
				f = F[:, :, t]
				o = F[:, :, t]
				if t == 0:
					self.c = (1 - f) * z
				else:
					self.c = f * self.c + (1 - f) * z
				self.h = o * self.c
			return self.h

		# ifo-pooling
		if len(self._pooling) == 3:
			assert len(Wx) == 4
			Z, F, O, I = Wx
			Z = tanh(Z)
			F = self.zoneout(F)
			O = sigmoid(O)
			I = sigmoid(I)
			for t in xrange(Z.shape[2]):
				z = Z[:, :, t]
				f = F[:, :, t]
				o = F[:, :, t]
				i = F[:, :, t]
				if t == 0:
					self.c = (1 - f) * z
				else:
					self.c = f * self.c + i * z
				self.h = o * self.c
			return self.h

		raise Exception()

	def reset_state(self):
		self.set_state(None, None)

	def set_state(self, c, h):
		self.c = c
		self.h = h

	def get_state(self):
		return self.c, self.h

class QRNNDecoder(QRNN):
	def __init__(self, in_channels, out_channels, kernel_size=2, pooling="f", zoneout=False, zoneout_ratio=0.5):
		super(QRNNDecoder, self).__init__(in_channels, out_channels, kernel_size, pooling, zoneout, zoneout_ratio)
		self.num_split = len(pooling) + 1
		self.add_link("V", L.Linear(out_channels, self.num_split * out_channels))
		self.add_link('o', L.Linear(2 * out_channels, out_channels))

	# x is the input of the decoder
	# h is the final encoder hidden state
	def __call__(self, x, h, test=False):
		assert isinstance(x, Variable)
		self._test = test
		pad = self._kernel_size - 1
		Wx = self.W(x)[:, :, :-pad]
		Vh = self.V(h)
		# copy Vh
		# e.g.
		# Wx = [[[  0	1	2]
		# 		 [	3	4	5]
		# 		 [	6	7	8]
		# Vh = [[11, 12, 13]]
		# 
		# Vh, Wx = F.broadcast(F.expand_dims(Vh, axis=2), Wx)
		# 
		# Wx = [[[  0	1	2]
		# 		 [	3	4	5]
		# 		 [	6	7	8]
		# Vh = [[[ 	11	11	11]
		# 		 [	12	12	12]
		# 		 [	13	13	13]
		Vh, Wx = F.broadcast(F.expand_dims(Vh, axis=2), Wx)
		return self.pool(F.split_axis(Wx + Vh, self.num_split, axis=1))

# QRNNAttentiveEncoder preserves all hidden states
class QRNNAttentiveEncoder(QRNN):
	def __init__(self, in_channels, out_channels, kernel_size=2, pooling="f", zoneout=False, zoneout_ratio=0.5):
		super(QRNNAttentiveEncoder, self).__init__(in_channels, out_channels, kernel_size, pooling, zoneout, zoneout_ratio)

	def pool(self, Wx):
		# f-pooling
		if len(self._pooling) == 1:
			assert len(Wx) == 2
			Z, F = Wx
			Z = tanh(Z)
			F = self.zoneout(F)
			for t in xrange(Z.shape[2]):
				z = Z[:, :, t]
				f = F[:, :, t]
				if t == 0:
					self.H = expand_dims((1 - f) * z, 1)
				else:
					h = f * self.H[:, -1, :] + (1 - f) * z
					h = expand_dims(h, 1)
					self.H = concat((self.H, h), axis=1)
			return self.H

		# fo-pooling
		if len(self._pooling) == 2:
			assert len(Wx) == 3
			Z, F, O = Wx
			Z = tanh(Z)
			F = self.zoneout(F)
			O = sigmoid(O)
			for t in xrange(Z.shape[2]):
				z = Z[:, :, t]
				f = F[:, :, t]
				o = F[:, :, t]
				if t == 0:
					self.c = (1 - f) * z
				else:
					self.c = f * self.c + (1 - f) * z
				h = o * self.c
				h = expand_dims(h, 1)
				if t == 0:
					self.H = h
				else:
					self.H = concat((self.H, h), axis=1)
			return self.H

		# ifo-pooling
		if len(self._pooling) == 3:
			assert len(Wx) == 4
			Z, F, O, I = Wx
			Z = tanh(Z)
			F = self.zoneout(F)
			O = sigmoid(O)
			I = sigmoid(I)
			for t in xrange(Z.shape[2]):
				z = Z[:, :, t]
				f = F[:, :, t]
				o = F[:, :, t]
				i = F[:, :, t]
				if t == 0:
					self.c = (1 - f) * z
				else:
					self.c = f * self.c + i * z
				h = o * self.c
				h = expand_dims(h, 1)
				if t == 0:
					self.H = h
				else:
					self.H = concat((self.H, h), axis=1)
			return self.H

		raise Exception()

	def set_state(self, c, H):
		self.c = c
		self.H = H

	def get_state(self):
		return self.c, self.H

class QRNNAttentiveDecoder(QRNN):
	def __init__(self, in_channels, out_channels, kernel_size=2, pooling="f", zoneout=False, zoneout_ratio=0.5):
		super(QRNNDecoder, self).__init__(in_channels, out_channels, kernel_size, pooling, zoneout, zoneout_ratio)
		self.num_split = len(pooling) + 1
		self.add_link("V", L.Linear(out_channels, self.num_split * out_channels))
		self.add_link('o', L.Linear(2 * out_channels, out_channels))

	# x is the input of the decoder
	# h is the final encoder hidden state
	def __call__(self, x, h, test=False):
		assert isinstance(x, Variable)
		self._test = test
		pad = self._kernel_size - 1
		Wx = self.W(x)[:, :, :-pad]
		Vh = self.V(h)
		# copy Vh
		# e.g.
		# Wx = [[[  0	1	2]
		# 		 [	3	4	5]
		# 		 [	6	7	8]
		# Vh = [[11, 12, 13]]
		# 
		# Vh, Wx = F.broadcast(F.expand_dims(Vh, axis=2), Wx)
		# 
		# Wx = [[[  0	1	2]
		# 		 [	3	4	5]
		# 		 [	6	7	8]
		# Vh = [[[ 	11	11	11]
		# 		 [	12	12	12]
		# 		 [	13	13	13]
		Vh, Wx = F.broadcast(F.expand_dims(Vh, axis=2), Wx)
		return self.pool(F.split_axis(Wx + Vh, self.num_split, axis=1))

class QRNNLayer(Chain):

	def __init__(self, in_size, out_size, kernel_size=2, attention=False,
				 decoder=False):
		if kernel_size == 1:
			super().__init__(W=Linear(in_size, 3 * out_size))
		elif kernel_size == 2:
			super().__init__(W=Linear(in_size, 3 * out_size, nobias=True),
							 V=Linear(in_size, 3 * out_size))
		else:
			super().__init__(
				conv=L.ConvolutionND(1, in_size, 3 * out_size, kernel_size,
									 stride=1, pad=kernel_size - 1))
		if attention:
			self.add_link('U', Linear(out_size, 3 * in_size))
			self.add_link('o', Linear(2 * out_size, out_size))
		self.in_size, self.size, self.attention = in_size, out_size, attention
		self.kernel_size = kernel_size

	def pre(self, x):
		dims = len(x.shape) - 1

		if self.kernel_size == 1:
			ret = self.W(x)
		elif self.kernel_size == 2:
			if dims == 2:
				xprev = Variable(
					self.xp.zeros((self.batch_size, 1, self.in_size),
								  dtype=np.float32), volatile='AUTO')
				xtminus1 = F.concat((xprev, x[:, :-1, :]), axis=1)
			else:
				xtminus1 = self.x
			ret = self.W(x) + self.V(xtminus1)
		else:
			ret = F.swapaxes(self.conv(F.swapaxes(x, 1, 2))[:, :, :x.shape[2]], 1, 2)

		if not self.attention:
			return ret

		if dims == 1:
			enc = self.encoding[:, -1, :]
		else:
			enc = self.encoding[:, -1:, :]
		return sum(F.broadcast(self.U(enc), ret))

	def init(self, encoder_c=None, encoder_h=None):
		self.encoding = encoder_c
		self.c, self.x = None, None
		if self.encoding is not None:
			self.batch_size = self.encoding.shape[0]
			if not self.attention:
				self.c = self.encoding[:, -1, :]

		if self.c is None or self.c.shape[0] < self.batch_size:
			self.c = Variable(self.xp.zeros((self.batch_size, self.size),
											dtype=np.float32), volatile='AUTO')

		if self.x is None or self.x.shape[0] < self.batch_size:
			self.x = Variable(self.xp.zeros((self.batch_size, self.in_size),
											dtype=np.float32), volatile='AUTO')

	def __call__(self, x):
		if not hasattr(self, 'encoding') or self.encoding is None:
			self.batch_size = x.shape[0]
			self.init()
		dims = len(x.shape) - 1
		f, z, o = F.split_axis(self.pre(x), 3, axis=dims)
		f = F.sigmoid(f)
		z = (1 - f) * F.tanh(z)
		o = F.sigmoid(o)

		if dims == 2:
			self.c = strnn(f, z, self.c[:self.batch_size])
		else:
			self.c = f * self.c + z

		if self.attention:
			context = attention_sum(self.encoding, self.c)
			self.h = o * self.o(F.concat((self.c, context), axis=dims))
		else:
			self.h = self.c * o

		self.x = x
		return self.h

	def get_state(self):
		return F.concat((self.x, self.c, self.h), axis=1)

	def set_state(self, state):
		self.x, self.c, self.h = F.split_axis(
			state, (self.in_size, self.in_size + self.size), axis=1)

	state = property(get_state, set_state)