from __future__ import division
from __future__ import print_function
from six.moves import xrange
import math
import numpy as np
import chainer
from chainer import cuda, Variable, function, link, functions, links, initializers
from chainer.utils import type_check
from chainer.links import EmbedID, Linear, BatchNormalization
from convolution_1d import Convolution1D

class Zoneout(function.Function):
	def __init__(self, p):
		self.p = p

	def check_type_forward(self, in_types):
		type_check.expect(in_types.size() == 1)
		type_check.expect(in_types[0].dtype.kind == 'f')

	def forward(self, x):
		if not hasattr(self, "mask"):
			xp = cuda.get_array_module(*x)
			if xp == np:
				flag = xp.random.rand(*x[0].shape) >= self.p
			else:
				flag = xp.random.rand(*x[0].shape, dtype=np.float32) >= self.p
			self.mask = flag
		return x[0] * self.mask,

	def backward(self, x, gy):
		return gy[0] * self.mask,

def zoneout(x, ratio=.5):
	return Zoneout(ratio)(x)

class QRNN(link.Chain):
	def __init__(self, in_channels, out_channels, kernel_size=2, pooling="f", zoneout=False, zoneout_ratio=0.1, wgain=1., weightnorm=False):
		self.num_split = len(pooling) + 1
		if weightnorm:
			wstd = 0.05
			W = Convolution1D(in_channels, self.num_split * out_channels, kernel_size, stride=1, pad=kernel_size - 1, initialV=initializers.HeNormal(wstd))
		else:
			wstd = math.sqrt(wgain / in_channels / kernel_size)
			W = links.ConvolutionND(1, in_channels, self.num_split * out_channels, kernel_size, stride=1, pad=kernel_size - 1, initialW=initializers.HeNormal(wstd))

		super(QRNN, self).__init__(W=W)
		self._in_channels, self._out_channels, self._kernel_size, self._pooling, self._zoneout, self._zoneout_ratio = in_channels, out_channels, kernel_size, pooling, zoneout, zoneout_ratio
		self.reset_state()

	def __call__(self, X, skip_mask=None):
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
		WX = self.W(X)[:, :, :-pad]

		return self.pool(functions.split_axis(WX, self.num_split, axis=1), skip_mask=skip_mask)

	def forward_one_step(self, X, skip_mask=None):
		pad = self._kernel_size - 1
		WX = self.W(X)[:, :, -pad-1, None]
		return self.pool(functions.split_axis(WX, self.num_split, axis=1), skip_mask=skip_mask)

	def zoneout(self, U):
		if self._zoneout and chainer.config.train:
			return 1 - zoneout(functions.sigmoid(-U), self._zoneout_ratio)
		return functions.sigmoid(U)

	def pool(self, WX, skip_mask=None):
		Z, F, O, I = None, None, None, None

		# f-pooling
		if len(self._pooling) == 1:
			assert len(WX) == 2
			Z, F = WX
			Z = functions.tanh(Z)
			F = self.zoneout(F)

		# fo-pooling
		if len(self._pooling) == 2:
			assert len(WX) == 3
			Z, F, O = WX
			Z = functions.tanh(Z)
			F = self.zoneout(F)
			O = functions.sigmoid(O)

		# ifo-pooling
		if len(self._pooling) == 3:
			assert len(WX) == 4
			Z, F, O, I = WX
			Z = functions.tanh(Z)
			F = self.zoneout(F)
			O = functions.sigmoid(O)
			I = functions.sigmoid(I)

		assert Z is not None
		assert F is not None

		T = Z.shape[2]
		for t in xrange(T):
			zt = Z[:, :, t]
			ft = F[:, :, t]
			ot = 1 if O is None else O[:, :, t]
			it = 1 - ft if I is None else I[:, :, t]
			xt = 1 if skip_mask is None else skip_mask[:, t, None]	# will be used for seq2seq to skip PAD

			if self.ct is None:
				self.ct = (1 - ft) * zt * xt
			else:
				self.ct = ft * self.ct + it * zt * xt
			self.ht = self.ct if O is None else ot * self.ct


			if self.H is None:
				self.H = functions.expand_dims(self.ht, 2)
			else:
				self.H = functions.concat((self.H, functions.expand_dims(self.ht, 2)), axis=2)

		return self.H

	def reset_state(self):
		self.set_state(None, None, None)

	def set_state(self, ct, ht, H):
		self.ct = ct	# last cell state
		self.ht = ht	# last hidden state
		self.H = H		# all hidden states

	def get_last_hidden_state(self):
		return self.ht

	def get_all_hidden_states(self):
		return self.H

class QRNNEncoder(QRNN):
	pass

class QRNNDecoder(QRNN):
	def __init__(self, in_channels, out_channels, kernel_size=2, pooling="f", zoneout=False, zoneout_ratio=0.1, wgain=1., weightnorm=False):
		super(QRNNDecoder, self).__init__(in_channels, out_channels, kernel_size, pooling, zoneout, zoneout_ratio, wgain, weightnorm)
		self.num_split = len(pooling) + 1
		wstd = math.sqrt(wgain / in_channels / kernel_size)
		self.add_link("V", links.Linear(out_channels, self.num_split * out_channels, initialW=initializers.Normal(wstd)))

	# ht_enc is the last encoder state
	def __call__(self, X, ht_enc):
		pad = self._kernel_size - 1
		WX = self.W(X)
		if pad > 0:
			WX = WX[:, :, :-pad]
		Vh = self.V(ht_enc)

		# copy Vh
		# e.g.
		# WX = [[[  0	1	2]
		# 		 [	3	4	5]
		# 		 [	6	7	8]
		# Vh = [[11, 12, 13]]
		# 
		# Vh, WX = F.broadcast(F.expand_dims(Vh, axis=2), WX)
		# 
		# WX = [[[  0	1	2]
		# 		 [	3	4	5]
		# 		 [	6	7	8]
		# Vh = [[[ 	11	11	11]
		# 		 [	12	12	12]
		# 		 [	13	13	13]
		Vh, WX = functions.broadcast(functions.expand_dims(Vh, axis=2), WX)

		return self.pool(functions.split_axis(WX + Vh, self.num_split, axis=1))

	def forward_one_step(self, X, ht_enc):
		pad = self._kernel_size - 1
		WX = self.W(X)[:, :, -pad-1, None]
		Vh = self.V(ht_enc)

		Vh, WX = functions.broadcast(functions.expand_dims(Vh, axis=2), WX)

		return self.pool(functions.split_axis(WX + Vh, self.num_split, axis=1))

class QRNNGlobalAttentiveDecoder(QRNNDecoder):
	def __init__(self, in_channels, out_channels, kernel_size=2, zoneout=False, zoneout_ratio=0.1, wgain=1., weightnorm=False):
		super(QRNNGlobalAttentiveDecoder, self).__init__(in_channels, out_channels, kernel_size, "fo", zoneout, zoneout_ratio, wgain, weightnorm)
		wstd = math.sqrt(wgain / in_channels / kernel_size)
		self.add_link('o', links.Linear(2 * out_channels, out_channels, initialW=initializers.Normal(wstd)))

	# X is the input of the decoder
	# ht_enc is the last encoder state
	# H_enc is the encoder's las layer's hidden sates
	def __call__(self, X, ht_enc, H_enc, skip_mask=None):
		pad = self._kernel_size - 1
		WX = self.W(X)
		if pad > 0:
			WX = WX[:, :, :-pad]
		Vh = self.V(ht_enc)
		Vh, WX = functions.broadcast(functions.expand_dims(Vh, axis=2), WX)

		# f-pooling
		Z, F, O = functions.split_axis(WX + Vh, 3, axis=1)
		Z = functions.tanh(Z)
		F = self.zoneout(F)
		O = functions.sigmoid(O)
		T = Z.shape[2]

		# compute ungated hidden states
		self.contexts = []
		for t in xrange(T):
			z = Z[:, :, t]
			f = F[:, :, t]
			if t == 0:
				ct = (1 - f) * z
				self.contexts.append(ct)
			else:
				ct = f * self.contexts[-1] + (1 - f) * z
				self.contexts.append(ct)

		if skip_mask is not None:
			assert skip_mask.shape[1] == H_enc.shape[2]
			softmax_getas = (skip_mask == 0) * -1e6

		# compute attention weights (eq.8)
		H_enc = functions.swapaxes(H_enc, 1, 2)
		for t in xrange(T):
			ct = self.contexts[t]
			geta = 0 if skip_mask is None else softmax_getas[..., None]	# to skip PAD
			mask = 1 if skip_mask is None else skip_mask[..., None]		# to skip PAD
			alpha = functions.batch_matmul(H_enc, ct) + geta
			alpha = functions.softmax(alpha) * mask
			alpha = functions.broadcast_to(alpha, H_enc.shape)	# copy
			kt = functions.sum(alpha * H_enc, axis=1)
			ot = O[:, :, t]
			self.ht = ot * self.o(functions.concat((kt, ct), axis=1))

			if t == 0:
				self.H = functions.expand_dims(self.ht, 2)
			else:
				self.H = functions.concat((self.H, functions.expand_dims(self.ht, 2)), axis=2)

		return self.H

	def forward_one_step(self, X, ht_enc, H_enc, skip_mask):
		pad = self._kernel_size - 1
		WX = self.W(X)[:, :, -pad-1, None]
		Vh = self.V(ht_enc)

		Vh, WX = functions.broadcast(functions.expand_dims(Vh, axis=2), WX)

		# f-pooling
		Z, F, O = functions.split_axis(WX + Vh, 3, axis=1)
		Z = functions.tanh(Z)
		F = self.zoneout(F)
		O = functions.sigmoid(O)
		T = Z.shape[2]

		# compute ungated hidden states
		for t in xrange(T):
			z = Z[:, :, t]
			f = F[:, :, t]
			if self.contexts is None:
				ct = (1 - f) * z
				self.contexts = [ct]
			else:
				ct = f * self.contexts[-1] + (1 - f) * z
				self.contexts.append(ct)

		if skip_mask is not None:
			assert skip_mask.shape[1] == H_enc.shape[2]
			softmax_getas = (skip_mask == 0) * -1e6

		# compute attention weights (eq.8)
		H_enc = functions.swapaxes(H_enc, 1, 2)
		for t in xrange(T):
			ct = self.contexts[t - T]
			geta = 0 if skip_mask is None else softmax_getas[..., None]	# to skip PAD
			mask = 1 if skip_mask is None else skip_mask[..., None]		# to skip PAD
			alpha = functions.batch_matmul(H_enc, ct) + geta
			alpha = functions.softmax(alpha) * mask
			alpha = functions.broadcast_to(alpha, H_enc.shape)	# copy
			kt = functions.sum(alpha * H_enc, axis=1)
			ot = O[:, :, t]
			self.ht = ot * self.o(functions.concat((kt, ct), axis=1))

			if self.H is None:
				self.H = functions.expand_dims(self.ht, 2)
			else:
				self.H = functions.concat((self.H, functions.expand_dims(self.ht, 2)), axis=2)

		return self.H


	def reset_state(self):
		self.set_state(None, None, None, None)

	def set_state(self, ct, ht, H, contexts):
		self.ct = ct	# last cell state
		self.ht = ht	# last hidden state
		self.H = H		# all hidden states
		self.contexts = contexts
