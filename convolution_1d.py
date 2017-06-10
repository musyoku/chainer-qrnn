from __future__ import division
import math
import numpy as np
from six import moves
from chainer import cuda, Variable, Parameter, initializers, link, functions
from chainer.functions.connection import convolution_nd
from chainer.utils import conv_nd, type_check

def _get_norm(W):
	xp = cuda.get_array_module(W)
	norm = xp.sqrt(xp.sum(W ** 2, axis=(1, 2))) + 1e-32
	norm = norm.reshape((-1, 1, 1))
	return norm

class Convolution1DFunction(convolution_nd.ConvolutionND):
		
	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(2 <= n_in, n_in <= 4)

		x_type = in_types[0]
		v_type = in_types[1]
		g_type = in_types[1]
		type_check.expect(
			x_type.dtype.kind == "f",
			v_type.dtype.kind == "f",
			g_type.dtype.kind == "f",
			x_type.ndim == self.ndim + 2,
			v_type.ndim == self.ndim + 2,
			g_type.ndim == self.ndim + 2,
			x_type.shape[1] == v_type.shape[1],
		)

		if type_check.eval(n_in) == 4:
			b_type = in_types[3]
			type_check.expect(
				b_type.dtype == x_type.dtype,
				b_type.ndim == 1,
				b_type.shape[0] == v_type.shape[0],
			)
			
	def forward(self, inputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None
		
		self.norm = _get_norm(V)
		self.V_normalized = V / self.norm
		self.W = g * self.V_normalized

		if b is None:
			return super(Convolution1DFunction, self).forward((x, self.W))
		return super(Convolution1DFunction, self).forward((x, self.W, b))

	def backward(self, inputs, grad_outputs):
		x, V, g = inputs[:3]
		if hasattr(self, "W") == False:
			self.norm = _get_norm(V)
			self.V_normalized = V / self.norm
			self.W = g * self.V_normalized

		b = inputs[3] if len(inputs) == 4 else None
		if b is None:
			gx, gW = super(Convolution1DFunction, self).backward((x, self.W), grad_outputs)
		else:
			gx, gW, gb = super(Convolution1DFunction, self).backward((x, self.W, b), grad_outputs)

		xp = cuda.get_array_module(x)
		gg = xp.sum(gW * self.V_normalized, axis=(1, 2), keepdims=True)
		gV = g * (gW - gg * self.V_normalized) / self.norm

		if b is None:
			return gx, gV, gg
		else:
			return gx, gV, gg, gb
			
def convolution_1d(x, V, g, b=None, stride=1, pad=0, cover_all=False):
	func = Convolution1DFunction(1, stride=stride, pad=pad, cover_all=cover_all)
	if b is None:
		return func(x, V, g)
	else:
		return func(x, V, g, b)

class Convolution1D(link.Link):

	def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, initialV=None, nobias=False, cover_all=False):
		super(Convolution1D, self).__init__()
		ksize = conv_nd.as_tuple(ksize, 1)
		self.ksize = ksize
		self.nobias = nobias
		self.stride = stride
		self.pad = pad
		self.out_channels = out_channels
		self.in_channels = in_channels
		self.cover_all = cover_all

		self.initialV = initialV

		with self.init_scope():
			V_shape = (out_channels, in_channels) + ksize
			initialV = initializers._get_initializer(initialV)
			self.V = Parameter(initialV, V_shape)

		if nobias:
			self.b = None

	@property
	def W(self):
		V = self.V.data
		xp = cuda.get_array_module(V)
		norm = _get_norm(V)
		V = V / norm
		return self.g.data * V

	# data-dependent initialization of parameters
	def _initialize_params(self, t):
		xp = cuda.get_array_module(t)

		self.mean_t = xp.mean(t, axis=(0, 2))			# calculate average for each channel
		self.std_t = xp.sqrt(xp.var(t, axis=(0, 2)))	# calculate stddev for each channel
		g = 1 / self.std_t
		b = -self.mean_t / self.std_t
		
		# print("g <- {}, b <- {}".format(g.reshape((-1,)), b.reshape((-1,))))

		with self.init_scope():
			if self.nobias == False:
				self.b = Parameter(b, b.shape)

			g_shape = (self.out_channels, 1) + (1,) * len(self.ksize)
			self.g = Parameter(g.reshape(g_shape), g_shape)

	def __call__(self, x):

		if hasattr(self, "b") == False or hasattr(self, "g") == False:
			xp = cuda.get_array_module(x.data)
			t = convolution_1d(x, self.V, Variable(xp.full((self.out_channels, 1, 1), 1.0).astype(x.dtype)), None, self.stride, self.pad)	# compute output with g = 1 and without bias
			self._initialize_params(t.data)
			return (t - self.mean_t.reshape(1, -1, 1)) / self.std_t.reshape((1, -1, 1))

		return convolution_1d(x, self.V, self.g, self.b, self.stride, self.pad, cover_all=self.cover_all)