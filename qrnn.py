import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer import cuda, Function, Variable, Chain, function
from chainer.utils import type_check

THREADS_PER_BLOCK = 32

class STRNNFunction(Function):

	def forward_gpu(self, inputs):
		f, z, hinit = inputs
		b, t, c = f.shape
		assert c % THREADS_PER_BLOCK == 0
		self.h = cuda.cupy.zeros((b, t + 1, c), dtype=np.float32)
		self.h[:, 0, :] = hinit
		cuda.raw('''
			#define THREADS_PER_BLOCK 32
			extern "C" __global__ void strnn_fwd(
					const CArray<float, 3> f, const CArray<float, 3> z,
					CArray<float, 3> h) {
				int index[3];
				const int t_size = f.shape()[1];
				index[0] = blockIdx.x;
				index[1] = 0;
				index[2] = blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;
				float prev_h = h[index];
				for (int i = 0; i < t_size; i++){
					index[1] = i;
					const float ft = f[index];
					const float zt = z[index];
					index[1] = i + 1;
					float &ht = h[index];
					prev_h = prev_h * ft + zt;
					ht = prev_h;
				}
			}''', 'strnn_fwd')(
				(b, c // THREADS_PER_BLOCK), (THREADS_PER_BLOCK,),
				(f, z, self.h))
		return self.h[:, 1:, :],

	def backward_gpu(self, inputs, grads):
		f, z = inputs[:2]
		gh, = grads
		b, t, c = f.shape
		gz = cuda.cupy.zeros_like(gh)
		cuda.raw('''
			#define THREADS_PER_BLOCK 32
			extern "C" __global__ void strnn_back(
				const CArray<float, 3> f, const CArray<float, 3> gh,
				CArray<float, 3> gz) {
				int index[3];
				const int t_size = f.shape()[1];
				index[0] = blockIdx.x;
				index[2] = blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;
				index[1] = t_size - 1;
				float &gz_last = gz[index];
				gz_last = gh[index];
				float prev_gz = gz_last;
				for (int i = t_size - 1; i > 0; i--){
					index[1] = i;
					const float ft = f[index];
					index[1] = i - 1;
					const float ght = gh[index];
					float &gzt = gz[index];
					prev_gz = prev_gz * ft + ght;
					gzt = prev_gz;
				}
			}''', 'strnn_back')(
				(b, c // THREADS_PER_BLOCK), (THREADS_PER_BLOCK,),
				(f, gh, gz))
		gf = self.h[:, :-1, :] * gz
		ghinit = f[:, 0, :] * gz[:, 0, :]
		return gf, gz, ghinit


def strnn(f, z, h0):
	return STRNNFunction()(f, z, h0)


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

class QRNN(L.ConvolutionND):
	def __init__(self, in_channels, out_channels, kernel_size=2, pooling="f", zoneout=False, zoneout_ratio=0.5):
		self.num_split = len(pooling) + 1
		super(QRNN, self).__init__(1, in_channels, self.num_split * out_channels, kernel_size, stride=1, pad=kernel_size - 1)
		self._in_channels, self._out_channels, self._kernel_size, self._pooling, self._zoneout, self._zoneout_ratio = in_channels, out_channels, kernel_size, pooling, zoneout, zoneout_ratio
		self.reset_state()

	def __call__(self, x, test=False):
		assert isinstance(x, Variable)
		self._test = test
		return self.pool(F.split_axis(super(QRNN, self).__call__(x), self.num_split, axis=2))

	def zoneout(self, u):
		if self._zoneout:
			return 1 - zoneout(F.sigmoid(-u), ratio=self._zoneout_ratio, train=not self._test)
		return F.sigmoid(u)

	def pool(self, conv):
		# f-pooling
		if self._pooling == "f":
			assert len(conv) == 2
			z, f = conv
			z = F.tanh(z)
			f = self.zoneout(f)
			print f.data
			if self.h is None:
				self.h = (1 - f) * z
			else:
				self.h = f * self.h + (1 - f) * z
			return self.h

		# fo-pooling
		if self._pooling == "fo":
			assert len(conv) == 3
			z, f, o = conv
			z = F.tanh(z)
			f = self.zoneout(f)
			o = F.sigmoid(o)
			if self.c is None:
				self.c = (1 - f) * z
			else:
				self.c = f * self.c + (1 - f) * z
			self.h = o * self.c
			return self.h

		# ifo-pooling
		if self._pooling == "ifo":
			assert len(conv) == 4
			z, f, o, i = conv
			z = F.tanh(z)
			f = self.zoneout(f)
			o = F.sigmoid(o)
			i = F.sigmoid(i)
			if self.c is None:
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
			ret = F.swapaxes(self.conv(
				F.swapaxes(x, 1, 2))[:, :, :x.shape[2]], 1, 2)

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