# coding: utf-8
import math, sys, argparse, six
import numpy as np
import chainer.functions as F
import chainer
from chainer import cuda, function
from chainer.utils import type_check
from chainer.functions.activation import log_softmax
from dataset import sample_batch_from_bucket, make_source_target_pair, read_data, make_buckets
from common import ID_UNK, ID_PAD, ID_BOS, ID_EOS, stdout, print_bold, bucket_sizes
from model import load_model

def _broadcast_to(array, shape):
	if hasattr(numpy, "broadcast_to"):
		return numpy.broadcast_to(array, shape)
	dummy = numpy.empty(shape, array.dtype)
	return numpy.broadcast_arrays(array, dummy)[0]


class SoftmaxCrossEntropy(function.Function):

	normalize = True

	def __init__(self, use_cudnn=True, normalize=True, cache_score=True,
				 class_weight=None, ignore_label=-1, reduce='mean'):
		self.use_cudnn = use_cudnn
		self.normalize = normalize
		self.cache_score = cache_score
		self.class_weight = class_weight
		if class_weight is not None:
			if self.class_weight.ndim != 1:
				raise ValueError('class_weight.ndim should be 1')
			if self.class_weight.dtype.kind != 'f':
				raise ValueError('The dtype of class_weight should be \'f\'')
			if isinstance(self.class_weight, chainer.Variable):
				raise ValueError('class_weight should be a numpy.ndarray or '
								 'cupy.ndarray, not a chainer.Variable')
		self.ignore_label = ignore_label
		if reduce not in ('mean', 'no'):
			raise ValueError(
				"only 'mean' and 'no' are valid for 'reduce', but '%s' is "
				'given' % reduce)
		self.reduce = reduce

	def check_type_forward(self, in_types):
		type_check.expect(in_types.size() == 2)
		x_type, t_type = in_types

		type_check.expect(
			x_type.dtype.kind == 'f',
			t_type.dtype == np.int32,
			t_type.ndim == x_type.ndim - 1,

			x_type.shape[0] == t_type.shape[0],
			x_type.shape[2:] == t_type.shape[1:],
		)

	def _check_input_values(self, x, t):
		if not (((0 <= t) &
				 (t < x.shape[1])) |
				(t == self.ignore_label)).all():
			msg = ('Each label `t` need to satisfy '
				   '`0 <= t < x.shape[1] or t == %d`' % self.ignore_label)
			raise ValueError(msg)

	def forward_cpu(self, inputs):
		x, t = inputs
		if chainer.is_debug():
			self._check_input_values(x, t)

		log_y = log_softmax._log_softmax(x)
		if self.cache_score:
			self.y = np.exp(log_y)
		if self.class_weight is not None:
			shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
			log_y *= _broadcast_to(self.class_weight.reshape(shape), x.shape)
		log_yd = np.rollaxis(log_y, 1)
		log_yd = log_yd.reshape(len(log_yd), -1)
		log_p = log_yd[np.maximum(t.ravel(), 0), np.arange(t.size)]

		log_p *= (t.ravel() != self.ignore_label)
		if self.reduce == 'mean':
			# deal with the case where the SoftmaxCrossEntropy is
			# unpickled from the old version
			if self.normalize:
				count = (t != self.ignore_label).sum()
			else:
				count = len(x)
			self._coeff = 1.0 / max(count, 1)

			y = log_p.sum(keepdims=True) * (-self._coeff)
			return y.reshape(()),
		else:
			return -log_p.reshape(t.shape),

	def forward_gpu(self, inputs):
		cupy = cuda.cupy
		x, t = inputs
		if chainer.is_debug():
			self._check_input_values(x, t)

		log_y = log_softmax._log_softmax(x)
		if self.cache_score:
			self.y = cupy.exp(log_y)
		if self.class_weight is not None:
			shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
			log_y *= cupy.broadcast_to(
				self.class_weight.reshape(shape), x.shape)
		if self.normalize:
			coeff = cupy.maximum(1, (t != self.ignore_label).sum())
		else:
			coeff = max(1, len(t))
		self._coeff = cupy.divide(1.0, coeff, dtype=x.dtype)

		log_y = cupy.rollaxis(log_y, 1, log_y.ndim)
		if self.reduce == 'mean':
			ret = cuda.reduce(
				'S t, raw T log_y, int32 n_channel, raw T coeff, '
				'S ignore_label',
				'T out',
				't == ignore_label ? T(0) : log_y[_j * n_channel + t]',
				'a + b', 'out = a * -coeff[0]', '0', 'crossent_fwd'
			)(t, log_y.reduced_view(), log_y.shape[-1],
			  self._coeff, self.ignore_label)
		else:
			ret = cuda.elementwise(
				'S t, raw T log_y, int32 n_channel, T ignore', 'T out',
				'''
				if (t == ignore) {
				  out = 0;
				} else {
				  out = -log_y[i * n_channel + t];
				}
				''',
				'softmax_crossent_no_reduce_fwd'
			)(t, log_y.reduced_view(), log_y.shape[-1], self.ignore_label)
			ret = ret.reshape(t.shape)
		return ret,

	def backward_cpu(self, inputs, grad_outputs):
		x, t = inputs
		gloss = grad_outputs[0]
		if hasattr(self, 'y'):
			y = self.y.copy()
		else:
			y = log_softmax._log_softmax(x)
			np.exp(y, out=y)
		if y.ndim == 2:
			gx = y
			gx[np.arange(len(t)), np.maximum(t, 0)] -= 1
			if self.class_weight is not None:
				shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
				c = _broadcast_to(self.class_weight.reshape(shape), x.shape)
				c = c[np.arange(len(t)), np.maximum(t, 0)]
				gx *= _broadcast_to(np.expand_dims(c, 1), gx.shape)
			gx *= (t != self.ignore_label).reshape((len(t), 1))
		else:
			n_unit = t.size // len(t)
			gx = y.reshape(y.shape[0], y.shape[1], -1)
			fst_index = np.arange(t.size) // n_unit
			trd_index = np.arange(t.size) % n_unit
			gx[fst_index, np.maximum(t.ravel(), 0), trd_index] -= 1
			if self.class_weight is not None:
				shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
				c = _broadcast_to(self.class_weight.reshape(shape), x.shape)
				c = c.reshape(gx.shape)
				c = c[fst_index, np.maximum(t.ravel(), 0), trd_index]
				c = c.reshape(y.shape[0], 1, -1)
				gx *= _broadcast_to(c, gx.shape)
			gx *= (t != self.ignore_label).reshape((len(t), 1, -1))
			gx = gx.reshape(y.shape)
		if self.reduce == 'mean':
			gx *= gloss * self._coeff
		else:
			gx *= gloss[:, None]
		return gx, None

	def backward_gpu(self, inputs, grad_outputs):
		cupy = cuda.cupy
		x, t = inputs
		if hasattr(self, 'y'):
			y = self.y
		else:
			y = log_softmax._log_softmax(x)
			cupy.exp(y, out=y)
		gloss = grad_outputs[0]
		n_unit = t.size // len(t)
		if self.reduce == 'mean':
			coeff = gloss * self._coeff
		else:
			coeff = gloss[:, None, ...]

		if self.class_weight is None:
			gx = cuda.elementwise(
				'T y, S t, T coeff, S n_channel, S n_unit, S ignore_label',
				'T gx',
				'''
					const int c = (i / n_unit % n_channel);
					gx = t == ignore_label ? 0 : coeff * (y - (c == t));
				''',
				'softmax_crossent_bwd')(
					y, cupy.expand_dims(t, 1), coeff, x.shape[1],
					n_unit, self.ignore_label)
		else:
			gx = cuda.elementwise(
				'T y, raw T w, S t, T coeff, S n_channel, S n_unit, '
				'S ignore_label',
				'T gx',
				'''
					const int c = (i / n_unit % n_channel);
					gx = t == ignore_label ? 0 : coeff * (y - (c == t)) * w[t];
				''',
				'softmax_crossent_weight_bwd')(
					y, self.class_weight, cupy.expand_dims(t, 1), coeff,
					x.shape[1], n_unit, self.ignore_label)

		return gx, None


def softmax_cross_entropy(x, t, use_cudnn=True, normalize=True, cache_score=True, class_weight=None, ignore_label=-1, reduce='mean'):
	return SoftmaxCrossEntropy(use_cudnn, normalize, cache_score, class_weight, ignore_label, reduce)(x, t)

def compute_accuracy_batch(model, batch):
	source, target = make_source_target_pair(batch)
	if model.xp is cuda.cupy:
		source = cuda.to_gpu(source)
		target = cuda.to_gpu(target)
	model.reset_state()
	Y = model(source, test=True)
	return float(F.accuracy(Y, target, ignore_label=ID_PAD).data)

def compute_accuracy(model, buckets, batchsize=100):
	result = []
	for bucket_index, dataset in enumerate(buckets):
		acc = []
		# split into minibatch
		if len(dataset) > batchsize:
			num_sections = len(dataset) // batchsize - 1
			if len(dataset) % batchsize > 0:
				num_sections += 1
			indices = [(i + 1) * batchsize for i in xrange(num_sections)]
			sections = np.split(dataset, indices, axis=0)
		else:
			sections = [dataset]
		# compute accuracy
		for batch_index, batch in enumerate(sections):
			sys.stdout.write("\rcomputing accuracy ... bucket {}/{} (batch {}/{})".format(bucket_index + 1, len(buckets), batch_index + 1, len(sections)))
			sys.stdout.flush()
			acc.append(compute_accuracy_batch(model, batch))

		result.append(reduce(lambda x, y: x + y, acc) / len(acc))
		sys.stdout.write("\r" + stdout.CLEAR)
		sys.stdout.flush()

	return result

def compute_random_accuracy(model, buckets, batchsize=100):
	acc = []
	for dataset in buckets:
		batch = sample_batch_from_bucket(dataset, batchsize)
		acc.append(compute_accuracy_batch(model, batch))
	return acc

def compute_perplexity_batch(model, batch):
	sum_log_likelihood = 0
	source, target = make_source_target_pair(batch)
	xp = model.xp
	if xp is cuda.cupy:
		source = cuda.to_gpu(source)
		target = cuda.to_gpu(target)
	model.reset_state()
	Y = model(source, test=True)
	neglogp = softmax_cross_entropy(Y, target, ignore_label=ID_PAD)
	return  math.exp(float(neglogp.data))

def compute_perplexity(model, buckets, batchsize=100):
	result = []
	for bucket_index, dataset in enumerate(buckets):
		ppl = []
		# split into minibatch
		if len(dataset) > batchsize:
			num_sections = len(dataset) // batchsize - 1
			if len(dataset) % batchsize > 0:
				num_sections += 1
			indices = [(i + 1) * batchsize for i in xrange(num_sections)]
			sections = np.split(dataset, indices, axis=0)
		else:
			sections = [dataset]
		# compute accuracy
		for batch_index, batch in enumerate(sections):
			sys.stdout.write("\rcomputing perplexity ... bucket {}/{} (batch {}/{})".format(bucket_index + 1, len(buckets), batch_index + 1, len(sections)))
			sys.stdout.flush()
			ppl.append(compute_perplexity_batch(model, batch))

		result.append(reduce(lambda x, y: x + y, ppl) / len(ppl))
		sys.stdout.write("\r" + stdout.CLEAR)
		sys.stdout.flush()
	return result

def compute_random_perplexity(model, buckets, batchsize=100):
	ppl = []
	for dataset in buckets:
		batch = sample_batch_from_bucket(dataset, batchsize)
		ppl.append(compute_perplexity_batch(model, batch))
	return ppl

def main(args):
	# load textfile
	dataset_train, dataset_dev, dataset_test, vocab, vocab_inv = read_data(args.train_filename, args.dev_filename, args.test_filename)
	vocab_size = len(vocab)
	print_bold("data	#	hash")
	print("train	{}	{}".format(len(dataset_train), hash(str(dataset_train))))
	if len(dataset_dev) > 0:
		print("dev	{}	{}".format(len(dataset_dev), hash(str(dataset_dev))))
	if len(dataset_test) > 0:
		print("test	{}	{}".format(len(dataset_test), hash(str(dataset_test))))
	print("vocab	{}".format(vocab_size))

	# split into buckets
	buckets_train = None
	if len(dataset_train) > 0:
		print_bold("buckets	#data	(train)")
		buckets_train = make_buckets(dataset_train)
		if args.buckets_slice is not None:
			buckets_train = buckets_train[:args.buckets_slice + 1]
		for size, data in zip(bucket_sizes, buckets_train):
			print("{}	{}".format(size, len(data)))

	buckets_dev = None
	if len(dataset_dev) > 0:
		print_bold("buckets	#data	(dev)")
		buckets_dev = make_buckets(dataset_dev)
		if args.buckets_slice is not None:
			buckets_dev = buckets_dev[:args.buckets_slice + 1]
		for size, data in zip(bucket_sizes, buckets_dev):
			print("{}	{}".format(size, len(data)))

	buckets_test = None
	if len(dataset_dev) > 0:
		print_bold("buckets	#data	(test)")
		buckets_test = make_buckets(dataset_test)
		if args.buckets_slice is not None:
			buckets_test = buckets_test[:args.buckets_slice + 1]
		for size, data in zip(bucket_sizes, buckets_test):
			print("{}	{}".format(size, len(data)))

	# init
	model = load_model(args.model_dir)
	assert model is not None
	if args.gpu_device >= 0:
		chainer.cuda.get_device(args.gpu_device).use()
		model.to_gpu()

	# show log
	def mean(l):
		return sum(l) / len(l)

	sys.stdout.write("\r" + stdout.CLEAR)
	sys.stdout.flush()

	with chainer.using_config("train", False):
		if buckets_train is not None:
			print_bold("ppl (train)")
			ppl_train = compute_perplexity(model, buckets_train, args.batchsize)
			print(mean(ppl_train), ppl_train)

		if buckets_dev is not None:
			print_bold("ppl (dev)")
			ppl_dev = compute_perplexity(model, buckets_dev, args.batchsize)
			print(mean(ppl_dev), ppl_dev)

		if buckets_test is not None:
			print_bold("ppl (test)")
			ppl_test = compute_perplexity(model, buckets_test, args.batchsize)
			print(mean(ppl_test), ppl_dev)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--batchsize", "-b", type=int, default=96)
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--train-split", type=float, default=0.9)
	parser.add_argument("--dev-split", type=float, default=0.05)
	parser.add_argument("--buckets-slice", type=int, default=None)
	parser.add_argument("--train-filename", "-train", default=None)
	parser.add_argument("--dev-filename", "-dev", default=None)
	parser.add_argument("--test-filename", "-test", default=None)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	args = parser.parse_args()
	main(args)