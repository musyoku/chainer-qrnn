# coding: utf-8
import argparse, sys
import numpy as np
import chainer.functions as F
import chainer
from chainer import cuda, functions
from chainer.utils import type_check
from chainer.functions.activation import log_softmax
from model import load_model, load_vocab
from common import ID_UNK, ID_PAD, ID_GO, ID_EOS, bucket_sizes, stdout, print_bold
from dataset import read_data, make_buckets, sample_batch_from_bucket
from translate import translate_beam_search, translate_greedy

def _broadcast_to(array, shape):
	if hasattr(numpy, "broadcast_to"):
		return numpy.broadcast_to(array, shape)
	dummy = numpy.empty(shape, array.dtype)
	return numpy.broadcast_arrays(array, dummy)[0]


class SoftmaxCrossEntropy(chainer.function.Function):

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

# https://github.com/zszyellow/WER-in-python
def compute_error_rate_target_prediction(r, h):
	# build the matrix
	d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape((len(r) + 1, len(h) + 1))
	for i in xrange(len(r) + 1):
		for j in xrange(len(h) + 1):
			if i == 0: d[0][j] = j
			elif j == 0: d[i][0] = i
	for i in xrange(1, len(r) + 1):
		for j in xrange(1, len(h) + 1):
			if r[i-1] == h[j-1]:
				d[i][j] = d[i-1][j-1]
			else:
				substitute = d[i-1][j-1] + 1
				insert = d[i][j-1] + 1
				delete = d[i-1][j] + 1
				d[i][j] = min(substitute, insert, delete)
	return float(d[len(r)][len(h)]) / len(r)

def compute_error_rate_source_batch(model, source_batch, target_batch, target_vocab_size):
	xp = model.xp
	sum_wer = 0
	batchsize = source_batch.shape[0]
	x = translate_greedy(model, source_batch, target_batch.shape[1] * 2, target_vocab_size)

	for n in xrange(batchsize):
		target_tokens = []
		for token in target_batch[n]:
			token = int(token)	# to cpu
			if token == ID_PAD:
				break
			if token == ID_EOS:
				break
			if token == ID_GO:
				continue
			target_tokens.append(token)
		assert len(target_tokens) > 0

		predict_tokens = []
		for token in x[n]:
			token = int(token)	# to cpu
			if token == ID_EOS:
				break
			if token == ID_PAD:
				break
			if token == ID_GO:
				continue
			predict_tokens.append(token)
		assert len(predict_tokens) > 0

		wer = compute_error_rate_target_prediction(target_tokens, predict_tokens)
		sum_wer += wer

	return sum_wer / batchsize

def compute_error_rate_source_sequence(model, source, target, target_vocab_size, beam_width=8, normalization_alpha=0):
	xp = model.xp
	x = translate_beam_search(model, source, target.size * 2, target_vocab_size, beam_width, normalization_alpha)

	target_tokens = []
	for token in target:
		token = int(token)	# to cpu
		if token == ID_PAD:
			break
		if token == ID_EOS:
			break
		if token == ID_GO:
			continue
		target_tokens.append(token)
		assert len(target_tokens) > 0

	predict_tokens = []
	for token in x:
		token = int(token)	# to cpu
		if token == ID_EOS:
			break
		if token == ID_PAD:
			break
		if token == ID_GO:
			continue
		predict_tokens.append(token)
		assert len(predict_tokens) > 0

	return compute_error_rate_target_prediction(target_tokens, predict_tokens)

def compute_error_rate_buckets(model, source_buckets, target_buckets, target_vocab_size, beam_width=8, normalization_alpha=0):
	result = []
	for bucket_index, (source_bucket, target_bucket) in enumerate(zip(source_buckets, target_buckets)):
		sum_wer = 0

		if beam_width == 1:	# greedy
			batchsize = 24

			if len(source_bucket) > batchsize:
				num_sections = len(source_bucket) // batchsize - 1
				if len(source_bucket) % batchsize > 0:
					num_sections += 1
				indices = [(i + 1) * batchsize for i in xrange(num_sections)]
				source_sections = np.split(source_bucket, indices, axis=0)
				target_sections = np.split(target_bucket, indices, axis=0)
			else:
				source_sections = [source_bucket]
				target_sections = [target_bucket]

			for batch_index, (source_batch, target_batch) in enumerate(zip(source_sections, target_sections)):
				sys.stdout.write("\rcomputing WER ... bucket {}/{} (batch {}/{})".format(bucket_index + 1, len(source_buckets), batch_index + 1, len(source_sections)))
				sys.stdout.flush()
				mean_wer = compute_error_rate_source_batch(model, source_batch, target_batch, target_vocab_size)
				sum_wer += mean_wer
			result.append(sum_wer / len(source_sections) * 100)

		else:	# beam search
			for index in xrange(len(source_bucket)):
				sys.stdout.write("\rcomputing WER ... bucket {}/{} (sequence {}/{})".format(bucket_index + 1, len(source_buckets), index + 1, len(source_bucket)))
				sys.stdout.flush()
				source = source_bucket[index]
				target = target_bucket[index]
				wer = compute_error_rate_source_sequence(model, source, target, target_vocab_size, beam_width, normalization_alpha)
				sum_wer += wer
			result.append(sum_wer / len(source_bucket) * 100)
		
		sys.stdout.write("\r" + stdout.CLEAR)
		sys.stdout.flush()

	return result

def compute_random_error_rate_buckets(model, source_buckets, target_buckets, target_vocab_size, sample_size=100, beam_width=8, normalization_alpha=0):
	xp = model.xp
	result = []
	for bucket_index, (source_bucket, target_bucket) in enumerate(zip(source_buckets, target_buckets)):
		source_batch, target_batch = sample_batch_from_bucket(source_bucket, target_bucket, sample_size)
		
		if beam_width == 1:	# greedy
			mean_wer = compute_error_rate_source_batch(model, source_batch, target_batch, target_vocab_size)

		else:	# beam search
			sum_wer = 0
			for index in xrange(sample_size):
				sys.stdout.write("\rcomputing WER ... bucket {}/{} (sequence {}/{})".format(bucket_index + 1, len(source_buckets), index + 1, sample_size))
				sys.stdout.flush()
				source = source_batch[index]
				target = target_batch[index]
				wer = compute_error_rate_source_sequence(model, source, target, target_vocab_size, beam_width, normalization_alpha)
				sum_wer += wer
			mean_wer = sum_wer / len(source_batch)

			sys.stdout.write("\r" + stdout.CLEAR)
			sys.stdout.flush()

		result.append(mean_wer * 100)

	return result

def main(args):
	vocab, vocab_inv = load_vocab(args.model_dir)
	vocab_source, vocab_target = vocab
	vocab_inv_source, vocab_inv_target = vocab_inv

	source_dataset, target_dataset = read_data(vocab_source, vocab_target, args.source_train, args.target_train, args.source_dev, args.target_dev, args.source_test, args.target_test, reverse_source=True)

	source_dataset_train, source_dataset_dev, source_dataset_test = source_dataset
	target_dataset_train, target_dataset_dev, target_dataset_test = target_dataset
	print_bold("data	#")
	if len(source_dataset_train) > 0:
		print("train	{}".format(len(source_dataset_train)))
	if len(source_dataset_dev) > 0:
		print("dev	{}".format(len(source_dataset_dev)))
	if len(source_dataset_test) > 0:
		print("test	{}".format(len(source_dataset_test)))


	print("vocab	{}	(source)".format(len(vocab_source)))
	print("vocab	{}	(target)".format(len(vocab_target)))

	# split into buckets
	source_buckets_train = None
	if len(source_dataset_train) > 0:
		print_bold("buckets 	#data	(train)")
		source_buckets_train, target_buckets_train = make_buckets(source_dataset_train, target_dataset_train)
		if args.buckets_slice is not None:
			source_buckets_train = source_buckets_train[:args.buckets_slice + 1]
			target_buckets_train = target_buckets_train[:args.buckets_slice + 1]
		for size, data in zip(bucket_sizes, source_buckets_train):
			print("{} 	{}".format(size, len(data)))

	source_buckets_dev = None
	if len(source_dataset_dev) > 0:
		print_bold("buckets 	#data	(dev)")
		source_buckets_dev, target_buckets_dev = make_buckets(source_dataset_dev, target_dataset_dev)
		if args.buckets_slice is not None:
			source_buckets_dev = source_buckets_dev[:args.buckets_slice + 1]
			target_buckets_dev = target_buckets_dev[:args.buckets_slice + 1]
		for size, data in zip(bucket_sizes, source_buckets_dev):
			print("{} 	{}".format(size, len(data)))

	source_buckets_test = None
	if len(source_dataset_test) > 0:
		print_bold("buckets		#data	(test)")
		source_buckets_test, target_buckets_test = make_buckets(source_dataset_test, target_dataset_test)
		if args.buckets_slice is not None:
			source_buckets_test = source_buckets_test[:args.buckets_slice + 1]
			target_buckets_test = target_buckets_test[:args.buckets_slice + 1]
		for size, data in zip(bucket_sizes, source_buckets_test):
			print("{} 	{}".format(size, len(data)))


	model = load_model(args.model_dir)
	assert model is not None
	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		model.to_gpu()

	with chainer.using_config("train", False):
		if source_buckets_train is not None:
			print_bold("WER (train)")
			wer_train = compute_error_rate_buckets(model, source_buckets_train, target_buckets_train, len(vocab_target), args.beam_width, args.alpha)
			print(wer_train)

		if source_buckets_dev is not None:
			print_bold("WER (dev)")
			wer_dev = compute_error_rate_buckets(model, source_buckets_dev, target_buckets_dev, len(vocab_target), args.beam_width, args.alpha)
			print(wer_dev)

		if source_buckets_test is not None:
			print_bold("WER (test)")
			wer_test = compute_error_rate_buckets(model, source_buckets_test, target_buckets_test, len(vocab_target), args.beam_width, args.alpha)
			print(wer_test)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--source-train", type=str, default=None)
	parser.add_argument("--source-dev", type=str, default=None)
	parser.add_argument("--source-test", type=str, default=None)
	parser.add_argument("--target-train", type=str, default=None)
	parser.add_argument("--target-dev", type=str, default=None)
	parser.add_argument("--target-test", type=str, default=None)
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--buckets-slice", type=int, default=None)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	parser.add_argument("--beam-width", "-beam", type=int, default=8)
	parser.add_argument("--alpha", "-alpha", type=float, default=0)
	args = parser.parse_args()
	main(args)