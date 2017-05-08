# coding: utf-8
import argparse, sys
import numpy as np
import chainer.functions as F
import chainer
from chainer import cuda, functions
from chainer.utils import type_check
from chainer.functions.activation import log_softmax
from model import Seq2SeqModel, AttentiveSeq2SeqModel, load_model
from common import ID_UNK, ID_PAD, ID_GO, ID_EOS, bucket_sizes, stdout, print_bold
from dataset import read_data, make_buckets, make_source_target_pair, sample_batch_from_bucket

class SoftmaxCrossEntropy(functions.loss.softmax_cross_entropy.SoftmaxCrossEntropy):

	def forward_gpu(self, inputs):
		cupy = cuda.cupy
		x, t = inputs
		if chainer.is_debug():
			self._check_input_values(x, t)

		log_y = log_softmax._log_softmax(x, self.use_cudnn)
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
		ret = cuda.reduce(
			'S t, raw T log_y, int32 n_channel, raw T coeff, S ignore_label', 'T out',
			't == ignore_label ? T(0) : log_y[_j * n_channel + t]',
			'a + b', 'out = a * -coeff[0]', '0', 'crossent_fwd'
		)(t, log_y.reduced_view(), log_y.shape[-1], self._coeff, self.ignore_label)
		return ret,

	def backward_gpu(self, inputs, grad_outputs):
		cupy = cuda.cupy
		x, t = inputs
		if hasattr(self, 'y'):
			y = self.y
		else:
			y = log_softmax._log_softmax(x, self.use_cudnn)
			cupy.exp(y, out=y)
		gloss = grad_outputs[0]
		n_unit = t.size // len(t)
		coeff = gloss * self._coeff
		if self.class_weight is None:
			gx = cuda.elementwise(
				'T y, S t, raw T coeff, S n_channel, S n_unit, S ignore_label',
				'T gx',
				'''
					const int c = (i / n_unit % n_channel);
					gx = (t == ignore_label) ? 0 : (coeff[0] * (y - (c == t)));
				''',
				'softmax_crossent_bwd')(
					y, cupy.expand_dims(t, 1), coeff, x.shape[1], n_unit, self.ignore_label)
		else:
			gx = cuda.elementwise(
				'T y, raw T w, S t, raw T coeff, S n_channel, S n_unit, S ignore_label',
				'T gx',
				'''
					const int c = (i / n_unit % n_channel);
					gx = t == ignore_label ? 0 : coeff[0] * (y - (c == t)) * w[t];
				''',
				'softmax_crossent_bwd')(
					y, self.class_weight, cupy.expand_dims(t, 1), coeff,
					x.shape[1], n_unit, self.ignore_label)
		return gx, None


def softmax_cross_entropy(x, t, use_cudnn=True, normalize=True, cache_score=True, class_weight=None, ignore_label=-1):
	return SoftmaxCrossEntropy(use_cudnn, normalize, cache_score, class_weight, ignore_label)(x, t)

# https://github.com/zszyellow/WER-in-python
def compute_word_error_rate_of_sequence(r, h):
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

def _compute_batch_wer_mean(model, source_batch, target_batch, target_vocab_size, argmax=True):
	xp = model.xp
	num_calculation = 0
	sum_wer = 0
	skip_mask = source_batch != ID_PAD
	batchsize = source_batch.shape[0]
	target_seq_length = target_batch.shape[1]

	# to gpu
	if xp is cuda.cupy:
		source_batch = cuda.to_gpu(source_batch)
		target_batch = cuda.to_gpu(target_batch)
		skip_mask = cuda.to_gpu(skip_mask)

	word_ids = xp.arange(0, target_vocab_size, dtype=xp.int32)

	model.reset_state()
	token = ID_GO
	x = xp.asarray([[token]]).astype(xp.int32)
	x = xp.broadcast_to(x, (batchsize, 1))

	# get encoder's last hidden states
	if isinstance(model, AttentiveSeq2SeqModel):
		encoder_last_hidden_states, encoder_last_layer_outputs = model.encode(source_batch, skip_mask, test=True)
	else:
		encoder_last_hidden_states = model.encode(source_batch, skip_mask, test=True)

	while x.shape[1] < target_seq_length * 2:
		if isinstance(model, AttentiveSeq2SeqModel):
			u = model.decode_one_step(x, encoder_last_hidden_states, encoder_last_layer_outputs, skip_mask, test=True)
		else:
			u = model.decode_one_step(x, encoder_last_hidden_states, test=True)
		p = F.softmax(u)	# convert to probability

		# concatenate
		if xp is np:
			x = xp.append(x, xp.zeros((batchsize, 1), dtype=xp.int32), axis=1)
		else:
			x = xp.concatenate((x, xp.zeros((batchsize, 1), dtype=xp.int32)), axis=1)

		for n in xrange(batchsize):
			pn = p.data[n]

			# argmax or sampling
			if argmax:
				token = xp.argmax(pn)
			else:
				token = xp.random.choice(word_ids, size=1, p=pn)[0]

			x[n, -1] = token

	for n in xrange(batchsize):
		target_tokens = []
		for token in target_batch[n, :]:
			token = int(token)	# to cpu
			if token == ID_PAD:
				break
			if token == ID_EOS:
				break
			if token == ID_GO:
				continue
			target_tokens.append(token)

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

		wer = compute_word_error_rate_of_sequence(target_tokens, predict_tokens)
		sum_wer += wer
		num_calculation += 1

	return sum_wer / num_calculation

def compute_mean_wer(model, source_buckets, target_buckets, target_vocab_size, batchsize=100, argmax=True):
	result = []
	for bucket_index, (source_bucket, target_bucket) in enumerate(zip(source_buckets, target_buckets)):
		num_calculation = 0
		sum_wer = 0

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
			mean_wer = _compute_batch_wer_mean(model, source_batch, target_batch, target_vocab_size, argmax=argmax)
			sum_wer += mean_wer
			num_calculation += 1

		result.append(sum_wer / num_calculation * 100)
		
		sys.stdout.write("\r" + stdout.CLEAR)
		sys.stdout.flush()

	return result

def compute_random_mean_wer(model, source_buckets, target_buckets, target_vocab_size, sample_size=100, argmax=True):
	xp = model.xp
	result = []
	for source_bucket, target_bucket in zip(source_buckets, target_buckets):
		# sample minibatch
		source_batch, target_batch = sample_batch_from_bucket(source_bucket, target_bucket, sample_size)
		
		# compute WER
		mean_wer = _compute_batch_wer_mean(model, source_batch, target_batch, target_vocab_size, argmax=argmax)

		result.append(mean_wer * 100)

	return result

def main(args):
	# load textfile
	source_dataset, target_dataset, vocab, vocab_inv = read_data(args.source_filename, args.target_filename, train_split_ratio=args.train_split, dev_split_ratio=args.dev_split, seed=args.seed)

	source_dataset_train, source_dataset_dev, source_dataset_test = source_dataset
	target_dataset_train, target_dataset_dev, target_dataset_test = target_dataset
	print_bold("data	#")
	print("train	{}".format(len(source_dataset_train)))
	print("dev	{}".format(len(source_dataset_dev)))
	print("test	{}".format(len(source_dataset_test)))

	vocab_source, vocab_target = vocab
	vocab_inv_source, vocab_inv_target = vocab_inv
	print("vocab	{}	(source)".format(len(vocab_source)))
	print("vocab	{}	(target)".format(len(vocab_target)))

	# split into buckets
	source_buckets_train, target_buckets_train = make_buckets(source_dataset_train, target_dataset_train)
	if args.buckets_limit is not None:
		source_buckets_train = source_buckets_train[:args.buckets_limit+1]
		target_buckets_train = target_buckets_train[:args.buckets_limit+1]
	print_bold("buckets 	#data	(train)")
	for size, data in zip(bucket_sizes, source_buckets_train):
		print("{} 	{}".format(size, len(data)))
	print_bold("buckets 	#data	(dev)")

	source_buckets_dev, target_buckets_dev = make_buckets(source_dataset_dev, target_dataset_dev)
	if args.buckets_limit is not None:
		source_buckets_dev = source_buckets_dev[:args.buckets_limit+1]
		target_buckets_dev = target_buckets_dev[:args.buckets_limit+1]
	for size, data in zip(bucket_sizes, source_buckets_dev):
		print("{} 	{}".format(size, len(data)))
	print_bold("buckets		#data	(test)")

	source_buckets_test, target_buckets_test = make_buckets(source_dataset_test, target_dataset_test)
	if args.buckets_limit is not None:
		source_buckets_test = source_buckets_test[:args.buckets_limit+1]
		target_buckets_test = target_buckets_test[:args.buckets_limit+1]
	for size, data in zip(bucket_sizes, source_buckets_test):
		print("{} 	{}".format(size, len(data)))

	model = load_model(args.model_dir)
	assert model is not None
	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		model.to_gpu()

	print_bold("WER (train)")
	wer_train = compute_mean_wer(model, source_buckets_train, target_buckets_train, len(vocab_inv_target), batchsize=args.batchsize, argmax=True)
	print(wer_train)
	print_bold("WER (dev)")
	wer_dev = compute_mean_wer(model, source_buckets_dev, target_buckets_dev, len(vocab_inv_target), batchsize=args.batchsize, argmax=True)
	print(wer_dev)
	print_bold("WER (test)")
	wer_test = compute_mean_wer(model, source_buckets_test, target_buckets_test, len(vocab_inv_target), batchsize=args.batchsize, argmax=True)
	print(wer_test)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=50)
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--train-split", type=float, default=0.9)
	parser.add_argument("--dev-split", type=float, default=0.05)
	parser.add_argument("--source-filename", "-source", default=None)
	parser.add_argument("--target-filename", "-target", default=None)
	parser.add_argument("--buckets-limit", type=int, default=None)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	parser.add_argument("--seed", type=int, default=0)
	args = parser.parse_args()
	main(args)