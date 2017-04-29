# coding: utf-8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import argparse, sys, os, codecs, random
import numpy as np
import chainer
import chainer.functions as F
from chainer import training, Variable, Chain
from chainer.training import extensions
sys.path.append(os.path.split(os.getcwd())[0])
import qrnn as L

_bucket_sizes = [10, 20, 40, 60, 100, 120]

def read_data(filepath, train_split_ratio=0.9, validation_split_ratio=0.05, seed=0):
	assert(train_split_ratio + validation_split_ratio <= 1)
	id_pad = 0
	id_bos = 1
	id_eos = 2
	vocab = {
		"<bos>": id_bos,
		"<eos>": id_eos,
		"<pad>": id_pad,
	}
	dataset = []
	with codecs.open(filepath, "r", "utf-8") as f:
		for sentence in f:
			sentence = sentence.strip()
			if len(sentence) == 0:
				continue
			word_ids = [id_bos]
			words = sentence.split(" ")
			for word in words:
				if word not in vocab:
					vocab[word] = len(vocab)
				word_id = vocab[word]
				word_ids.append(word_id)
			word_ids.append(id_eos)
			dataset.append(word_ids)

	random.seed(seed)
	random.shuffle(dataset)

	# [train][validation] | [test]
	train_split = int(len(dataset) * (train_split_ratio + validation_split_ratio))
	train_validation_dataset = dataset[:train_split]
	test_dataset = dataset[train_split:]

	# [train] | [validation]
	validation_split = int(len(train_validation_dataset) * validation_split_ratio)
	validation_dataset = train_validation_dataset[:validation_split]
	train_dataset = train_validation_dataset[validation_split:]

	return train_dataset, validation_dataset, test_dataset, vocab

# input:
# [0, a, b, c, 1]
# [0, d, e, 1]
# output:
# [[0, a, b, c,  1]
#  [0, d, e, 1, -1]]
def make_batch_buckets(dataset):
	max_length = 0
	for word_ids in dataset:
		if len(word_ids) > max_length:
			max_length = len(word_ids)
	_bucket_sizes.append(max_length)
	buckets_list = [[] for _ in xrange(len(_bucket_sizes))]
	for word_ids in dataset:
		length = len(word_ids)
		bucket_index = 0
		for size in _bucket_sizes:
			if length <= size:
				if size - length > 0:
					for _ in xrange(size - length):
						word_ids.append(0)
				break
			bucket_index += 1
		buckets_list[bucket_index].append(word_ids)
	buckets = []
	for bucket in buckets_list:
		if len(bucket) == 0:
			continue
		buckets.append(np.asarray(bucket).astype(np.int32))
	return buckets

def sample_batch_from_bucket(bucket, num_samples):
	num_samples = num_samples if len(bucket) >= num_samples else len(bucket)
	indices = np.random.choice(np.arange(len(bucket), dtype=np.int32), size=num_samples, replace=False)
	return bucket[indices]

def make_source_target_pair(batch):
	source = batch[:, :-1]
	target = batch[:, 1:]
	target = np.reshape(target, (-1,))
	return Variable(source), Variable(target)

class QRNN(Chain):
	def __init__(self, num_vocab, ndim_embedding):
		super(QRNN, self).__init__(
			embed=L.EmbedID(num_vocab, ndim_embedding, ignore_label=0),
			l1=L.QRNN(ndim_embedding, ndim_embedding, kernel_size=4, pooling="fo", zoneout=True),
			l2=L.QRNN(ndim_embedding, ndim_embedding, kernel_size=4, pooling="fo", zoneout=True),
			l3=L.Linear(ndim_embedding, num_vocab),
		)
		for param in self.params():
			param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

		self.ndim_embedding = ndim_embedding

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()

	# we use "dense convolution"
	# https://arxiv.org/abs/1608.06993
	def __call__(self, X):
		H0 = self.embed(X)
		H0 = F.swapaxes(H0, 1, 2)
		self.l1(H0)
		H1 = self.l1.get_all_hidden_states() + H0
		self.l2(H1)
		H2 = self.l2.get_all_hidden_states() + H1
		H2 = F.reshape(F.swapaxes(H2, 1, 2), (-1, self.ndim_embedding))
		Y = self.l3(H2)
		return Y

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=50, help="Number of examples in each mini-batch")
	parser.add_argument("--bproplen", "-l", type=int, default=35, help="Number of words in each mini-batch (= length of truncated BPTT)")
	parser.add_argument("--epoch", "-e", type=int, default=39, help="Number of sweeps over the dataset to train")
	parser.add_argument("--gpu_device", "-g", type=int, default=0, help="GPU ID (negative value indicates CPU)")
	parser.add_argument("--gradclip", "-c", type=float, default=5, help="Gradient norm threshold to clip")
	parser.add_argument("--out", "-o", default="result", help="Directory to output the result")
	parser.add_argument("--resume", "-r", default="", help="Resume the training from snapshot")
	parser.add_argument("--test", action="store_true", help="Use tiny datasets for quick tests")
	parser.set_defaults(test=False)
	parser.add_argument("--unit", "-u", type=int, default=650, help="Number of LSTM units in each layer")
	parser.add_argument("--model", "-m", default="model.npz", help="Model file name to serialize")
	parser.add_argument("--input-filename", "-i", default=None, help="Model file name to serialize")
	args = parser.parse_args()

	ndim_embedding = 200

	# load textfile
	train_dataset, validation_dataset, test_dataset, vocab = read_data(args.input_filename)
	vocab_size = len(vocab)
	print("#train =", len(train_dataset))
	print("#validation =", len(validation_dataset))
	print("#test =", len(test_dataset))
	print("#vocab =", vocab_size)

	# split into buckets
	train_buckets = make_batch_buckets(train_dataset)
	for size, data in zip(_bucket_sizes, train_buckets):
		print("{}	{}".format(size, len(data)))

	# init
	model = QRNN(vocab_size, ndim_embedding)
	if args.gpu_device >= 0:
		chainer.cuda.get_device(args.gpu_device).use()
		model.to_gpu()

	# setup an optimizer
	optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.9)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

	# training
	num_iteration = len(train_dataset) // args.batchsize
	for epoch in xrange(1, args.epoch + 1):
		for itr in xrange(num_iteration):
			for dataset in train_buckets:
				batch = sample_batch_from_bucket(dataset, args.batchsize)
				source, target = make_source_target_pair(batch)
				if args.gpu_device >= 0:
					source.to_gpu()
					target.to_gpu()
				Y = model(source)
				loss = F.softmax_cross_entropy(Y, target)
				optimizer.update(lossfun=lambda: loss)
			print itr

if __name__ == "__main__":
	main()