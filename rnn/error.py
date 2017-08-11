# coding: utf-8
from __future__ import division
from __future__ import print_function
import math, sys, argparse
import numpy as np
import chainer.functions as F
import chainer
from functools import reduce
from six.moves import xrange
from chainer import cuda, function
from chainer.utils import type_check
from chainer.functions.activation import log_softmax
from dataset import sample_batch_from_bucket, make_source_target_pair, read_data, make_buckets
from common import ID_PAD, ID_BOS, ID_EOS, stdout, print_bold, bucket_sizes
from model import load_model, load_vocab

def _broadcast_to(array, shape):
	if hasattr(numpy, "broadcast_to"):
		return numpy.broadcast_to(array, shape)
	dummy = numpy.empty(shape, array.dtype)
	return numpy.broadcast_arrays(array, dummy)[0]

def compute_accuracy_batch(model, batch):
	source, target = make_source_target_pair(batch)
	if model.xp is cuda.cupy:
		source = cuda.to_gpu(source)
		target = cuda.to_gpu(target)
	model.reset_state()
	Y = model(source)
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

		result.append(sum(acc) / len(acc))
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
	Y = model(source)
	neglogp = F.softmax_cross_entropy(Y, target, ignore_label=ID_PAD)
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

		result.append(sum(ppl) / len(ppl))
		
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
	vocab, vocab_inv = load_vocab(args.model_dir)
	dataset_train, dataset_dev, dataset_test, _, _ = read_data(args.train_filename, args.dev_filename, args.test_filename, vocab=vocab)
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
	if len(dataset_test) > 0:
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
	parser.add_argument("--buckets-slice", type=int, default=None)
	parser.add_argument("--train-filename", "-train", default=None)
	parser.add_argument("--dev-filename", "-dev", default=None)
	parser.add_argument("--test-filename", "-test", default=None)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	args = parser.parse_args()
	main(args)