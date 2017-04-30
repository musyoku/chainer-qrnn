# coding: utf-8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import argparse, sys, os, codecs, random, math
import numpy as np
import chainer
import chainer.functions as F
from chainer import training, Variable, serializers, optimizers, cuda
from chainer.training import extensions
sys.path.append(os.path.split(os.getcwd())[0])
from eve import Eve
from model import QRNN
from train import ID_BOS

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-device", "-g", type=int, default=0) 
parser.add_argument("--num-generate", "-n", type=int, default=10)
parser.add_argument("--model-filename", "-m", type=str, default=None)
args = parser.parse_args()

def main():
	# load textfile
	train_dataset, validation_dataset, test_dataset, vocab = read_data(args.text_filename)
	vocab_size = len(vocab)
	print_bold("data	#")
	print("train	{}".format(len(train_dataset)))
	print("dev	{}".format(len(validation_dataset)))
	print("test	{}".format(len(test_dataset)))
	print("vocab	{}".format(vocab_size))

	# split into buckets
	train_buckets = make_buckets(train_dataset)
	print_bold("buckets	#data	(train)")
	for size, data in zip(bucket_sizes, train_buckets):
		print("{}	{}".format(size, len(data)))
	print_bold("buckets	#data	(dev)")
	validation_buckets = make_buckets(validation_dataset)
	for size, data in zip(bucket_sizes, validation_buckets):
		print("{}	{}".format(size, len(data)))
	print_bold("buckets	#data	(test)")
	test_buckets = make_buckets(test_dataset)
	for size, data in zip(bucket_sizes, test_buckets):
		print("{}	{}".format(size, len(data)))

	# init
	model = QRNN(vocab_size, args.ndim_embedding, ndim_h=args.ndim_h, pooling=args.pooling, zoneout=args.zoneout, wstd=args.wstd)
	load_model(args.model_filename, model)
	if args.gpu_device >= 0:
		chainer.cuda.get_device(args.gpu_device).use()
		model.to_gpu()

	# setup an optimizer
	if args.eve:
		optimizer = Eve(alpha=0.001, beta1=0.9)
	else:
		optimizer = optimizers.Adam(alpha=0.0005, beta1=0.9)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))

	# training
	num_iteration = len(train_dataset) // args.batchsize
	for epoch in xrange(1, args.epoch + 1):
		print("Epoch", epoch)
		for itr in xrange(1, num_iteration + 1):
			for dataset in train_buckets:
				batch = sample_batch_from_bucket(dataset, args.batchsize)
				source, target = make_source_target_pair(batch)
				if args.gpu_device >= 0:
					source.to_gpu()
					target.to_gpu()
				Y = model(source)
				loss = F.softmax_cross_entropy(Y, target)
				optimizer.update(lossfun=lambda: loss)

			sys.stdout.write("\r{} / {}".format(itr, num_iteration))
			sys.stdout.flush()
			if itr % 500 == 0:
				print("\raccuracy: {} (train), {} (dev)".format(compute_minibatch_accuracy(model, train_buckets), compute_accuracy(model, validation_buckets)))
				print("\rppl: {} (train), {} (dev)".format(compute_minibatch_perplexity(model, train_buckets), compute_perplexity(model, validation_buckets)))
				save_model(args.model_filename, model)

if __name__ == "__main__":
	main()