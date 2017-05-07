# coding: utf-8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import argparse, sys, os, codecs, random, math
import numpy as np
import chainer
import chainer.functions as F
from chainer import training, Variable, optimizers, cuda
from chainer.training import extensions
sys.path.append(os.path.split(os.getcwd())[0])
from eve import Eve
from model import RNNModel, load_model, save_model, save_vocab
from common import ID_UNK, ID_PAD, ID_BOS, ID_EOS, bucket_sizes, stdout, print_bold
from dataset import read_data, make_buckets, sample_batch_from_bucket, make_source_target_pair
from error import compute_accuracy, compute_random_accuracy, compute_perplexity, compute_random_perplexity, softmax_cross_entropy

def main(args):
	# load textfile
	train_dataset, dev_dataset, test_dataset, vocab, vocab_inv = read_data(args.text_filename, train_split_ratio=args.train_split, dev_split_ratio=args.dev_split, seed=args.seed)
	save_vocab(args.model_dir, vocab, vocab_inv)
	vocab_size = len(vocab)
	print_bold("data	#")
	print("train	{}".format(len(train_dataset)))
	print("dev	{}".format(len(dev_dataset)))
	print("test	{}".format(len(test_dataset)))
	print("vocab	{}".format(vocab_size))

	# split into buckets
	train_buckets = make_buckets(train_dataset)

	print_bold("buckets	#data	(train)")
	if args.buckets_limit is not None:
		train_buckets = train_buckets[:args.buckets_limit+1]
	for size, data in zip(bucket_sizes, train_buckets):
		print("{}	{}".format(size, len(data)))

	print_bold("buckets	#data	(dev)")
	dev_buckets = make_buckets(dev_dataset)
	if args.buckets_limit is not None:
		dev_buckets = dev_buckets[:args.buckets_limit+1]
	for size, data in zip(bucket_sizes, dev_buckets):
		print("{}	{}".format(size, len(data)))

	print_bold("buckets	#data	(test)")
	test_buckets = make_buckets(test_dataset)
	for size, data in zip(bucket_sizes, test_buckets):
		print("{}	{}".format(size, len(data)))

	# to maintain equilibrium
	min_num_data = 0
	for data in train_buckets:
		if min_num_data == 0 or len(data) < min_num_data:
			min_num_data = len(data)
	repeats = []
	for data in train_buckets:
		repeats.append(len(data) // min_num_data + 1)

	num_updates_per_iteration = 0
	for repeat, data in zip(repeats, train_buckets):
		num_updates_per_iteration += repeat * args.batchsize
	num_iteration = len(train_dataset) // num_updates_per_iteration + 1

	# init
	model = load_model(args.model_dir)
	if model is None:
		model = RNNModel(vocab_size, args.ndim_embedding, args.num_layers, ndim_h=args.ndim_h, kernel_size=args.kernel_size, pooling=args.pooling, zoneout=args.zoneout, wstd=args.wstd, densely_connected=args.densely_connected, ignore_label=ID_PAD)
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
	optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

	# training
	for epoch in xrange(1, args.epoch + 1):
		print("Epoch", epoch)
		for itr in xrange(1, num_iteration + 1):
			sys.stdout.write("\r{} / {}".format(itr, num_iteration))
			sys.stdout.flush()

			for repeat, dataset in zip(repeats, train_buckets):
				for r in xrange(repeat):
					batch = sample_batch_from_bucket(dataset, args.batchsize)
					source, target = make_source_target_pair(batch)
					if model.xp is cuda.cupy:
						source = cuda.to_gpu(source)
						target = cuda.to_gpu(target)
					model.reset_state()
					Y = model(source)
					loss = softmax_cross_entropy(Y, target, ignore_label=ID_PAD)
					optimizer.update(lossfun=lambda: loss)


			if itr % args.interval == 0 or itr == num_iteration:
				save_model(args.model_dir, model)
				# show log
				def mean(l):
					return sum(l) / len(l)
				sys.stdout.write("\r" + stdout.CLEAR)
				sys.stdout.flush()
				print_bold("accuracy (sampled train)")
				acc_train = compute_random_accuracy(model, train_buckets, args.batchsize)
				print(mean(acc_train), acc_train)
				print_bold("accuracy (dev)")
				acc_dev = compute_accuracy(model, dev_buckets, args.batchsize)
				print(mean(acc_dev), acc_dev)
				print_bold("ppl (sampled train)")
				ppl_train = compute_random_perplexity(model, train_buckets, args.batchsize)
				print(mean(ppl_train), ppl_train)
				print_bold("ppl (dev)")
				ppl_dev = compute_perplexity(model, dev_buckets, args.batchsize)
				print(mean(ppl_dev), ppl_dev)

		sys.stdout.write("\r" + stdout.CLEAR)
		sys.stdout.flush()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=24)
	parser.add_argument("--epoch", "-e", type=int, default=1000)
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--grad-clip", "-gc", type=float, default=5) 
	parser.add_argument("--weight-decay", "-wd", type=float, default=5e-5) 
	parser.add_argument("--kernel-size", "-ksize", type=int, default=4)
	parser.add_argument("--ndim-h", "-nh", type=int, default=640)
	parser.add_argument("--ndim-embedding", "-ne", type=int, default=320)
	parser.add_argument("--num-layers", "-layers", type=int, default=2)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--train-split", type=float, default=0.9)
	parser.add_argument("--dev-split", type=float, default=0.05)
	parser.add_argument("--interval", type=int, default=100)
	parser.add_argument("--pooling", "-p", type=str, default="fo")
	parser.add_argument("--wstd", "-w", type=float, default=0.02)
	parser.add_argument("--buckets-limit", type=int, default=None)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	parser.add_argument("--text-filename", "-f", default=None)
	parser.add_argument("--densely-connected", "-dense", default=False, action="store_true")
	parser.add_argument("--zoneout", "-zoneout", default=False, action="store_true")
	parser.add_argument("--eve", default=False, action="store_true")
	args = parser.parse_args()
	main(args)