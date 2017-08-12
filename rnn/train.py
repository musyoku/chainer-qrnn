# coding: utf-8
from __future__ import division
from __future__ import print_function
import argparse, sys, os, codecs, random, math, time
import numpy as np
import chainer
import chainer.functions as F
from chainer import Variable, optimizers, cuda
from model import RNNModel, load_model, save_model, save_vocab
from common import ID_PAD, ID_BOS, ID_EOS, bucket_sizes, printb, printr
from dataset import read_data, make_buckets, sample_batch_from_bucket, make_source_target_pair
from error import compute_accuracy, compute_random_accuracy, compute_perplexity, compute_random_perplexity
from optim import get_current_learning_rate, get_optimizer, decay_learning_rate

def dump_dataset(dataset_train, dataset_dev, train_buckets, dev_buckets, vocab_size):
	printb("data	#	hash")
	print("train	{}	{}".format(len(dataset_train), hash(str(dataset_train))))
	if len(dataset_dev) > 0:
		print("dev	{}	{}".format(len(dataset_dev), hash(str(dataset_dev))))
	print("vocab	{}".format(vocab_size))

	printb("buckets	#data	(train)")
	for size, data in zip(bucket_sizes, train_buckets):
		print("{}	{}".format(size, len(data)))

	if len(dev_buckets) > 0:
		printb("buckets	#data	(dev)")
		for size, data in zip(bucket_sizes, dev_buckets):
			print("{}	{}".format(size, len(data)))

def main():
	# load textfile
	dataset_train, dataset_dev, _, vocab, vocab_inv = read_data(args.train_filename, args.dev_filename)
	vocab_size = len(vocab)

	save_vocab(args.model_dir, vocab, vocab_inv)

	# split into buckets
	train_buckets = make_buckets(dataset_train)

	if args.buckets_slice is not None:
		train_buckets = train_buckets[:args.buckets_slice + 1]

	dev_buckets = None
	if len(dataset_dev) > 0:
		dev_buckets = make_buckets(dataset_dev)
		if args.buckets_slice is not None:
			dev_buckets = dev_buckets[:args.buckets_slice + 1]

	# print
	dump_dataset(dataset_train, dataset_dev, train_buckets, dev_buckets, vocab_size)

	# to maintain equilibrium
	required_interations = []
	for data in train_buckets:
		itr = math.ceil(len(data) / args.batchsize)
		required_interations.append(itr)
	total_iterations = sum(required_interations)
	buckets_distribution = np.asarray(required_interations, dtype=float) / total_iterations

	# init
	model = load_model(args.model_dir)
	if model is None:
		model = RNNModel(vocab_size, args.ndim_embedding, args.num_layers, ndim_h=args.ndim_h, kernel_size=args.kernel_size, pooling=args.pooling, zoneout=args.zoneout, dropout=args.dropout, weightnorm=args.weightnorm, wgain=args.wgain, densely_connected=args.densely_connected, ignore_label=ID_PAD)

	if args.gpu_device >= 0:
		chainer.cuda.get_device(args.gpu_device).use()
		model.to_gpu()

	# setup an optimizer
	optimizer = get_optimizer(args.optimizer, args.learning_rate, args.momentum)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))
	optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
	final_learning_rate = 1e-4
	total_time = 0

	def mean(l):
		return sum(l) / len(l)

	# training
	for epoch in range(1, args.epoch + 1):
		print("Epoch", epoch)
		start_time = time.time()

		with chainer.using_config("train", True):
			for itr in range(total_iterations):
				bucket_idx = int(np.random.choice(np.arange(len(train_buckets)), size=1, p=buckets_distribution))
				dataset = train_buckets[bucket_idx]
				np.random.shuffle(dataset)
				data_batch = dataset[:args.batchsize]

				source_batch, target_batch = make_source_target_pair(data_batch)

				if args.gpu_device >= 0:
					source_batch = cuda.to_gpu(source_batch)
					target_batch = cuda.to_gpu(target_batch)

				# update params
				model.reset_state()
				y_batch = model(source_batch)
				loss = F.softmax_cross_entropy(y_batch, target_batch, ignore_label=ID_PAD)
				optimizer.update(lossfun=lambda: loss)

				# show log
				printr("iteration {}/{}".format(itr + 1, total_iterations))

		save_model(args.model_dir, model)

		# clear console
		printr("")

		# compute perplexity
		with chainer.using_config("train", False):
			if dev_buckets is not None:
				printb("	ppl (dev)")
				ppl_dev = compute_perplexity(model, dev_buckets, args.batchsize)
				print("	", mean(ppl_dev), ppl_dev)

		# show log
		elapsed_time = (time.time() - start_time) / 60.
		total_time += elapsed_time
		print("	done in {} min, lr = {}, total {} min".format(int(elapsed_time), get_current_learning_rate(optimizer), int(total_time)))

		# decay learning rate
		decay_learning_rate(optimizer, args.lr_decay_factor, final_learning_rate)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=96)
	parser.add_argument("--epoch", "-e", type=int, default=1000)
	parser.add_argument("--grad-clip", "-gc", type=float, default=1) 
	parser.add_argument("--weight-decay", "-wd", type=float, default=1e-5) 
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.01)
	parser.add_argument("--lr-decay-factor", "-decay", type=float, default=0.95)
	parser.add_argument("--momentum", "-mo", type=float, default=0.99)
	parser.add_argument("--optimizer", "-opt", type=str, default="adam")
	
	parser.add_argument("--kernel-size", "-ksize", type=int, default=4)
	parser.add_argument("--ndim-h", "-nh", type=int, default=640)
	parser.add_argument("--ndim-embedding", "-ne", type=int, default=640)
	parser.add_argument("--num-layers", "-layers", type=int, default=2)
	parser.add_argument("--pooling", "-p", type=str, default="fo")
	parser.add_argument("--wgain", "-w", type=float, default=1)

	parser.add_argument("--densely-connected", "-dense", default=False, action="store_true")
	parser.add_argument("--zoneout", "-zoneout", type=float, default=0)
	parser.add_argument("--dropout", "-dropout", type=float, default=0)
	parser.add_argument("--weightnorm", "-weightnorm", default=False, action="store_true")
	
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--interval", type=int, default=100)
	parser.add_argument("--buckets-slice", type=int, default=None)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	parser.add_argument("--train-filename", "-train", default=None)
	parser.add_argument("--dev-filename", "-dev", default=None)
	args = parser.parse_args()
	main()