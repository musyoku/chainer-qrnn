# coding: utf-8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import argparse, sys, os, codecs, random, math, time
import numpy as np
import chainer
import chainer.functions as F
from chainer import training, Variable, optimizers, cuda
from chainer.training import extensions
from model import RNNModel, load_model, save_model, save_vocab
from common import ID_UNK, ID_PAD, ID_BOS, ID_EOS, bucket_sizes, stdout, print_bold
from dataset import read_data, make_buckets, sample_batch_from_bucket, make_source_target_pair
from error import compute_accuracy, compute_random_accuracy, compute_perplexity, compute_random_perplexity, softmax_cross_entropy

def get_current_learning_rate(opt):
	if isinstance(opt, optimizers.NesterovAG):
		return opt.lr
	if isinstance(opt, optimizers.Adam):
		return opt.alpha
	raise NotImplementationError()

def get_optimizer(name, lr, momentum):
	if name == "nesterov":
		return optimizers.NesterovAG(lr=lr, momentum=momentum)
	if name == "adam":
		return optimizers.Adam(alpha=lr, beta1=momentum)
	raise NotImplementationError()

def decay_learning_rate(opt, factor, final_value):
	if isinstance(opt, optimizers.NesterovAG):
		if opt.lr <= final_value:
			return
		opt.lr *= factor
		return
	if isinstance(opt, optimizers.Adam):
		if opt.alpha <= final_value:
			return
		opt.alpha *= factor
		return
	raise NotImplementationError()

def main(args):
	# load textfile
	dataset_train, dataset_dev, _, vocab, vocab_inv = read_data(args.train_filename, args.dev_filename)
	save_vocab(args.model_dir, vocab, vocab_inv)
	vocab_size = len(vocab)
	print_bold("data	#	hash")
	print("train	{}	{}".format(len(dataset_train), hash(str(dataset_train))))
	if len(dataset_dev) > 0:
		print("dev	{}	{}".format(len(dataset_dev), hash(str(dataset_dev))))
	print("vocab	{}".format(vocab_size))

	# split into buckets
	train_buckets = make_buckets(dataset_train)

	print_bold("buckets	#data	(train)")
	if args.buckets_slice is not None:
		train_buckets = train_buckets[:args.buckets_slice + 1]
	for size, data in zip(bucket_sizes, train_buckets):
		print("{}	{}".format(size, len(data)))

	dev_buckets = None
	if len(dataset_dev) > 0:
		print_bold("buckets	#data	(dev)")
		dev_buckets = make_buckets(dataset_dev)
		if args.buckets_slice is not None:
			dev_buckets = dev_buckets[:args.buckets_slice + 1]
		for size, data in zip(bucket_sizes, dev_buckets):
			print("{}	{}".format(size, len(data)))

	# to maintain equilibrium
	repeats = []
	for data in train_buckets:
		repeat = len(data) // args.batchsize
		repeats.append(repeat)

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
	for epoch in xrange(1, args.epoch + 1):
		print("Epoch", epoch)
		start_time = time.time()

		with chainer.using_config("train", True):
			for bucket_idx, (repeat, dataset) in enumerate(zip(repeats, train_buckets)):
				for itr in xrange(repeat):
					data_batch = dataset[:args.batchsize]
					source_batch, target_batch = make_source_target_pair(data_batch)

					if model.xp is cuda.cupy:
						source_batch = cuda.to_gpu(source_batch)
						target_batch = cuda.to_gpu(target_batch)

					model.reset_state()
					y_batch = model(source_batch)
					loss = softmax_cross_entropy(y_batch, target_batch, ignore_label=ID_PAD)
					optimizer.update(lossfun=lambda: loss)

					dataset = np.roll(dataset, args.batchsize)	# shift
					
					sys.stdout.write("\r" + stdout.CLEAR)
					sys.stdout.write("\rbucket {}/{} - iteration {}/{}".format(bucket_idx + 1, len(train_buckets), itr + 1, repeat))
					sys.stdout.flush()

				train_buckets[bucket_idx] = dataset

			# shuffle
			for bucket_idx in xrange(len(train_buckets)):
				np.random.shuffle(train_buckets[bucket_idx])

		# serialize
		save_model(args.model_dir, model)

		# show log
		sys.stdout.write("\r" + stdout.CLEAR)
		sys.stdout.flush()

		with chainer.using_config("train", False):
			print_bold("	accuracy (train)")
			acc_train = compute_random_accuracy(model, train_buckets, args.batchsize)
			print("	", mean(acc_train), acc_train)

			if dev_buckets is not None:
				print_bold("	accuracy (dev)")
				acc_dev = compute_accuracy(model, dev_buckets, args.batchsize)
				print("	", mean(acc_dev), acc_dev)

			print_bold("	ppl (train)")
			ppl_train = compute_random_perplexity(model, train_buckets, args.batchsize)
			print("	", mean(ppl_train), ppl_train)

			if dev_buckets is not None:
				print_bold("	ppl (dev)")
				ppl_dev = compute_perplexity(model, dev_buckets, args.batchsize)
				print("	", mean(ppl_dev), ppl_dev)

		elapsed_time = (time.time() - start_time) / 60.
		total_time += elapsed_time
		print("	done in {} min, lr = {}, total {} min".format(int(elapsed_time), get_current_learning_rate(optimizer), int(total_time)))

		# decay learning rate
		decay_learning_rate(optimizer, args.lr_decay, final_learning_rate)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=24)
	parser.add_argument("--epoch", "-e", type=int, default=1000)
	parser.add_argument("--grad-clip", "-gc", type=float, default=1) 
	parser.add_argument("--weight-decay", "-wd", type=float, default=2e-5) 
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.1)
	parser.add_argument("--lr-decay", "-decay", type=float, default=0.95)
	parser.add_argument("--momentum", "-mo", type=float, default=0.99)
	parser.add_argument("--optimizer", "-opt", type=str, default="nesterov")
	
	parser.add_argument("--kernel-size", "-ksize", type=int, default=4)
	parser.add_argument("--ndim-h", "-nh", type=int, default=640)
	parser.add_argument("--ndim-embedding", "-ne", type=int, default=640)
	parser.add_argument("--num-layers", "-layers", type=int, default=2)
	parser.add_argument("--pooling", "-p", type=str, default="fo")
	parser.add_argument("--wgain", "-w", type=float, default=1)

	parser.add_argument("--densely-connected", "-dense", default=False, action="store_true")
	parser.add_argument("--zoneout", "-zoneout", default=False, action="store_true")
	parser.add_argument("--dropout", "-dropout", default=False, action="store_true")
	parser.add_argument("--weightnorm", "-weightnorm", default=False, action="store_true")
	
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--interval", type=int, default=100)
	parser.add_argument("--buckets-slice", type=int, default=None)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	parser.add_argument("--train-filename", "-train", default=None)
	parser.add_argument("--dev-filename", "-dev", default=None)
	args = parser.parse_args()
	main(args)