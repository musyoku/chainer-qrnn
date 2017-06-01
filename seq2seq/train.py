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
from common import ID_UNK, ID_PAD, ID_GO, ID_EOS, bucket_sizes, stdout, print_bold
from dataset import read_data_and_vocab, make_buckets, make_source_target_pair, sample_batch_from_bucket
from model import seq2seq, load_model, save_model, save_vocab
from error import compute_error_rate_buckets, compute_random_error_rate_buckets, softmax_cross_entropy
from translate import dump_random_source_target_translation

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

# reference
# https://www.tensorflow.org/tutorials/seq2seq

def main(args):
	# load textfile
	source_dataset, target_dataset, vocab, vocab_inv = read_data_and_vocab(args.source_train, args.target_train, args.source_dev, args.target_dev, args.source_test, args.target_test, reverse_source=True)
	save_vocab(args.model_dir, vocab, vocab_inv)

	source_dataset_train, source_dataset_dev, source_dataset_test = source_dataset
	target_dataset_train, target_dataset_dev, target_dataset_test = target_dataset
	print_bold("data	#")
	print("train	{}".format(len(source_dataset_train)))
	if len(source_dataset_dev) > 0:
		print("dev	{}".format(len(source_dataset_dev)))
	if len(source_dataset_test) > 0:
		print("test	{}".format(len(source_dataset_test)))

	vocab_source, vocab_target = vocab
	vocab_inv_source, vocab_inv_target = vocab_inv
	print("vocab	{}	(source)".format(len(vocab_source)))
	print("vocab	{}	(target)".format(len(vocab_target)))

	# split into buckets
	source_buckets_train, target_buckets_train = make_buckets(source_dataset_train, target_dataset_train)
	if args.buckets_slice is not None:
		source_buckets_train = source_buckets_train[:args.buckets_slice + 1]
		target_buckets_train = target_buckets_train[:args.buckets_slice + 1]

	print_bold("buckets 	#data	(train)")
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

	# to maintain equilibrium
	required_interations = []
	for data in source_buckets_train:
		itr = len(data) // args.batchsize + 1
		required_interations.append(itr)
	total_iterations = sum(required_interations)
	buckets_distribution = np.asarray(required_interations, dtype=float) / total_iterations

	# init
	model = load_model(args.model_dir)
	if model is None:
		model = seq2seq(len(vocab_source), len(vocab_target), args.ndim_embedding, args.ndim_h, args.num_layers, pooling=args.pooling, dropout=args.dropout, zoneout=args.zoneout, weightnorm=args.weightnorm, wgain=args.wgain, densely_connected=args.densely_connected, attention=args.attention)
	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		model.to_gpu()

	# setup an optimizer
	optimizer = get_optimizer(args.optimizer, args.learning_rate, args.momentum)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))
	optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
	final_learning_rate = 1e-5
	decay_factor = 0.95
	total_time = 0

	indices_train = []
	for bucket_idx, bucket in enumerate(source_buckets_train):
		indices = np.arange(len(bucket))
		np.random.shuffle(indices)
		indices_train.append(indices)

	def mean(l):
		return sum(l) / len(l)

	# training
	for epoch in xrange(1, args.epoch + 1):
		print("Epoch", epoch)
		start_time = time.time()

		with chainer.using_config("train", True):

			for itr in xrange(total_iterations):
				bucket_idx = int(np.random.choice(np.arange(len(source_buckets_train)), size=1, p=buckets_distribution))
				source_bucket = source_buckets_train[bucket_idx]
				target_bucket = target_buckets_train[bucket_idx]

				# sample minibatch
				source_batch = source_bucket[:args.batchsize]
				target_batch = target_bucket[:args.batchsize]
				skip_mask = source_batch != ID_PAD
				target_batch_input, target_batch_output = make_source_target_pair(target_batch)

				# to gpu
				if model.xp is cuda.cupy:
					skip_mask = cuda.to_gpu(skip_mask)
					source_batch = cuda.to_gpu(source_batch)
					target_batch_input = cuda.to_gpu(target_batch_input)
					target_batch_output = cuda.to_gpu(target_batch_output)

				# compute loss
				model.reset_state()
				if args.attention:
					last_hidden_states, last_layer_outputs = model.encode(source_batch, skip_mask)
					Y = model.decode(target_batch_input, last_hidden_states, last_layer_outputs, skip_mask)
				else:
					last_hidden_states = model.encode(source_batch, skip_mask)
					Y = model.decode(target_batch_input, last_hidden_states)
				loss = softmax_cross_entropy(Y, target_batch_output, ignore_label=ID_PAD)

				# update parameters
				optimizer.update(lossfun=lambda: loss)

				sys.stdout.write("\r" + stdout.CLEAR)
				sys.stdout.write("\riteration {}/{}".format(itr + 1, total_iterations))
				sys.stdout.flush()

				source_buckets_train[bucket_idx] = np.roll(source_bucket, -args.batchsize, axis=0)	# shift
				target_buckets_train[bucket_idx] = np.roll(target_bucket, -args.batchsize, axis=0)	# shift

			# shuffle
			for bucket_idx in xrange(len(source_buckets_train)):
				indices = indices_train[bucket_idx]
				np.random.shuffle(indices)
				source_buckets_train[bucket_idx] = source_buckets_train[bucket_idx][indices]
				target_buckets_train[bucket_idx] = target_buckets_train[bucket_idx][indices]

		# serialize
		save_model(args.model_dir, model)

		# show log
		with chainer.using_config("train", False):
			sys.stdout.write("\r" + stdout.CLEAR)
			sys.stdout.flush()

			if epoch % args.interval == 0:
				print_bold("translate (train)")
				dump_random_source_target_translation(model, source_buckets_train, target_buckets_train, vocab_inv_source, vocab_inv_target, num_translate=5, beam_width=1)

				if source_dataset_dev is not None:
					print_bold("translate (dev)")
					dump_random_source_target_translation(model, source_buckets_dev, target_buckets_dev, vocab_inv_source, vocab_inv_target, num_translate=5, beam_width=1)

				if source_dataset_dev is not None:
					print_bold("WER (dev)")
					wer_dev = compute_error_rate_buckets(model, source_buckets_dev, target_buckets_dev, len(vocab_inv_target), beam_width=1)
					print(mean(wer_dev), wer_dev)

			elapsed_time = (time.time() - start_time) / 60.
			total_time += elapsed_time
			print("done in {} min, lr = {}, total {} min".format(int(elapsed_time), get_current_learning_rate(optimizer), int(total_time)))

		# decay learning rate
		decay_learning_rate(optimizer, decay_factor, final_learning_rate)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--source-train", type=str, default=None)
	parser.add_argument("--source-dev", type=str, default=None)
	parser.add_argument("--source-test", type=str, default=None)
	parser.add_argument("--target-train", type=str, default=None)
	parser.add_argument("--target-dev", type=str, default=None)
	parser.add_argument("--target-test", type=str, default=None)

	parser.add_argument("--batchsize", "-b", type=int, default=64)
	parser.add_argument("--epoch", "-e", type=int, default=1000)
	parser.add_argument("--interval", type=int, default=10)
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--grad-clip", "-gc", type=float, default=0.1) 
	parser.add_argument("--weight-decay", "-wd", type=float, default=2e-4) 
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.1)
	parser.add_argument("--momentum", "-mo", type=float, default=0.99)
	parser.add_argument("--optimizer", "-opt", type=str, default="nesterov")

	parser.add_argument("--ndim-h", "-nh", type=int, default=320)
	parser.add_argument("--ndim-embedding", "-ne", type=int, default=320)
	parser.add_argument("--num-layers", "-layers", type=int, default=4)
	parser.add_argument("--pooling", "-p", type=str, default="fo")
	parser.add_argument("--wgain", "-w", type=float, default=1)
	parser.add_argument("--zoneout", "-zoneout", type=float, default=0)
	parser.add_argument("--dropout", "-dropout", type=float, default=0)
	parser.add_argument("--densely-connected", "-dense", default=False, action="store_true")
	parser.add_argument("--weightnorm", "-weightnorm", default=False, action="store_true")
	parser.add_argument("--attention", "-attention", default=False, action="store_true")
	
	parser.add_argument("--buckets-slice", type=int, default=None)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	args = parser.parse_args()
	main(args)