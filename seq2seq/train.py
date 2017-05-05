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
from common import ID_UNK, ID_PAD, ID_GO, ID_EOS, bucket_sizes, stdout, print_bold
from dataset import read_data, make_buckets, make_source_target_pair, sample_batch_from_bucket
from eve import Eve
from model import seq2seq, load_model, save_model, save_vocab
from error import compute_mean_wer, compute_random_mean_wer
from translate import show_random_source_target_translation

# reference
# https://www.tensorflow.org/tutorials/seq2seq

def main(args):
	# load textfile
	source_dataset, target_dataset, vocab, vocab_inv = read_data(args.source_filename, args.target_filename, train_split_ratio=args.train_split, dev_split_ratio=args.dev_split)
	save_vocab(args.model_dir, vocab, vocab_inv)

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

	# to maintain equilibrium
	min_num_data = 0
	for data in source_buckets_train:
		if min_num_data == 0 or len(data) < min_num_data:
			min_num_data = len(data)
	repeats = []
	for data in source_buckets_train:
		repeats.append(len(data) // min_num_data)

	# init
	model = load_model(args.model_dir)
	if model is None:
		model = seq2seq(len(vocab_source), len(vocab_target), args.ndim_embedding, args.num_layers, ndim_h=args.ndim_h, pooling=args.pooling, zoneout=args.zoneout, wstd=args.wstd, attention=args.attention)
	if args.gpu_device >= 0:
		chainer.cuda.get_device(args.gpu_device).use()
		model.to_gpu()

	# setup an optimizer
	if args.eve:
		optimizer = Eve(alpha=0.001, beta1=0.9)
	else:
		optimizer = optimizers.Adam(alpha=0.001, beta1=0.9)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))
	optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

	# training
	num_iteration = len(source_dataset_train) // args.batchsize + 1
	for epoch in xrange(1, args.epoch + 1):
		print("Epoch", epoch)
		for itr in xrange(1, num_iteration + 1):
			for repeat, source_bucket, target_bucket in zip(repeats, source_buckets_train, target_buckets_train):
				for r in xrange(repeat):
					# sample minibatch
					source_batch, target_batch = sample_batch_from_bucket(source_bucket, target_bucket, args.batchsize)
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
					loss = F.softmax_cross_entropy(Y, target_batch_output, ignore_label=ID_PAD)
					optimizer.update(lossfun=lambda: loss)

				sys.stdout.write("\r{} / {}".format(itr, num_iteration))
				sys.stdout.flush()

			if itr % args.interval == 0:
				save_model(args.model_dir, model)
				sys.stdout.write("\r" + stdout.CLEAR)
				sys.stdout.flush()
				print_bold("translate (train)")
				show_random_source_target_translation(model, source_buckets_train, target_buckets_train, vocab_inv_source, vocab_inv_target, num_translate=5, argmax=True)
				print_bold("translate (dev)")
				show_random_source_target_translation(model, source_buckets_dev, target_buckets_dev, vocab_inv_source, vocab_inv_target, num_translate=5, argmax=True)
				print_bold("WER (sampled train)")
				wer_train = compute_random_mean_wer(model, source_buckets_train, target_buckets_train, len(vocab_inv_target), sample_size=args.batchsize, argmax=True)
				print(wer_train)
				print_bold("WER (dev)")
				wer_dev = compute_mean_wer(model, source_buckets_dev, target_buckets_dev, len(vocab_inv_target), batchsize=args.batchsize, argmax=True)
				print(wer_dev)

		sys.stdout.write("\r")
		sys.stdout.flush()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=50)
	parser.add_argument("--epoch", "-e", type=int, default=100)
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--grad-clip", "-gc", type=float, default=5) 
	parser.add_argument("--weight-decay", "-wd", type=float, default=2e-5) 
	parser.add_argument("--ndim-h", "-nh", type=int, default=640)
	parser.add_argument("--ndim-embedding", "-ne", type=int, default=320)
	parser.add_argument("--num-layers", "-layers", type=int, default=2)
	parser.add_argument("--interval", type=int, default=100)
	parser.add_argument("--pooling", "-p", type=str, default="fo")
	parser.add_argument("--wstd", "-w", type=float, default=0.02)
	parser.add_argument("--train-split", type=float, default=0.9)
	parser.add_argument("--dev-split", type=float, default=0.05)
	parser.add_argument("--source-filename", "-source", default=None)
	parser.add_argument("--target-filename", "-target", default=None)
	parser.add_argument("--buckets-limit", type=int, default=None)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	parser.add_argument("--zoneout", default=False, action="store_true")
	parser.add_argument("--eve", default=False, action="store_true")
	parser.add_argument("--attention", default=False, action="store_true")
	args = parser.parse_args()
	main(args)