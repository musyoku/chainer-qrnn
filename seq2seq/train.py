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
from model import Seq2Seq, load_model, save_model, save_vocab

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"

def print_bold(str):
	print(stdout.BOLD + str + stdout.END)

# reference
# https://www.tensorflow.org/tutorials/seq2seq

bucket_sizes = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 110), (200, 210)]
ID_PAD = -1
ID_GO = 0
ID_UNK = 1
ID_EOS = 2

def read_data(source_filename, target_filename, train_split_ratio=0.9, dev_split_ratio=0.05, seed=0, reverse=True):
	assert(train_split_ratio + dev_split_ratio <= 1)
	vocab = {
		"<go>": ID_GO,
		"<unk>": ID_UNK,
		"<eos>": ID_EOS,
	}
	source_dataset = []
	with codecs.open(source_filename, "r", "utf-8") as f:
		for sentence in f:
			sentence = sentence.strip()
			if len(sentence) == 0:
				continue
			word_ids = []
			words = sentence.split(" ")
			for word in words:
				if word not in vocab:
					vocab[word] = len(vocab)
				word_id = vocab[word]
				word_ids.append(word_id)
			if reverse:
				word_ids.reverse()
			source_dataset.append(word_ids)

	target_dataset = []
	with codecs.open(target_filename, "r", "utf-8") as f:
		for sentence in f:
			sentence = sentence.strip()
			if len(sentence) == 0:
				continue
			word_ids = [ID_GO]
			words = sentence.split(" ")
			for word in words:
				if word not in vocab:
					vocab[word] = len(vocab)
				word_id = vocab[word]
				word_ids.append(word_id)
			word_ids.append(ID_EOS)
			target_dataset.append(word_ids)

	vocab_inv = {}
	for word, word_id in vocab.items():
		vocab_inv[word_id] = word

	assert len(target_dataset) == len(source_dataset)

	random.seed(seed)
	random.shuffle(source_dataset)
	random.seed(seed)
	random.shuffle(target_dataset)

	# [train][validation] | [test]
	train_split = int(len(source_dataset) * (train_split_ratio + dev_split_ratio))
	source_train_dev = source_dataset[:train_split]
	target_train_dev = target_dataset[:train_split]
	source_test = source_dataset[train_split:]
	target_test = target_dataset[train_split:]

	# [train] | [validation]
	dev_split = int(len(source_train_dev) * dev_split_ratio)
	source_dev = source_train_dev[:dev_split]
	target_dev = target_train_dev[:dev_split]
	source_train = source_train_dev[dev_split:]
	target_train = target_train_dev[dev_split:]

	assert len(source_train) == len(target_train)
	assert len(source_dev) == len(target_dev)
	assert len(source_test) == len(target_test)

	return (source_train, source_dev, source_test), (target_train, target_dev, target_test), vocab, vocab_inv

# input:
# [34, 1093, 22504, 16399]
# [0, 202944, 205277, 144530, 111190, 205428, 186775, 111190, 205601, 58779, 2]
# output:
# [-1, -1, -1, -1, -1, -1, 34, 1093, 22504, 16399]
# [0, 202944, 205277, 144530, 111190, 205428, 186775, 111190, 205601, 58779, 2, -1]
def make_buckets(source, target):
	buckets_list_source = [[] for _ in xrange(len(bucket_sizes))]
	buckets_list_target = [[] for _ in xrange(len(bucket_sizes))]
	for word_ids_source, word_ids_target in zip(source, target):
		source_length = len(word_ids_source)
		target_length = len(word_ids_target)
		# source
		bucket_index_source = 0
		for size in bucket_sizes:
			if source_length <= size[0]:
				break
			bucket_index_source += 1
		# target
		bucket_index_target = 0
		for size in bucket_sizes:
			if target_length <= size[1]:
				break
			bucket_index_target += 1

		bucket_index = max(bucket_index_source, bucket_index_target)
		if bucket_index >= len(bucket_sizes):
			continue	# ignore long sequence

		source_size, target_size = bucket_sizes[bucket_index]
		
		for _ in xrange(max(source_size - source_length, 0)):
			word_ids_source.insert(0, ID_PAD)	# prepend
		assert len(word_ids_source) == source_size
		
		for _ in xrange(max(target_size - target_length, 0)):
			word_ids_target.append(ID_PAD)
		assert len(word_ids_target) == target_size

		buckets_list_source[bucket_index].append(word_ids_source)
		buckets_list_target[bucket_index].append(word_ids_target)

	buckets_source = []
	buckets_target = []
	for bucket_source, bucket_target in zip(buckets_list_source, buckets_list_target):
		if len(bucket_source) == 0:
			continue
		buckets_source.append(np.asarray(bucket_source).astype(np.int32))
		buckets_target.append(np.asarray(bucket_target).astype(np.int32))
	return buckets_source, buckets_target

def sample_batch_from_bucket(source_bucket, target_bucket, num_samples):
	num_samples = num_samples if len(bucket) >= num_samples else len(bucket)
	indices = np.random.choice(np.arange(len(bucket), dtype=np.int32), size=num_samples, replace=False)
	return source_bucket[indices], target_bucket[indices]

def make_source_target_pair(batch):
	source = batch[:, :-1]
	target = batch[:, 1:]
	target = np.reshape(target, (-1,))
	return source, target

def compute_accuracy_batch(model, batch):
	source, target = make_source_target_pair(batch)
	if model.xp is cuda.cupy:
		source = cuda.to_gpu(source)
		target = cuda.to_gpu(target)
	model.reset_state()
	Y = model(source, test=True)
	return float(F.accuracy(Y, target, ignore_label=ID_PAD).data)

def compute_accuracy(model, buckets, batchsize=100):
	acc = []
	for dataset in buckets:
		# split into minibatch
		if len(dataset) > batchsize:
			num_sections = len(dataset) // batchsize
			indices = [(i + 1) * batchsize for i in xrange(num_sections)]
			sections = np.split(dataset, indices, axis=0)
		else:
			sections = [dataset]
		# compute accuracy
		for batch in sections:
			acc.append(compute_accuracy_batch(model, batch))
	return reduce(lambda x, y: x + y, acc) / len(acc)

def compute_minibatch_accuracy(model, buckets, batchsize=100):
	acc = []
	for dataset in buckets:
		batch = sample_batch_from_bucket(dataset, batchsize)
		acc.append(compute_accuracy_batch(model, batch))
	return reduce(lambda x, y: x + y, acc) / len(acc)

def compute_perplexity_batch(model, batch):
	sum_log_likelihood = 0
	source, target = make_source_target_pair(batch)
	xp = model.xp
	if xp is cuda.cupy:
		source = cuda.to_gpu(source)
		target = cuda.to_gpu(target)
	model.reset_state()
	Y = F.softmax(model(source, test=True))
	Y.unchain_backward()
	P = Y.data[xp.arange(0, len(target)), target] + 1e-32
	log_P = xp.log(P)
	mask = target != ID_PAD
	log_P *= mask
	num_tokens = xp.count_nonzero(mask)
	mean_log_P = xp.sum(log_P) / num_tokens
	return math.exp(-float(mean_log_P))

	# batchsize = batch.shape[0]
	# seq_batch = xp.split(Y, batchsize)
	# target_batch = xp.split(target, batchsize)
	# for seq, target in zip(seq_batch, target_batch):
	# 	assert len(seq) == len(target)
	# 	log_likelihood = 0
	# 	num_tokens = 0
	# 	for t in xrange(len(seq)):
	# 		if target[t] == ID_PAD:
	# 			break
	# 		log_likelihood += math.log(seq[t, target[t]] + 1e-32)
	# 		num_tokens += 1
	# 	assert num_tokens > 0
	# 	sum_log_likelihood += log_likelihood / num_tokens
	# return math.exp(-sum_log_likelihood / batchsize)

def compute_perplexity(model, buckets, batchsize=100):
	ppl = []
	for dataset in buckets:
		# split into minibatch
		if len(dataset) > batchsize:
			num_sections = len(dataset) // batchsize
			indices = [(i + 1) * batchsize for i in xrange(num_sections)]
			sections = np.split(dataset, indices, axis=0)
		else:
			sections = [dataset]
		# compute accuracy
		for batch in sections:
			ppl.append(compute_perplexity_batch(model, batch))
	return reduce(lambda x, y: x + y, ppl) / len(ppl)

def compute_minibatch_perplexity(model, buckets, batchsize=100):
	ppl = []
	for dataset in buckets:
		batch = sample_batch_from_bucket(dataset, batchsize)
		ppl.append(compute_perplexity_batch(model, batch))
	return reduce(lambda x, y: x + y, ppl) / len(ppl)

def main(args):
	# load textfile
	source_dataset, target_dataset, vocab, vocab_inv = read_data(args.source_filename, args.target_filename)
	save_vocab(args.model_dir, vocab, vocab_inv)
	vocab_size = len(vocab)

	source_dataset_train, source_dataset_dev, source_dataset_test = source_dataset
	target_dataset_train, target_dataset_dev, target_dataset_test = target_dataset
	print_bold("data	#")
	print("train	{}".format(len(source_dataset_train)))
	print("dev	{}".format(len(source_dataset_dev)))
	print("test	{}".format(len(source_dataset_test)))
	print("vocab	{}".format(vocab_size))

	# split into buckets
	source_buckets_train, target_buckets_train = make_buckets(source_dataset_train, target_dataset_train)
	print_bold("buckets 	#data	(train)")
	for size, data in zip(bucket_sizes, source_buckets_train):
		print("{} 	{}".format(size, len(data)))
	print_bold("buckets 	#data	(dev)")
	source_buckets_dev, target_buckets_dev = make_buckets(source_dataset_dev, target_dataset_dev)
	for size, data in zip(bucket_sizes, source_buckets_dev):
		print("{} 	{}".format(size, len(data)))
	print_bold("buckets		#data	(test)")
	source_buckets_test, target_buckets_test = make_buckets(source_dataset_test, target_dataset_test)
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
		model = Seq2Seq(vocab_size, args.ndim_embedding, args.num_layers, ndim_h=args.ndim_h, pooling=args.pooling, zoneout=args.zoneout, wstd=args.wstd)
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
	num_iteration = len(train_dataset) // args.batchsize
	for epoch in xrange(1, args.epoch + 1):
		print("Epoch", epoch)
		for itr in xrange(1, num_iteration + 1):
			for repeat, source_bucket, target_bucket in zip(repeats, source_buckets_train, target_buckets_train):
				for r in xrange(repeat):
					source_batch, target_batch = sample_batch_from_bucket(source_bucket, target_bucket, args.batchsize)
					skip_mask = source_batch != ID_PAD
					print(skip_mask)
					source, target = make_source_target_pair(batch)
					if model.xp is cuda.cupy:
						source = cuda.to_gpu(source)
						target = cuda.to_gpu(target)
					model.reset_state()
					Y = model(source)
					loss = F.softmax_cross_entropy(Y, target, ignore_label=ID_PAD)
					optimizer.update(lossfun=lambda: loss)

				sys.stdout.write("\r{} / {}".format(itr, num_iteration))
				sys.stdout.flush()

			if itr % args.interval == 0:
				print("\raccuracy: {} (train), {} (dev)".format(compute_minibatch_accuracy(model, train_buckets, args.batchsize), compute_accuracy(model, validation_buckets, args.batchsize)))
				print("\rppl: {} (train), {} (dev)".format(compute_minibatch_perplexity(model, train_buckets, args.batchsize), compute_perplexity(model, validation_buckets, args.batchsize)))
				save_model(args.model_dir, model)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=50)
	parser.add_argument("--epoch", "-e", type=int, default=30)
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--grad-clip", "-gc", type=float, default=5) 
	parser.add_argument("--weight-decay", "-wd", type=float, default=2e-4) 
	parser.add_argument("--ndim-h", "-nh", type=int, default=640)
	parser.add_argument("--ndim-embedding", "-ne", type=int, default=320)
	parser.add_argument("--num-layers", "-layers", type=int, default=2)
	parser.add_argument("--interval", type=int, default=100)
	parser.add_argument("--pooling", "-p", type=str, default="fo")
	parser.add_argument("--wstd", "-w", type=float, default=1)
	parser.add_argument("--source-filename", "-source", default=None)
	parser.add_argument("--target-filename", "-target", default=None)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	parser.add_argument("--zoneout", default=False, action="store_true")
	parser.add_argument("--eve", default=False, action="store_true")
	args = parser.parse_args()
	main(args)