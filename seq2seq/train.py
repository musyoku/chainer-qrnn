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
from model import seq2seq, load_model, save_model, save_vocab

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"

def print_bold(str):
	print(stdout.BOLD + str + stdout.END)

# reference
# https://www.tensorflow.org/tutorials/seq2seq

bucket_sizes = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 110), (200, 210)]
ID_PAD = 0
ID_UNK = 1
ID_EOS = 2
ID_GO = 3

def read_data(source_filename, target_filename, train_split_ratio=0.9, dev_split_ratio=0.05, seed=0, reverse=True):
	assert(train_split_ratio + dev_split_ratio <= 1)
	vocab_source = {
		"<pad>": ID_PAD,
		"<unk>": ID_UNK,
	}
	vocab_target = {
		"<pad>": ID_PAD,
		"<unk>": ID_UNK,
		"<eos>": ID_EOS,
		"<go>": ID_GO,
	}
	source_dataset = []
	with codecs.open(source_filename, "r", "utf-8") as f:
		new_word_id = ID_UNK + 1
		for sentence in f:
			sentence = sentence.strip()
			if len(sentence) == 0:
				continue
			word_ids = []
			words = sentence.split(" ")
			for word in words:
				if word not in vocab_source:
					vocab_source[word] = new_word_id
					new_word_id += 1
				word_id = vocab_source[word]
				word_ids.append(word_id)
			if reverse:
				word_ids.reverse()
			source_dataset.append(word_ids)

	target_dataset = []
	with codecs.open(target_filename, "r", "utf-8") as f:
		new_word_id = ID_GO + 1
		for sentence in f:
			sentence = sentence.strip()
			if len(sentence) == 0:
				continue
			word_ids = [ID_GO]
			words = sentence.split(" ")
			for word in words:
				if word not in vocab_target:
					vocab_target[word] = new_word_id
					new_word_id += 1
				word_id = vocab_target[word]
				word_ids.append(word_id)
			word_ids.append(ID_EOS)
			target_dataset.append(word_ids)

	vocab_inv_source = {}
	for word, word_id in vocab_source.items():
		vocab_inv_source[word_id] = word

	vocab_inv_target = {}
	for word, word_id in vocab_target.items():
		vocab_inv_target[word_id] = word

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

	return (source_train, source_dev, source_test), (target_train, target_dev, target_test), (vocab_source, vocab_target), (vocab_inv_source, vocab_inv_target)

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
	assert len(source_bucket) == len(target_bucket)
	num_samples = num_samples if len(source_bucket) >= num_samples else len(source_bucket)
	indices = np.random.choice(np.arange(len(source_bucket), dtype=np.int32), size=num_samples, replace=False)
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
		model = seq2seq(len(vocab_source), len(vocab_target), args.ndim_embedding, args.num_layers, ndim_h=args.ndim_h, pooling=args.pooling, zoneout=args.zoneout, wstd=args.wstd, attention=False)
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
					encoder_hidden_states = model.encode(source_batch, skip_mask)
					Y = model.decode(target_batch_input, encoder_hidden_states)
					loss = F.softmax_cross_entropy(Y, target_batch_output, ignore_label=ID_PAD)
					optimizer.update(lossfun=lambda: loss)

				sys.stdout.write("\r{} / {}".format(itr, num_iteration))
				sys.stdout.flush()

			if itr % args.interval == 0:
				save_model(args.model_dir, model)
				model.to_cpu()
				sys.stdout.write("\r")
				sys.stdout.flush()
				for _, source_bucket, target_bucket in zip(repeats, source_buckets_train, target_buckets_train):
					source_batch, target_batch = sample_batch_from_bucket(source_bucket, target_bucket, 10)
					skip_mask = source_batch != ID_PAD
					word_ids = np.arange(0, len(vocab_target), dtype=np.int32)
					for n in xrange(len(source_batch)):
						model.reset_state()
						token = ID_GO
						x = np.asarray([[token]]).astype(np.int32)
						encoder_hidden_states = model.encode(source_batch[None, n, :], skip_mask[None, n, :], test=True)
						while token != ID_EOS and x.shape[1] < 50:
							model.reset_decoder_state()
							u = model.decode(x, encoder_hidden_states, test=True)
							p = F.softmax(u).data[-1]
							token = np.random.choice(word_ids, size=1, p=p)
							x = np.append(x, np.asarray([token]).astype(np.int32), axis=1)

						sentence = []
						for token in source_batch[n, :]:
							if token == ID_PAD:
								continue
							word = vocab_inv_source[token]
							sentence.append(word)
						sentence.reverse()
						print("source:", " ".join(sentence))

						sentence = []
						for token in x[0]:
							# if token == ID_EOS:
							# 	break
							# if token == ID_PAD:
							# 	break
							if token == ID_GO:
								continue
							word = vocab_inv_target[token]
							sentence.append(word)
						print("target:", " ".join(sentence))
				model.to_gpu()

		sys.stdout.write("\r")
		sys.stdout.flush()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=50)
	parser.add_argument("--epoch", "-e", type=int, default=100)
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--grad-clip", "-gc", type=float, default=5) 
	parser.add_argument("--weight-decay", "-wd", type=float, default=2e-4) 
	parser.add_argument("--ndim-h", "-nh", type=int, default=640)
	parser.add_argument("--ndim-embedding", "-ne", type=int, default=320)
	parser.add_argument("--num-layers", "-layers", type=int, default=2)
	parser.add_argument("--interval", type=int, default=100)
	parser.add_argument("--pooling", "-p", type=str, default="fo")
	parser.add_argument("--wstd", "-w", type=float, default=0.02)
	parser.add_argument("--source-filename", "-source", default=None)
	parser.add_argument("--target-filename", "-target", default=None)
	parser.add_argument("--buckets-limit", type=int, default=None)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	parser.add_argument("--zoneout", default=False, action="store_true")
	parser.add_argument("--eve", default=False, action="store_true")
	args = parser.parse_args()
	main(args)