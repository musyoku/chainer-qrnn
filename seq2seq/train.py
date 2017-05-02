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
from model import QRNN, load_model, save_model, save_vocab

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"

def print_bold(str):
	print(stdout.BOLD + str + stdout.END)

bucket_sizes = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 110)]
ID_PAD = 0
ID_UNK = 1
ID_GO = 2
ID_EOS = 4

def read_data(source_filename, target_filename, train_split_ratio=0.9, validation_split_ratio=0.05, seed=0, reverse=True):
	assert(train_split_ratio + validation_split_ratio <= 1)
	vocab = {
		"<pad>": ID_PAD,
		"<unk>": ID_UNK,
		"<go>": ID_GO,
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
			word_ids.append(ID_EOS)
			source_dataset.append(word_ids)

	vocab_inv = {}
	for word, word_id in vocab.items():
		vocab_inv[word_id] = word

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

	return train_dataset, validation_dataset, test_dataset, vocab, vocab_inv

# input:
# [0, a, b, c, 1]
# [0, d, e, 1]
# output:
# [[0, a, b, c,  1]
#  [0, d, e, 1, -1]]
def make_buckets(dataset):
	max_length = 0
	for word_ids in dataset:
		if len(word_ids) > max_length:
			max_length = len(word_ids)
	bucket_sizes.append(max_length)
	buckets_list = [[] for _ in xrange(len(bucket_sizes))]
	for word_ids in dataset:
		length = len(word_ids)
		bucket_index = 0
		for size in bucket_sizes:
			if length <= size:
				if size - length > 0:
					for _ in xrange(size - length):
						word_ids.append(ID_PAD)
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
	train_dataset, validation_dataset, test_dataset, vocab, vocab_inv = read_data(args.text_filename)
	save_vocab(args.model_dir, vocab, vocab_inv)
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

	# to maintain equilibrium
	min_num_data = 0
	for data in train_buckets:
		if min_num_data == 0 or len(data) < min_num_data:
			min_num_data = len(data)
	repeats = []
	for data in train_buckets:
		repeats.append(len(data) // min_num_data)

	# init
	model = load_model(args.model_dir)
	if model is None:
		model = QRNN(vocab_size, args.ndim_embedding, args.num_layers, ndim_h=args.ndim_h, pooling=args.pooling, zoneout=args.zoneout, wstd=args.wstd)
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
			for repeat, dataset in zip(repeats, train_buckets):
				for r in xrange(repeat):
					batch = sample_batch_from_bucket(dataset, args.batchsize)
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
	parser.add_argument("--text-filename", "-f", default=None)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	parser.add_argument("--zoneout", default=False, action="store_true")
	parser.add_argument("--eve", default=False, action="store_true")
	args = parser.parse_args()
	main(args)