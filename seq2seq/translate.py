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
from model import seq2seq, load_model, load_vocab
from train import stdout, print_bold, bucket_sizes, ID_PAD, ID_UNK, ID_EOS, ID_GO

def read_data(source_filename, vocab_source, reverse=True):
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
				assert(word in vocab_source)
				word_id = vocab_source[word]
				word_ids.append(word_id)
			if reverse:
				word_ids.reverse()
			source_dataset.append(word_ids)

	return source_dataset

def make_buckets(dataset):
	buckets_list = [[] for _ in xrange(len(bucket_sizes))]
	for word_ids in dataset:
		length = len(word_ids)
		bucket_index = 0
		for size in bucket_sizes:
			if length <= size[0]:
				break
			bucket_index += 1
		if bucket_index >= len(bucket_sizes):
			continue	# ignore long sequence

		required_length, _ = bucket_sizes[bucket_index]
		
		for _ in xrange(max(required_length - length, 0)):
			word_ids.insert(0, ID_PAD)	# prepend
		assert len(word_ids) == required_length

		buckets_list[bucket_index].append(word_ids)

	buckets = []
	for bucket_source in buckets_list:
		if len(bucket_source) == 0:
			continue
		buckets.append(np.asarray(bucket_source).astype(np.int32))
	return buckets

def translate(model, buckets, vocab_inv_source, vocab_inv_target):
	for bucket in buckets:
		batch = bucket[:10]
		skip_mask = batch != ID_PAD
		word_ids = np.arange(0, len(vocab_inv_target), dtype=np.int32)
		for n in xrange(len(batch)):
			model.reset_state()
			token = ID_GO
			x = np.asarray([[token]]).astype(np.int32)
			encoder_hidden_states = model.encode(batch[None, n, :], skip_mask[None, n, :], test=True)
			while token != ID_EOS and x.shape[1] < 50:
				model.reset_decoder_state()
				u = model.decode(x, encoder_hidden_states, test=True)
				p = F.softmax(u).data[-1]
				token = np.random.choice(word_ids, size=1, p=p)
				x = np.append(x, np.asarray([token]).astype(np.int32), axis=1)

			sentence = []
			for token in batch[n, :]:
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

def translate(model, buckets, vocab_inv_source, vocab_inv_target):
	for bucket in buckets:
		batch = bucket[:10]
		skip_mask = batch != ID_PAD
		word_ids = np.arange(0, len(vocab_inv_target), dtype=np.int32)
		for n in xrange(len(batch)):
			model.reset_state()
			token = ID_GO
			x = np.asarray([[token]]).astype(np.int32)
			encoder_hidden_states = model.encode(batch[None, n, :], skip_mask[None, n, :], test=True)
			while token != ID_EOS and x.shape[1] < 50:
				model.reset_decoder_state()
				u = model.decode(x, encoder_hidden_states, test=True)
				p = F.softmax(u).data[-1]
				token = np.random.choice(word_ids, size=1, p=p)
				x = np.append(x, np.asarray([token]).astype(np.int32), axis=1)

			sentence = []
			for token in batch[n, :]:
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
			
def main(args):
	# load vocab
	vocab, vocab_inv = load_vocab(args.model_dir)
	vocab_source, vocab_target = vocab
	vocab_inv_source, vocab_inv_target = vocab_inv

	# load textfile
	source_dataset = read_data(args.source_filename, vocab_source)

	print_bold("data	#")
	print("source	{}".format(len(source_dataset)))

	# split into buckets
	source_buckets = make_buckets(source_dataset)
	if args.buckets_limit is not None:
		source_buckets = source_buckets[:args.buckets_limit+1]
	print_bold("buckets 	#data	(train)")
	for size, data in zip(bucket_sizes, source_buckets):
		print("{} 	{}".format(size, len(data)))
	print_bold("buckets 	#data	(dev)")

	# init
	model = load_model(args.model_dir)
	assert model is not None

	np.random.seed(0)
	translate(model, source_buckets, vocab_inv_source, vocab_inv_target)
	np.random.seed(0)
	_translate(model, source_buckets, vocab_inv_source, vocab_inv_target)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--source-filename", "-source", default=None)
	parser.add_argument("--buckets-limit", type=int, default=None)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	args = parser.parse_args()
	main(args)