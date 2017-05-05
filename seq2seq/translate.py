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
from model import load_model, load_vocab, Seq2SeqModel, AttentiveSeq2SeqModel
from common import ID_UNK, ID_PAD, ID_GO, ID_EOS, bucket_sizes, stdout, print_bold
from dataset import sample_batch_from_bucket

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

def _translate_batch(model, source_batch, max_predict_length, vocab_inv_source, vocab_inv_target, argmax=True, source_reversed=True):
	xp = model.xp
	skip_mask = source_batch != ID_PAD
	batchsize = source_batch.shape[0]

	# to gpu
	if xp is cuda.cupy:
		source_batch = cuda.to_gpu(source_batch)
		skip_mask = cuda.to_gpu(skip_mask)

	word_ids = xp.arange(0, len(vocab_inv_target), dtype=xp.int32)

	model.reset_state()
	token = ID_GO
	x = xp.asarray([[token]]).astype(xp.int32)
	x = xp.broadcast_to(x, (batchsize, 1))

	# get encoder's last hidden states
	if isinstance(model, AttentiveSeq2SeqModel):
		encoder_last_hidden_states, encoder_last_layer_outputs = model.encode(source_batch, skip_mask, test=True)
	else:
		encoder_last_hidden_states = model.encode(source_batch, skip_mask, test=True)

	while x.shape[1] < max_predict_length:
		if isinstance(model, AttentiveSeq2SeqModel):
			u = model.decode_one_step(x, encoder_last_hidden_states, encoder_last_layer_outputs, skip_mask, test=True)
		else:
			u = model.decode_one_step(x, encoder_last_hidden_states, test=True)
		p = F.softmax(u)	# convert to probability

		# concatenate
		if xp is np:
			x = xp.append(x, xp.zeros((batchsize, 1), dtype=xp.int32), axis=1)
		else:
			x = xp.concatenate((x, xp.zeros((batchsize, 1), dtype=xp.int32)), axis=1)

		for n in xrange(batchsize):
			pn = p.data[n]

			# argmax or sampling
			if argmax:
				token = xp.argmax(pn)
			else:
				token = xp.random.choice(word_ids, size=1, p=pn)[0]

			x[n, -1] = token

	return x

def show_translate_results(vocab_inv_source, vocab_inv_target, source_batch, translation_batch, target_batch=None, source_reversed=False):
	batchsize = source_batch.shape[0]
	for n in xrange(batchsize):
		sentence = []
		for token in source_batch[n]:
			token = int(token)	# to cpu
			if token == ID_PAD:
				continue
			word = vocab_inv_source[token]
			sentence.append(word)
		if source_reversed:
			sentence.reverse()
		print(">source: ", " ".join(sentence))

		if target_batch is not None:
			sentence = []
			for token in target_batch[n]:
				token = int(token)	# to cpu
				if token == ID_PAD:
					break
				if token == ID_EOS:
					break
				if token == ID_GO:
					continue
				word = vocab_inv_target[token]
				sentence.append(word)
			print(" target: ", " ".join(sentence))

		sentence = []
		for token in translation_batch[n]:
			token = int(token)	# to cpu
			if token == ID_EOS:
				break
			if token == ID_PAD:
				break
			if token == ID_GO:
				continue
			word = vocab_inv_target[token]
			sentence.append(word)
		print(" predict:", " ".join(sentence))

def show_source_target_translation(model, source_buckets, target_buckets, vocab_inv_source, vocab_inv_target, batchsize=100, argmax=True):
	for source_bucket, target_bucket in zip(source_buckets, target_buckets):
		num_calculation = 0
		sum_wer = 0

		if len(source_bucket) > batchsize:
			num_sections = len(source_bucket) // batchsize - 1
			if len(source_bucket) % batchsize > 0:
				num_sections += 1
			indices = [(i + 1) * batchsize for i in xrange(num_sections)]
			source_sections = np.split(source_bucket, indices, axis=0)
			target_sections = np.split(target_bucket, indices, axis=0)
		else:
			source_sections = [source_bucket]
			target_sections = [target_bucket]

		for source_batch, target_batch in zip(source_sections, target_sections):
			translation_batch = _translate_batch(model, source_batch, target_batch.shape[1] * 2, vocab_inv_source, vocab_inv_target, argmax=argmax)
			show_translate_results(vocab_inv_source, vocab_inv_target, source_batch, translation_batch, target_batch)

def show_source_translation(model, source_buckets, vocab_inv_source, vocab_inv_target, batchsize=100, argmax=True):
	for source_bucket in source_buckets:
		num_calculation = 0
		sum_wer = 0

		if len(source_bucket) > batchsize:
			num_sections = len(source_bucket) // batchsize - 1
			if len(source_bucket) % batchsize > 0:
				num_sections += 1
			indices = [(i + 1) * batchsize for i in xrange(num_sections)]
			source_sections = np.split(source_bucket, indices, axis=0)
		else:
			source_sections = [source_bucket]

		for source_batch in source_sections:
			translation_batch = _translate_batch(model, source_batch, source_batch.shape[1] * 2, vocab_inv_source, vocab_inv_target, argmax=argmax)
			show_translate_results(vocab_inv_source, vocab_inv_target, source_batch, translation_batch)

def show_random_source_target_translation(model, source_buckets, target_buckets, vocab_inv_source, vocab_inv_target, num_translate=100, argmax=True):
	xp = model.xp
	for source_bucket, target_bucket in zip(source_buckets, target_buckets):
		# sample minibatch
		source_batch, target_batch = sample_batch_from_bucket(source_bucket, target_bucket, num_translate)
		translation_batch = _translate_batch(model, source_batch, target_batch.shape[1] * 2, vocab_inv_source, vocab_inv_target, argmax=argmax)
		show_translate_results(vocab_inv_source, vocab_inv_target, source_batch, translation_batch, target_batch)

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

	show_source_translation(model, source_buckets, vocab_inv_source, vocab_inv_target)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--source-filename", "-source", default=None)
	parser.add_argument("--buckets-limit", type=int, default=None)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	args = parser.parse_args()
	main(args)