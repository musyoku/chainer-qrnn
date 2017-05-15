# coding: utf-8
import codecs, random
import numpy as np
from common import ID_UNK, ID_PAD, ID_BOS, ID_EOS, bucket_sizes

def read_data(filename_train, filename_dev):
	vocab = {
		"<pad>": ID_PAD,
		"<unk>": ID_UNK,
		"<bos>": ID_BOS,
		"<eos>": ID_EOS,
	}
	dataset_train = []
	dataset_dev = []

	with codecs.open(filename_train, "r", "utf-8") as f:
		for sentence in f:
			sentence = sentence.strip()
			if len(sentence) == 0:
				continue
			word_ids = [ID_BOS]
			words = sentence.split(" ")
			for word in words:
				if word not in vocab:
					vocab[word] = len(vocab)
				word_id = vocab[word]
				word_ids.append(word_id)
			word_ids.append(ID_EOS)
			dataset_train.append(word_ids)

	if filename_dev is not None:
		with codecs.open(filename_dev, "r", "utf-8") as f:
			for sentence in f:
				sentence = sentence.strip()
				if len(sentence) == 0:
					continue
				word_ids = [ID_BOS]
				words = sentence.split(" ")
				for word in words:
					if word not in vocab:
						vocab[word] = len(vocab)
					word_id = vocab[word]
					word_ids.append(word_id)
				word_ids.append(ID_EOS)
				dataset_dev.append(word_ids)

	vocab_inv = {}
	for word, word_id in vocab.items():
		vocab_inv[word_id] = word

	return dataset_train, dataset_dev, vocab, vocab_inv

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