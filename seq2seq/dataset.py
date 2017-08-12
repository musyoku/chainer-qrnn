# coding: utf-8
import codecs, random
import numpy as np
from common import ID_UNK, ID_PAD, ID_GO, ID_EOS, bucket_sizes

def read_data_and_vocab(source_filename_train=None, target_filename_train=None, source_filename_dev=None, target_filename_dev=None, source_filename_test=None, target_filename_test=None, reverse_source=True):
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
	source_dataset_train = []
	source_dataset_dev = []
	source_dataset_test = []
	target_dataset_train = []
	target_dataset_dev = []
	target_dataset_test = []

	def add_file(filename, vocab, dataset, prefix=None, suffix=None, reverse=False):
		if filename is None:
			return
		with codecs.open(filename, "r", "utf-8") as f:
			for sentence in f:
				sentence = sentence.strip()
				if len(sentence) == 0:
					raise Exception("empty line {}".format(len(dataset) + 1))
				word_ids = []
				if prefix:
					word_ids.append(prefix)
				words = sentence.split(" ")
				for word in words:
					if word not in vocab:
						vocab[word] = len(vocab)
					word_id = vocab[word]
					word_ids.append(word_id)
				if suffix:
					word_ids.append(suffix)
				if reverse:
					word_ids.reverse()
				dataset.append(word_ids)

	add_file(source_filename_train, vocab_source, source_dataset_train, reverse=reverse_source)
	add_file(source_filename_dev, vocab_source, source_dataset_dev, reverse=reverse_source)
	add_file(source_filename_test, vocab_source, source_dataset_test, reverse=reverse_source)

	add_file(target_filename_train, vocab_target, target_dataset_train, ID_GO, ID_EOS)
	add_file(target_filename_dev, vocab_target, target_dataset_dev, ID_GO, ID_EOS)
	add_file(target_filename_test, vocab_target, target_dataset_test, ID_GO, ID_EOS)

	assert len(source_dataset_train) == len(target_dataset_train)
	assert len(source_dataset_dev) == len(target_dataset_dev)
	assert len(source_dataset_test) == len(target_dataset_test)

	vocab_inv_source = {}
	for word, word_id in vocab_source.items():
		vocab_inv_source[word_id] = word

	vocab_inv_target = {}
	for word, word_id in vocab_target.items():
		vocab_inv_target[word_id] = word
		
	return (source_dataset_train, source_dataset_dev, source_dataset_test), (target_dataset_train, target_dataset_dev, target_dataset_test), (vocab_source, vocab_target), (vocab_inv_source, vocab_inv_target)

def read_data(vocab_source, vocab_target, source_filename_train=None, target_filename_train=None, source_filename_dev=None, target_filename_dev=None, source_filename_test=None, target_filename_test=None, reverse_source=True):
	source_dataset_train = []
	source_dataset_dev = []
	source_dataset_test = []
	target_dataset_train = []
	target_dataset_dev = []
	target_dataset_test = []

	def add_file(filename, vocab, dataset, prefix=None, suffix=None, reverse=False):
		if filename is None:
			return
		with codecs.open(filename, "r", "utf-8") as f:
			for sentence in f:
				sentence = sentence.strip()
				if len(sentence) == 0:
					continue
				word_ids = []
				if prefix:
					word_ids.append(prefix)
				words = sentence.split(" ")
				for word in words:
					if word in vocab:
						word_id = vocab[word]
					else:
						word_id = ID_UNK
					word_ids.append(word_id)
				if suffix:
					word_ids.append(suffix)
				if reverse:
					word_ids.reverse()
				dataset.append(word_ids)

	add_file(source_filename_train, vocab_source, source_dataset_train, reverse=reverse_source)
	add_file(source_filename_dev, vocab_source, source_dataset_dev, reverse=reverse_source)
	add_file(source_filename_test, vocab_source, source_dataset_test, reverse=reverse_source)

	add_file(target_filename_train, vocab_target, target_dataset_train, ID_GO, ID_EOS)
	add_file(target_filename_dev, vocab_target, target_dataset_dev, ID_GO, ID_EOS)
	add_file(target_filename_test, vocab_target, target_dataset_test, ID_GO, ID_EOS)
		
	return (source_dataset_train, source_dataset_dev, source_dataset_test), (target_dataset_train, target_dataset_dev, target_dataset_test)

# input:
# [34, 1093, 22504, 16399]
# [0, 202944, 205277, 144530, 111190, 205428, 186775, 111190, 205601, 58779, 2]
# output:
# [-1, -1, -1, -1, -1, -1, 34, 1093, 22504, 16399]
# [0, 202944, 205277, 144530, 111190, 205428, 186775, 111190, 205601, 58779, 2, -1]
def make_buckets(source, target):
	buckets_list_source = [[] for _ in range(len(bucket_sizes))]
	buckets_list_target = [[] for _ in range(len(bucket_sizes))]
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
		
		for _ in range(max(source_size - source_length, 0)):
			word_ids_source.insert(0, ID_PAD)	# prepend
		assert len(word_ids_source) == source_size
		
		for _ in range(max(target_size - target_length, 0)):
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