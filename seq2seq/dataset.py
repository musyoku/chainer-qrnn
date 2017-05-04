# coding: utf-8
import codecs, random
import numpy as np
from common import ID_UNK, ID_PAD, ID_GO, ID_EOS, bucket_sizes

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
		i = 0
		new_word_id = ID_GO + 1
		for sentence in f:
			i += 1
			sentence = sentence.strip()
			if len(sentence) == 0:
				print(i)
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