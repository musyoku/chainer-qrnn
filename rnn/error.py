# coding: utf-8
import math, sys
import numpy as np
import chainer.functions as F
from chainer import cuda
from dataset import sample_batch_from_bucket, make_source_target_pair
from common import ID_UNK, ID_PAD, ID_BOS, ID_EOS, stdout

def compute_accuracy_batch(model, batch):
	source, target = make_source_target_pair(batch)
	if model.xp is cuda.cupy:
		source = cuda.to_gpu(source)
		target = cuda.to_gpu(target)
	model.reset_state()
	Y = model(source, test=True)
	return float(F.accuracy(Y, target, ignore_label=ID_PAD).data)

def compute_accuracy(model, buckets, batchsize=100):
	result = []
	for bucket_index, dataset in enumerate(buckets):
		acc = []
		# split into minibatch
		if len(dataset) > batchsize:
			num_sections = len(dataset) // batchsize - 1
			if len(dataset) % batchsize > 0:
				num_sections += 1
			indices = [(i + 1) * batchsize for i in xrange(num_sections)]
			sections = np.split(dataset, indices, axis=0)
		else:
			sections = [dataset]
		# compute accuracy
		for batch_index, batch in enumerate(sections):
			sys.stdout.write("\rcomputing accuracy ... bucket {}/{} (batch {}/{})".format(bucket_index + 1, len(buckets), batch_index + 1, len(sections)))
			sys.stdout.flush()
			acc.append(compute_accuracy_batch(model, batch))

		result.append(reduce(lambda x, y: x + y, acc) / len(acc))
		sys.stdout.write("\r" + stdout.CLEAR)
		sys.stdout.flush()

	return result

def compute_random_accuracy(model, buckets, batchsize=100):
	acc = []
	for dataset in buckets:
		batch = sample_batch_from_bucket(dataset, batchsize)
		acc.append(compute_accuracy_batch(model, batch))
	return acc

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
	result = []
	for bucket_index, dataset in enumerate(buckets):
		ppl = []
		# split into minibatch
		if len(dataset) > batchsize:
			num_sections = len(dataset) // batchsize - 1
			if len(dataset) % batchsize > 0:
				num_sections += 1
			indices = [(i + 1) * batchsize for i in xrange(num_sections)]
			sections = np.split(dataset, indices, axis=0)
		else:
			sections = [dataset]
		# compute accuracy
		for batch_index, batch in enumerate(sections):
			sys.stdout.write("\rcomputing perplexity ... bucket {}/{} (batch {}/{})".format(bucket_index + 1, len(buckets), batch_index + 1, len(sections)))
			sys.stdout.flush()
			ppl.append(compute_perplexity_batch(model, batch))

		result.append(reduce(lambda x, y: x + y, ppl) / len(ppl))
		sys.stdout.write("\r" + stdout.CLEAR)
		sys.stdout.flush()
	return result

def compute_random_perplexity(model, buckets, batchsize=100):
	ppl = []
	for dataset in buckets:
		batch = sample_batch_from_bucket(dataset, batchsize)
		ppl.append(compute_perplexity_batch(model, batch))
	return ppl
