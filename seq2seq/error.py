# coding: utf-8
import numpy as np
import chainer.functions as F
from chainer import cuda
from common import ID_UNK, ID_PAD, ID_GO, ID_EOS, bucket_sizes
from dataset import sample_batch_from_bucket

# https://github.com/zszyellow/WER-in-python
def compute_word_error_rate_of_sequence(r, h):
	# build the matrix
	d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape((len(r) + 1, len(h) + 1))
	for i in xrange(len(r) + 1):
		for j in xrange(len(h) + 1):
			if i == 0: d[0][j] = j
			elif j == 0: d[i][0] = i
	for i in xrange(1, len(r) + 1):
		for j in xrange(1, len(h) + 1):
			if r[i-1] == h[j-1]:
				d[i][j] = d[i-1][j-1]
			else:
				substitute = d[i-1][j-1] + 1
				insert = d[i][j-1] + 1
				delete = d[i-1][j] + 1
				d[i][j] = min(substitute, insert, delete)
	return float(d[len(r)][len(h)]) / len(r)

def _compute_batch_wer_mean(model, source_batch, target_batch, vocab_size, argmax=True):
	xp = model.xp
	num_calculation = 0
	sum_wer = 0
	skip_mask = source_batch != ID_PAD
	# to gpu
	if xp is cuda.cupy:
		source_batch = cuda.to_gpu(source_batch)
		target_batch = cuda.to_gpu(target_batch)
		skip_mask = cuda.to_gpu(skip_mask)

	target_seq_length = target_batch.shape[1]
	word_ids = xp.arange(0, vocab_size, dtype=xp.int32)

	for n in xrange(len(source_batch)):
		# reset
		model.reset_state()
		token = ID_GO
		x = xp.asarray([[token]]).astype(xp.int32)

		# get encoder's last hidden states
		encoder_hidden_states = model.encode(source_batch[None, n, :], skip_mask[None, n, :], test=True)

		# decode step by step
		while token != ID_EOS and x.shape[1] < target_seq_length:
			u = model.decode_one_step(x, encoder_hidden_states, test=True)[None, -1]	# take the output vector at the last time
			p = F.softmax(u).data[-1]	# convert to probability

			# argmax or sampling
			if argmax:
				token = [xp.argmax(p)]
			else:
				token = xp.random.choice(word_ids, size=1, p=p)

			# concatenate
			if xp is np:
				x = xp.append(x, xp.asarray([token]).astype(xp.int32), axis=1)
			else:
				a = cuda.to_gpu(np.asarray([token]).astype(np.int32))	# hack
				x = xp.concatenate((x, a), axis=1)

		target_tokens = []
		for token in target_batch[n, :]:
			token = int(token)	# to cpu
			if token == ID_PAD:
				break
			if token == ID_EOS:
				break
			if token == ID_GO:
				continue
			target_tokens.append(token)

		predict_tokens = []
		for token in x[0]:
			token = int(token)	# to cpu
			if token == ID_EOS:
				break
			if token == ID_GO:
				continue
			predict_tokens.append(token)

		wer = compute_word_error_rate_of_sequence(target_tokens, predict_tokens)
		sum_wer += wer
		num_calculation += 1
	return sum_wer / num_calculation

def compute_mean_wer(model, source_buckets, target_buckets, vocab_inv_source, vocab_inv_target, batchsize=100, argmax=True):
	xp = model.xp
	result = []
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
			mean_wer = _compute_batch_wer_mean(model, source_batch, target_batch, len(vocab_inv_target), argmax=argmax)
			sum_wer += mean_wer
			num_calculation += 1

		result.append(sum_wer / num_calculation * 100)

	return result

def compute_random_mean_wer(model, source_buckets, target_buckets, vocab_inv_source, vocab_inv_target, sample_size=100, argmax=True):
	xp = model.xp
	result = []
	for source_bucket, target_bucket in zip(source_buckets, target_buckets):
		# sample minibatch
		source_batch, target_batch = sample_batch_from_bucket(source_bucket, target_bucket, sample_size)
		
		# compute WER
		mean_wer = _compute_batch_wer_mean(model, source_batch, target_batch, len(vocab_inv_target), argmax=argmax)

		result.append(mean_wer * 100)

	return result