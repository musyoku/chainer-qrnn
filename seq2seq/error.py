# coding: utf-8
import numpy as np
import chainer.functions as F
from chainer import cuda
from model import Seq2SeqModel, AttentiveSeq2SeqModel
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

def _compute_batch_wer_mean(model, source_batch, target_batch, target_vocab_size, argmax=True):
	xp = model.xp
	num_calculation = 0
	sum_wer = 0
	skip_mask = source_batch != ID_PAD
	batchsize = source_batch.shape[0]
	target_seq_length = target_batch.shape[1]

	# to gpu
	if xp is cuda.cupy:
		source_batch = cuda.to_gpu(source_batch)
		target_batch = cuda.to_gpu(target_batch)
		skip_mask = cuda.to_gpu(skip_mask)

	word_ids = xp.arange(0, target_vocab_size, dtype=xp.int32)

	model.reset_state()
	token = ID_GO
	x = xp.asarray([[token]]).astype(xp.int32)
	x = xp.broadcast_to(x, (batchsize, 1))

	# get encoder's last hidden states
	if isinstance(model, AttentiveSeq2SeqModel):
		encoder_last_hidden_states, encoder_last_layer_outputs = model.encode(source_batch, skip_mask, test=True)
	else:
		encoder_last_hidden_states = model.encode(source_batch, skip_mask, test=True)

	while x.shape[1] < target_seq_length * 2:
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

	for n in xrange(batchsize):
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
		for token in x[n]:
			token = int(token)	# to cpu
			if token == ID_EOS:
				break
			if token == ID_PAD:
				break
			if token == ID_GO:
				continue
			predict_tokens.append(token)

		wer = compute_word_error_rate_of_sequence(target_tokens, predict_tokens)
		sum_wer += wer
		num_calculation += 1

	return sum_wer / num_calculation

def compute_mean_wer(model, source_buckets, target_buckets, target_vocab_size, batchsize=100, argmax=True):
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
			mean_wer = _compute_batch_wer_mean(model, source_batch, target_batch, target_vocab_size, argmax=argmax)
			sum_wer += mean_wer
			num_calculation += 1

		result.append(sum_wer / num_calculation * 100)

	return result

def compute_random_mean_wer(model, source_buckets, target_buckets, target_vocab_size, sample_size=100, argmax=True):
	xp = model.xp
	result = []
	for source_bucket, target_bucket in zip(source_buckets, target_buckets):
		# sample minibatch
		source_batch, target_batch = sample_batch_from_bucket(source_bucket, target_bucket, sample_size)
		
		# compute WER
		mean_wer = _compute_batch_wer_mean(model, source_batch, target_batch, target_vocab_size, argmax=argmax)

		result.append(mean_wer * 100)

	return result