# encoding: utf-8
from __future__ import division
from __future__ import print_function
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable, Chain
from model import AttentiveSeq2SeqModel, Seq2SeqModel

def test_seq2seq():
	enc_seq_length = 5
	dec_seq_length = 7
	batchsize = 3
	enc_vocab_size = 6
	dec_vocab_size = 3
	enc_data = np.random.randint(0, enc_vocab_size, size=(batchsize, enc_seq_length), dtype=np.int32)
	dec_data = np.random.randint(0, dec_vocab_size, size=(batchsize, dec_seq_length), dtype=np.int32)
	skip_mask = np.ones_like(enc_data).astype(np.float32)
	skip_mask[:, :1] = 0
	skip_mask[0, :2] = 0
	print(skip_mask)

	model = Seq2SeqModel(10, 3, 30, 1, 3, pooling="fo", zoneout=False, wstd=1)

	ht = model.encode(enc_data, skip_mask)
	Y = model.decode(dec_data, ht)
	print(Y.data)

	model.reset_decoder_state()
	for t in xrange(dec_seq_length):
		y = model.decode_one_step(dec_data[:, :t+1], ht).data
		target = np.swapaxes(np.reshape(Y.data, (batchsize, -1, dec_vocab_size)), 1, 2)
		target = np.reshape(np.swapaxes(target[:, :, t, None], 1, 2), (batchsize, -1))
		assert np.sum((y - target) ** 2) == 0

def test_attentive_seq2seq():
	enc_seq_length = 5
	dec_seq_length = 7
	batchsize = 3
	enc_vocab_size = 6
	dec_vocab_size = 3
	enc_data = np.random.randint(0, enc_vocab_size, size=(batchsize, enc_seq_length), dtype=np.int32)
	dec_data = np.random.randint(0, dec_vocab_size, size=(batchsize, dec_seq_length), dtype=np.int32)
	skip_mask = np.ones_like(enc_data).astype(np.float32)
	skip_mask[:, :1] = 0
	skip_mask[0, :2] = 0
	print(skip_mask)

	model = AttentiveSeq2SeqModel(10, 3, 30, 1, 3, pooling="fo", zoneout=False, wstd=1)

	ht, H = model.encode(enc_data, skip_mask)
	Y = model.decode(dec_data, ht, H, skip_mask)
	print(Y.data)

	model.reset_decoder_state()
	for t in xrange(dec_seq_length):
		y = model.decode_one_step(dec_data[:, :t+1], ht, H, skip_mask).data
		target = np.swapaxes(np.reshape(Y.data, (batchsize, -1, dec_vocab_size)), 1, 2)
		target = np.reshape(np.swapaxes(target[:, :, t, None], 1, 2), (batchsize, -1))
		assert np.sum((y - target) ** 2) == 0

if __name__ == "__main__":
	test_seq2seq()
	test_attentive_seq2seq()
