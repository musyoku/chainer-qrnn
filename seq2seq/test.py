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
	num_layers = 13
	enc_seq_length = num_layers * 2
	dec_seq_length = num_layers * 3
	batchsize = 3
	enc_vocab_size = 6
	dec_vocab_size = 3
	enc_data = np.random.randint(0, enc_vocab_size, size=(batchsize, enc_seq_length), dtype=np.int32)
	dec_data = np.random.randint(0, dec_vocab_size, size=(batchsize, dec_seq_length), dtype=np.int32)
	skip_mask = np.ones_like(enc_data).astype(np.float32)
	skip_mask[:, :1] = 0
	skip_mask[0, :2] = 0
	skip_mask[1, :4] = 0
	skip_mask[2, :7] = 0

	model = Seq2SeqModel(enc_vocab_size, dec_vocab_size, ndim_embedding=30, num_layers=num_layers, ndim_h=3, pooling="fo", zoneout=False, wstd=1, densely_connected=True)

	ht = model.encode(enc_data, skip_mask)
	Y = model.decode(dec_data, ht)

	model.reset_decoder_state()
	for t in xrange(dec_seq_length):
		y = model.decode_one_step(dec_data[:, :t+1], ht).data
		target = np.swapaxes(np.reshape(Y.data, (batchsize, -1, dec_vocab_size)), 1, 2)
		target = np.reshape(np.swapaxes(target[:, :, t, None], 1, 2), (batchsize, -1))
		assert np.sum((y - target) ** 2) == 0
		print("t = {} OK".format(t))

def test_attentive_seq2seq():
	num_layers = 13
	enc_seq_length = num_layers * 2
	dec_seq_length = num_layers * 3
	batchsize = 3
	enc_vocab_size = 6
	dec_vocab_size = 3
	enc_data = np.random.randint(0, enc_vocab_size, size=(batchsize, enc_seq_length), dtype=np.int32)
	dec_data = np.random.randint(0, dec_vocab_size, size=(batchsize, dec_seq_length), dtype=np.int32)
	skip_mask = np.ones_like(enc_data).astype(np.float32)
	skip_mask[:, :1] = 0
	skip_mask[0, :2] = 0
	skip_mask[1, :4] = 0
	skip_mask[2, :7] = 0

	model = AttentiveSeq2SeqModel(enc_vocab_size, dec_vocab_size, ndim_embedding=30, num_layers=num_layers, ndim_h=3, pooling="fo", zoneout=False, wstd=1, densely_connected=True)

	ht, H = model.encode(enc_data, skip_mask)
	Y = model.decode(dec_data, ht, H, skip_mask)

	model.reset_decoder_state()
	for t in xrange(dec_seq_length):
		y = model.decode_one_step(dec_data[:, :t+1], ht, H, skip_mask).data
		target = np.swapaxes(np.reshape(Y.data, (batchsize, -1, dec_vocab_size)), 1, 2)
		target = np.reshape(np.swapaxes(target[:, :, t, None], 1, 2), (batchsize, -1))
		assert np.sum((y - target) ** 2) == 0
		print("t = {} OK".format(t))

if __name__ == "__main__":
	test_seq2seq()
	test_attentive_seq2seq()
