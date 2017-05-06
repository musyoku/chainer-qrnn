# encoding: utf-8
from __future__ import division
from __future__ import print_function
import numpy as np
import chainer, sys
import chainer.links as L
import chainer.functions as F
from chainer import Variable, Chain, cuda
from model import RNNModel
from dataset import make_source_target_pair

def test_rnn():
	np.random.seed(0)
	num_layers = 50
	seq_length = num_layers * 2
	batchsize = 2
	vocab_size = 4
	data = np.random.randint(0, vocab_size, size=(batchsize, seq_length), dtype=np.int32)
	source, target = make_source_target_pair(data)
	model = RNNModel(vocab_size, ndim_embedding=100, num_layers=num_layers, ndim_h=3, kernel_size=3, pooling="fo", zoneout=False, wstd=1)

	np.random.seed(0)
	model.reset_state()
	Y = model(source, test=True).data

	model.reset_state()
	np.random.seed(0)
	for t in xrange(source.shape[1]):
		y = model.forward_one_step(source[:, :t+1], test=True).data
		target = np.swapaxes(np.reshape(Y, (batchsize, -1, vocab_size)), 1, 2)
		target = np.reshape(np.swapaxes(target[:, :, t, None], 1, 2), (batchsize, -1))
		assert np.sum((y - target) ** 2) == 0
		print("t = {} OK".format(t))

if __name__ == "__main__":
	test_rnn()
