# encoding: utf-8
from __future__ import division
from __future__ import print_function
import numpy as np
from qrnn import QRNN, QRNNEncoder, QRNNDecoder, QRNNGlobalAttentiveDecoder

def test_decoder():
	np.random.seed(0)
	enc_shape = (2, 3, 5)
	dec_shape = (2, 4, 7)
	prod = enc_shape[0] * enc_shape[1] * enc_shape[2]
	enc_data = np.arange(0, prod, dtype=np.float32).reshape(enc_shape) / prod
	prod = dec_shape[0] * dec_shape[1] * dec_shape[2]
	dec_data = np.arange(0, prod, dtype=np.float32).reshape(dec_shape) / prod
	skip_mask = np.ones((enc_data.shape[0], enc_data.shape[2]), dtype=np.float32)
	skip_mask[:, :1] = 0
	skip_mask[0, :2] = 0

	encoder = QRNNEncoder(enc_shape[1], 4, kernel_size=4, pooling="fo", zoneout=False, zoneout_ratio=0.5)
	decoder = QRNNDecoder(dec_shape[1], 4, kernel_size=4, pooling="fo", zoneout=False, zoneout_ratio=0.5)

	np.random.seed(0)
	H = encoder(enc_data, skip_mask)
	ht = encoder.get_last_hidden_state()
	Y = decoder(dec_data, ht)

	np.random.seed(0)
	decoder.reset_state()
	for t in xrange(dec_shape[2]):
		y = decoder.forward_one_step(dec_data[:, :, :t+1], ht)
		assert np.sum((y.data - Y.data[:, :, :t+1]) ** 2) == 0


def test_attentive_decoder():
	np.random.seed(0)
	enc_shape = (2, 3, 5)
	dec_shape = (2, 4, 7)
	prod = enc_shape[0] * enc_shape[1] * enc_shape[2]
	enc_data = np.arange(0, prod, dtype=np.float32).reshape(enc_shape) / prod
	prod = dec_shape[0] * dec_shape[1] * dec_shape[2]
	dec_data = np.arange(0, prod, dtype=np.float32).reshape(dec_shape) / prod
	skip_mask = np.ones((enc_data.shape[0], enc_data.shape[2]), dtype=np.float32)
	skip_mask[:, :1] = 0
	skip_mask[0, :2] = 0

	encoder = QRNNEncoder(enc_shape[1], 4, kernel_size=4, pooling="fo", zoneout=False, zoneout_ratio=0.5)
	decoder = QRNNGlobalAttentiveDecoder(dec_shape[1], 4, kernel_size=4, zoneout=False, zoneout_ratio=0.5)

	H = encoder(enc_data, skip_mask)
	ht = encoder.get_last_hidden_state()
	Y = decoder(dec_data, ht, H, skip_mask)
	print(Y.data)

	decoder.reset_state()
	for t in xrange(dec_shape[2]):
		y = decoder.forward_one_step(dec_data[:, :, :t+1], ht, H, skip_mask)
		print(y.data)
		assert np.sum((y.data - Y.data[:, :, :t+1]) ** 2) == 0



if __name__ == "__main__":
	test_decoder()
	test_attentive_decoder()