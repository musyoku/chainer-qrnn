import sys, os
import chainer.functions as F
from chainer import Chain
sys.path.append(os.path.split(os.getcwd())[0])
import qrnn as L

class QRNN(Chain):
	def __init__(self, num_vocab, ndim_embedding, ndim_h, pooling="fo", zoneout=False, wstd=1):
		super(QRNN, self).__init__(
			embed=L.EmbedID(num_vocab, ndim_embedding, ignore_label=0),
			l1=L.QRNN(ndim_embedding, ndim_h, kernel_size=4, pooling=pooling, zoneout=zoneout, wstd=wstd),
			l2=L.QRNN(ndim_h, ndim_h, kernel_size=4, pooling=pooling, zoneout=zoneout, wstd=wstd),
			l3=L.Linear(ndim_h, num_vocab),
		)
		self.ndim_embedding = ndim_embedding

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()

	# we use "dense convolution"
	# https://arxiv.org/abs/1608.06993
	def __call__(self, X, test=False):
		batchsize = X.shape[0]
		seq_length = X.shape[1]
		H0 = self.embed(X)
		H0 = F.swapaxes(H0, 1, 2)
		self.l1(H0, test=test)
		H1 = self.l1.get_all_hidden_states()
		self.l2(H1, test=test)
		H2 = self.l2.get_all_hidden_states() + H1
		H2 = F.reshape(F.swapaxes(H2, 1, 2), (batchsize * seq_length, -1))
		Y = self.l3(H2)
		return Y