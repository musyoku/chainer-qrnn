# coding: utf-8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import argparse, sys, os
import numpy as np
import chainer.functions as F
sys.path.append(os.path.split(os.getcwd())[0])
from model import load_model, load_vocab
from train import ID_BOS, ID_EOS

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-device", "-g", type=int, default=0) 
parser.add_argument("--num-generate", "-n", type=int, default=10)
parser.add_argument("--max-sentence-length", "-max", type=int, default=50)
parser.add_argument("--model-dir", "-m", type=str, default="model")
args = parser.parse_args()

def main():
	model = load_model(args.model_dir)
	assert model is not None

	vocab, vocab_inv = load_vocab(args.model_dir)
	assert vocab is not None
	assert vocab_inv is not None

	vocab_size = model.vocab_size
	word_ids = np.arange(0, vocab_size, dtype=np.int32)
	token = ID_BOS
	x = np.full((1, 1), token, dtype=np.int32)
	while token != ID_EOS and x.shape[1] < args.max_sentence_length:
		p = F.softmax(model(x)).data[0]
		token = np.random.choice(word_ids, size=1, p=p)
		x = np.append(x, np.asarray([token]).astype(np.int32), axis=1)

	sentence = []
	for token in x[0]:
		word = vocab_inv[token]
		sentence.append(word)
	print(" ".join(sentence))

if __name__ == "__main__":
	main()