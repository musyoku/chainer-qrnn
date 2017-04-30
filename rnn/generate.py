# coding: utf-8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import argparse, sys, os
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
from model import load_model
from train import ID_BOS

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-device", "-g", type=int, default=0) 
parser.add_argument("--num-generate", "-n", type=int, default=10)
parser.add_argument("--model-dir", "-m", type=str, default="model")
args = parser.parse_args()

def main():
	model = load_model(args.model_dir)
	assert model is not None
	x = np.full((1, 1), ID_BOS, dtype=np.int32)
	y = model(x)


if __name__ == "__main__":
	main()