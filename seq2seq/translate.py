# coding: utf-8
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import argparse, sys, os, codecs, random, math
import numpy as np
import chainer
import chainer.functions as F
from chainer import training, Variable, optimizers, cuda
from chainer.training import extensions
sys.path.append(os.path.split(os.getcwd())[0])
from model import load_model, load_vocab, Seq2SeqModel, AttentiveSeq2SeqModel
from common import ID_UNK, ID_PAD, ID_GO, ID_EOS, bucket_sizes, stdout, print_bold
from dataset import sample_batch_from_bucket, read_data

def make_buckets(dataset):
	buckets_list = [[] for _ in xrange(len(bucket_sizes))]
	for word_ids in dataset:
		length = len(word_ids)
		bucket_index = 0
		for size in bucket_sizes:
			if length <= size[0]:
				break
			bucket_index += 1
		if bucket_index >= len(bucket_sizes):
			continue	# ignore long sequence

		required_length, _ = bucket_sizes[bucket_index]
		
		for _ in xrange(max(required_length - length, 0)):
			word_ids.insert(0, ID_PAD)	# prepend
		assert len(word_ids) == required_length

		buckets_list[bucket_index].append(word_ids)

	buckets = []
	for bucket_source in buckets_list:
		if len(bucket_source) == 0:
			continue
		buckets.append(np.asarray(bucket_source).astype(np.int32))
	return buckets

# https://harvardnlp.github.io/seq2seq-talk/slidescmu.pdf
# def _beam_search(model, x, t, beam_width, log_p_t_beam, sum_log_p_beam, vocab_size, backward_table, token_table, endpoint_list, perplexity_list, top_k):
# 	if top_k <= 0:
# 		return 0
# 	assert beam_width == len(sum_log_p_beam)
# 	xp = model.xp

# 	def argmax_k(array, k):
# 		if xp is np:
# 			return array.argsort()[-k:][::-1]
# 		else:
# 			result = []
# 			min_value = xp.amin(array)
# 			for n in xrange(k):
# 				result.append(xp.argmax(array))
# 				array[result[-1]] = min_value
# 			return result

# 	if t == 0:
# 		score = log_p_t_beam[0]
# 		indices_beam = argmax_k(score, beam_width)
# 	else:
# 		score = log_p_t_beam + xp.repeat(sum_log_p_beam, vocab_size, axis=1)
# 		score = score.reshape((-1,))
# 		indices_beam = argmax_k(score, beam_width)

# 	if t == 0:
# 		for beam, index in enumerate(indices_beam):
# 			token = index % vocab_size
# 			backward = index // vocab_size
# 			backward_table[beam, t] = backward
# 			token_table[beam, t] = token
# 			sum_log_p_beam[beam] += log_p_t_beam[backward, token]
# 			if token == ID_EOS:
# 				endpoint_list.append((t, beam))
# 				perplexity_list.append(float(sum_log_p_beam[beam]) / t)
# 				top_k -= 1
# 		return top_k

# 	k = 0
# 	for beam in xrange(beam_width):
# 		backward_table[beam, t] = -1

# 	for beam, index in enumerate(indices_beam):
# 		token = index % vocab_size
# 		backward = index // vocab_size
# 		prev_token = x[backward, t]
# 		if prev_token == ID_EOS or prev_token == ID_PAD:
# 			continue
# 		sum_log_p_beam[beam] += log_p_t_beam[backward, token]
# 		print(beam, token, backward)
# 		backward_table[beam, t] = backward
# 		token_table[beam, t] = token
# 		if token == ID_EOS:
# 			endpoint_list.append((t, beam))
# 			perplexity_list.append(float(sum_log_p_beam[beam]) / t)
# 			top_k -= 1

# 	for beam in xrange(beam_width):
# 		backward = backward_table[beam, t]
# 		if backward == -1:
# 			print(beam)
# 	return top_k


def translate_greedy(model, source_batch, max_predict_length, vocab_size, source_reversed=True):
	xp = model.xp
	skip_mask = source_batch != ID_PAD
	batchsize = source_batch.shape[0]

	# to gpu
	if xp is cuda.cupy:
		source_batch = cuda.to_gpu(source_batch)
		skip_mask = cuda.to_gpu(skip_mask)

	word_ids = xp.arange(0, vocab_size, dtype=xp.int32)

	model.reset_state()
	token = ID_GO
	x = xp.asarray([[token]]).astype(xp.int32)
	x = xp.broadcast_to(x, (batchsize, 1))

	# get encoder's last hidden states
	if isinstance(model, AttentiveSeq2SeqModel):
		encoder_last_hidden_states, encoder_last_layer_outputs = model.encode(source_batch, skip_mask, test=True)
	else:
		encoder_last_hidden_states = model.encode(source_batch, skip_mask, test=True)

	while x.shape[1] < max_predict_length:
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
			token = xp.argmax(pn)

			x[n, -1] = token

	return x

# http://opennmt.net/OpenNMT/translation/beam_search/
def translate_beam_search(model, source, max_predict_length, vocab_size, beam_width=8, normalization_alpha=0, source_reversed=True, return_all_candidates=False):
	xp = model.xp
	if source.ndim == 1:
		source = xp.reshape(source, (1, -1))
	skip_mask = source != ID_PAD
	batchsize = source.shape[0]

	# to gpu
	if xp is cuda.cupy:
		source = cuda.to_gpu(source)
		skip_mask = cuda.to_gpu(skip_mask)

	word_ids = xp.arange(0, vocab_size, dtype=xp.int32)

	model.reset_state()
	x = xp.full((beam_width, 1), ID_GO, dtype=xp.int32)

	# get encoder's last hidden states
	if isinstance(model, AttentiveSeq2SeqModel):
		encoder_last_hidden_states, encoder_last_layer_outputs = model.encode(source, skip_mask, test=True)
	else:
		encoder_last_hidden_states, encoder_last_layer_outputs = model.encode(source, skip_mask, test=True), None

	# copy beam_width times
	for i, state in enumerate(encoder_last_hidden_states):
		encoder_last_hidden_states[i] = xp.repeat(state.data, beam_width, axis=0)
	if encoder_last_layer_outputs is not None:
		encoder_last_layer_outputs = xp.repeat(encoder_last_layer_outputs.data, beam_width, axis=0)

	sum_log_p = xp.zeros((beam_width, 1), dtype=xp.float32)
	skip_mask = xp.repeat(skip_mask, beam_width, axis=0)

	def argmax_k(array, k):
		if xp is np:
			return array.argsort()[-k:][::-1]
		else:
			result = []
			min_value = xp.amin(array)
			for n in xrange(k):
				result.append(xp.argmax(array))
				array[result[-1]] = min_value
			return result

	current_beam_width = beam_width
	candidates = []
	log_likelihood = []

	for t in xrange(max_predict_length):
		model.reset_decoder_state()
		if isinstance(model, AttentiveSeq2SeqModel):
			u_t = model.decode(x, encoder_last_hidden_states, encoder_last_layer_outputs, skip_mask, test=True, return_last=True)
		else:
			u_t = model.decode(x, encoder_last_hidden_states, test=True, return_last=True)
		p_t = F.softmax(u_t)	# convert to probability
		log_p_t = F.log(p_t).data

		# compute scores
		if t == 0:
			score = log_p_t[0]	# <go>
			top_indices = argmax_k(score, current_beam_width)
		else:
			score = log_p_t + xp.repeat(sum_log_p, vocab_size, axis=1)
			score = score.reshape((-1,))
			top_indices = argmax_k(score, current_beam_width)

		backward_table = [-1] * current_beam_width
		token_table = [-1] * current_beam_width
		stopped_beams = []
		for beam, index in enumerate(top_indices):
			index = int(index)
			token = index % vocab_size
			backward = index // vocab_size
			backward_table[beam] = backward
			token_table[beam] = token
			sum_log_p[beam] += log_p_t[backward, token]
			if token == ID_EOS:
				stopped_beams.append(beam)
				log_likelihood.append(float(sum_log_p[beam]))

		# concatenate
		if xp is np:
			x = xp.append(x, xp.full((current_beam_width, 1), ID_PAD, dtype=xp.int32), axis=1)
		else:
			x = xp.concatenate((x, xp.full((current_beam_width, 1), ID_PAD, dtype=xp.int32)), axis=1)
		new_x = xp.copy(x)

		# reconstruct input sequense
		new_sum_log_p = xp.empty_like(sum_log_p)
		for beam in xrange(current_beam_width):
			new_x[beam, -1] = token_table[beam]
			backward = backward_table[beam]
			new_x[beam, :-1] = x[backward, :-1]
			new_sum_log_p[beam] = sum_log_p[backward]
		x = new_x
		sum_log_p = new_sum_log_p

		# remove stopped beam
		if len(stopped_beams) > 0:
			flag = xp.ones((current_beam_width,), dtype=bool)
			flag[stopped_beams] = False
			new_x = x[flag]
			sum_log_p = sum_log_p[flag]
			stopped_x = x[xp.invert(flag)]
			for n in xrange(len(stopped_x)):
				candidates.append(stopped_x[n])
			x = new_x

			# slice
			num_to_remove = len(stopped_beams)
			for i, state in enumerate(encoder_last_hidden_states):
				encoder_last_hidden_states[i] = encoder_last_hidden_states[i][num_to_remove:]
			if encoder_last_layer_outputs is not None:
				encoder_last_layer_outputs = encoder_last_layer_outputs[num_to_remove:]
			skip_mask = skip_mask[num_to_remove:]
			current_beam_width -= len(stopped_beams)

		if current_beam_width <= 0:
			break

	assert len(candidates) == len(log_likelihood)
	num_sampled = len(candidates)

	# if empty
	if num_sampled == 0:
		result = []
		for token in (x[0]):
			result.append(token)
		return result

	# compute score
	scores = np.empty((num_sampled,), dtype=float)
	for i, (sequence, log_p) in enumerate(zip(candidates, log_likelihood)):
		length = sequence.size
		penalty = math.pow(5 + length, normalization_alpha) / math.pow(5 + 1, normalization_alpha)
		score = log_p / penalty
		scores[i] = score

	if return_all_candidates == True:
		indices = np.flip(np.argsort(scores))
		print(scores)
		print(indices)

	best_index = np.argmax(scores)
	result = []
	for token in (candidates[best_index]):
		result.append(token)
	return result

# def translate_greedy(model, source_batch, max_predict_length, vocab_size, beam_width=8, source_reversed=True):
# 	xp = model.xp
# 	skip_mask = source_batch != ID_PAD
# 	batchsize = source_batch.shape[0]

# 	# to gpu
# 	if xp is cuda.cupy:
# 		source_batch = cuda.to_gpu(source_batch)
# 		skip_mask = cuda.to_gpu(skip_mask)

# 	word_ids = xp.arange(0, vocab_size, dtype=xp.int32)

# 	model.reset_state()
# 	x = xp.full((batchsize * beam_width, 1), ID_GO, dtype=xp.int32)

# 	# get encoder's last hidden states
# 	if isinstance(model, AttentiveSeq2SeqModel):
# 		encoder_last_hidden_states, encoder_last_layer_outputs = model.encode(source_batch, skip_mask, test=True)
# 	else:
# 		encoder_last_hidden_states, encoder_last_layer_outputs = model.encode(source_batch, skip_mask, test=True), None

# 	# copy beam_width times
# 	for i, state in enumerate(encoder_last_hidden_states):
# 		encoder_last_hidden_states[i] = xp.repeat(state.data, beam_width, axis=0)
# 	if encoder_last_layer_outputs is not None:
# 		encoder_last_layer_outputs = xp.repeat(encoder_last_layer_outputs.data, beam_width, axis=0)
# 	skip_mask = xp.repeat(skip_mask, beam_width, axis=0)

# 	backward_table_batch = xp.zeros((beam_width * batchsize, max_predict_length), dtype=xp.int32)
# 	token_table_batch = xp.zeros((beam_width * batchsize, max_predict_length), dtype=xp.int32)
# 	sum_log_p_batch = xp.zeros((batchsize * beam_width, 1), dtype=xp.float32)
# 	perplexity_list_batch = [[] * batchsize]
# 	endpoint_list_batch = [[] * batchsize]
# 	top_k = [beam_width] * batchsize

# 	for t in xrange(max_predict_length):
# 		model.reset_decoder_state()
# 		if isinstance(model, AttentiveSeq2SeqModel):
# 			u_t = model.decode(x, encoder_last_hidden_states, encoder_last_layer_outputs, skip_mask, test=True, return_last=True)
# 		else:
# 			u_t = model.decode(x, encoder_last_hidden_states, test=True, return_last=True)
# 		p_t = F.softmax(u_t)	# convert to probability
# 		log_p_t = F.log(p_t)

# 		# concatenate
# 		if xp is np:
# 			x = xp.append(x, xp.full((batchsize * beam_width, 1), ID_PAD, dtype=xp.int32), axis=1)
# 		else:
# 			x = xp.concatenate((x, xp.full((batchsize * beam_width, 1), ID_PAD, dtype=xp.int32)), axis=1)

# 		# beam search
# 		for b in xrange(batchsize):
# 			start = b * beam_width
# 			end = (b + 1) * beam_width
# 			log_p = log_p_t.data[start:end]
# 			sum_log_p = sum_log_p_batch[start:end]
# 			backward_table = backward_table_batch[start:end]
# 			token_table = token_table_batch[start:end]
# 			top_k[b] = _beam_search(model, x[start:end], t, beam_width, log_p, sum_log_p, vocab_size, backward_table, token_table, endpoint_list_batch[b], perplexity_list_batch[b], top_k[b])
# 			# reconstruct input sequence
# 			for beam in xrange(beam_width):
# 				k = beam
# 				for i in xrange(t + 1):
# 					token = int(token_table[k, t-max_predict_length-i])
# 					k = backward_table[k, t-max_predict_length-i]
# 					x[start + beam, -i-1] = token

# 	# backward
# 	result = np.zeros((batchsize, max_predict_length), dtype=np.int32)
# 	for b in xrange(batchsize):
# 		start = b * beam_width
# 		end = (b + 1) * beam_width
# 		log_p = log_p_batch.data[start:end]
# 		perplexity = perplexity_batch[start:end]
# 		sum_log_p = sum_log_p_batch[start:end]
# 		print(sum_log_p)
# 		print(perplexity)
# 		backward_table = backward_table_batch[start:end]
# 		token_table = token_table_batch[start:end]
# 		k = xp.argmax(sum_log_p)	# take maximum
# 		tokens = []
# 		for t in xrange(1, max_predict_length + 1):
# 			token = int(token_table[k, -t])
# 			tokens.append(token)
# 			k = backward_table[k, -t]
# 		tokens.reverse()
# 		result[b] = tokens
# 		print(tokens)

# 	return result

def dump_translation(vocab_inv_source, vocab_inv_target, source, translation, target=None, source_reversed=True):
	sentence = []
	for token in source:
		token = int(token)	# to cpu
		if token == ID_PAD:
			continue
		word = vocab_inv_source[token]
		sentence.append(word)
	if source_reversed:
		sentence.reverse()
	print(">source: ", " ".join(sentence))

	if target is not None:
		sentence = []
		for token in target:
			token = int(token)	# to cpu
			if token == ID_PAD:
				break
			if token == ID_EOS:
				break
			if token == ID_GO:
				continue
			word = vocab_inv_target[token]
			sentence.append(word)
		print(" target: ", " ".join(sentence))

	sentence = []
	for token in translation:
		token = int(token)	# to cpu
		if token == ID_EOS:
			break
		if token == ID_PAD:
			break
		if token == ID_GO:
			continue
		word = vocab_inv_target[token]
		sentence.append(word)
	print(" predict:", " ".join(sentence))

# def dump_source_target_translation(model, source_buckets, target_buckets, vocab_inv_source, vocab_inv_target, beam_width=8, batchsize=10):
# 	for source_bucket, target_bucket in zip(source_buckets, target_buckets):
# 		num_calculation = 0
# 		sum_wer = 0

# 		if len(source_bucket) > batchsize:
# 			num_sections = len(source_bucket) // batchsize - 1
# 			if len(source_bucket) % batchsize > 0:
# 				num_sections += 1
# 			indices = [(i + 1) * batchsize for i in xrange(num_sections)]
# 			source_sections = np.split(source_bucket, indices, axis=0)
# 			target_sections = np.split(target_bucket, indices, axis=0)
# 		else:
# 			source_sections = [source_bucket]
# 			target_sections = [target_bucket]

# 		for source_batch, target_batch in zip(source_sections, target_sections):
# 			translation_batch = translate_beam_search(model, source_batch, target_batch.shape[1] * 2, len(vocab_inv_target), beam_width)
# 			dump_translation(vocab_inv_source, vocab_inv_target, source_batch, translation_batch, target_batch)

def dump_source_translation(model, source_buckets, vocab_inv_source, vocab_inv_target, beam_width=8, normalization_alpha=0):
	for source_bucket in source_buckets:
		if beam_width == 1:	# greedy
			batchsize = 24
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
				translation_batch = translate_greedy(model, source_batch, target_batch.shape[1] * 2, len(vocab_inv_target), beam_width)
				for index in xrange(len(translation_batch)):
					source = source_batch[index]
					translation = translation_batch[index]
					target = target_batch[index]
					dump_translation(vocab_inv_source, vocab_inv_target, source, translation, target)
		else:	# beam search
			for index in xrange(len(source_bucket)):
				source = source_bucket[index]
				translation = translate_beam_search(model, source, source.size * 2, len(vocab_inv_target), beam_width, normalization_alpha, return_all_candidates=True)
				dump_translation(vocab_inv_source, vocab_inv_target, source, translation)

def dump_random_source_target_translation(model, source_buckets, target_buckets, vocab_inv_source, vocab_inv_target, num_translate=3, beam_width=8):
	xp = model.xp
	for source_bucket, target_bucket in zip(source_buckets, target_buckets):
		source_batch, target_batch = sample_batch_from_bucket(source_bucket, target_bucket, num_translate)
		
		if beam_width == 1:	# greedy
			translation_batch = translate_greedy(model, source_batch, target_batch.shape[1] * 2, len(vocab_inv_target), beam_width)
			for index in xrange(len(translation_batch)):
				source = source_batch[index]
				translation = translation_batch[index]
				target = target_batch[index]
				dump_translation(vocab_inv_source, vocab_inv_target, source, translation, target)

		else:	# beam search
			for index in xrange(len(source_batch)):
				source = source_batch[index]
				target = target_batch[index]
				translation_batch = translate_beam_search(model, source, target.size * 2, len(vocab_inv_target), beam_width)
				dump_translation(vocab_inv_source, vocab_inv_target, source, translation_batch, target)

def main(args):
	source_dataset, target_dataset, _, _ = read_data(args.source_train, None, args.source_dev, None, args.source_test, None, reverse_source=True)
	vocab, vocab_inv = load_vocab(args.model_dir)

	source_dataset_train, source_dataset_dev, source_dataset_test = source_dataset
	target_dataset_train, target_dataset_dev, target_dataset_test = target_dataset
	print_bold("data	#")
	if len(source_dataset_train) > 0:
		print("train	{}".format(len(source_dataset_train)))
	if len(source_dataset_dev) > 0:
		print("dev	{}".format(len(source_dataset_dev)))
	if len(source_dataset_test) > 0:
		print("test	{}".format(len(source_dataset_test)))

	vocab_source, vocab_target = vocab
	vocab_inv_source, vocab_inv_target = vocab_inv

	# split into buckets
	source_buckets_train = None
	if len(source_dataset_train) > 0:
		print_bold("buckets 	#data	(train)")
		source_buckets_train = make_buckets(source_dataset_train)
		if args.buckets_slice is not None:
			source_buckets_train = source_buckets_train[:args.buckets_slice + 1]
		for size, data in zip(bucket_sizes, source_buckets_train):
			print("{} 	{}".format(size, len(data)))

	source_buckets_dev = None
	if len(source_dataset_dev) > 0:
		print_bold("buckets 	#data	(dev)")
		source_buckets_dev = make_buckets(source_dataset_dev)
		if args.buckets_slice is not None:
			source_buckets_dev = source_buckets_dev[:args.buckets_slice + 1]
		for size, data in zip(bucket_sizes, source_buckets_dev):
			print("{} 	{}".format(size, len(data)))

	source_buckets_test = None
	if len(source_dataset_test) > 0:
		print_bold("buckets		#data	(test)")
		source_buckets_test = make_buckets(source_dataset_test)
		if args.buckets_slice is not None:
			source_buckets_test = source_buckets_test[:args.buckets_slice + 1]
		for size, data in zip(bucket_sizes, source_buckets_test):
			print("{} 	{}".format(size, len(data)))

	# init
	model = load_model(args.model_dir)
	assert model is not None
	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		model.to_gpu()

	if source_buckets_train is not None:
		dump_source_translation(model, source_buckets_train, vocab_inv_source, vocab_inv_target, beam_width=args.beam_width, normalization_alpha=0)

	if source_buckets_dev is not None:
		dump_source_translation(model, source_buckets_dev, vocab_inv_source, vocab_inv_target, beam_width=args.beam_width, normalization_alpha=0)

	if source_buckets_test is not None:
		dump_source_translation(model, source_buckets_test, vocab_inv_source, vocab_inv_target, beam_width=args.beam_width, normalization_alpha=0)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--source-train", type=str, default=None)
	parser.add_argument("--source-dev", type=str, default=None)
	parser.add_argument("--source-test", type=str, default=None)
	parser.add_argument("--gpu-device", "-g", type=int, default=0) 
	parser.add_argument("--buckets-slice", type=int, default=None)
	parser.add_argument("--beam-width", "-beam", type=int, default=8)
	parser.add_argument("--model-dir", "-m", type=str, default="model")
	args = parser.parse_args()
	main(args)