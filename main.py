import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
from pathlib import Path
from model import model
import time
from tqdm import tqdm
import sys


def read_input(task):
	assert task in {"copy", "reverse", "sort"}
	path = Path("data")
	train_f = path / ("train.txt")
	test_f = path / ("repeat.txt")

	with open(train_f, encoding='utf-8', errors='ignore') as train_file:
		train_inputs = [line.split() for line in train_file]
	train_file.close()

	with open(test_f, encoding='utf-8', errors='ignore') as test_file:
		test_inputs = [line.split() for line in test_file]
	test_file.close()

	if task == "copy":
		train_outputs, test_outputs = train_inputs, test_inputs
	elif task == "reverse":
		train_outputs = [seq[::-1] for seq in train_inputs]
		test_outputs = [seq[::-1] for seq in test_inputs]
	else:
		train_outputs = [sorted(seq) for seq in train_inputs]
		test_outputs = [sorted(seq) for seq in test_inputs]
	return [train_inputs, train_outputs, test_inputs, test_outputs]


def build_indices(train_set):
	tokens = [token for line in train_set for token in line]

	# From token to its index
	forward_dict = {'UNK': 0}

	# From index to token
	backward_dict = {0: 'UNK'}
	i = 1
	for token in tokens:
		if token not in forward_dict:
			forward_dict[token] = i
			backward_dict[i] = token
			i += 1
	return forward_dict, backward_dict


def encode(data, forward_dict):
	return [list(map(lambda t: forward_dict.get(t,0), line)) for line in data]


if __name__ == '__main__':
	datasets = read_input("copy") # Change this to change task
	forward_dict, backward_dict = build_indices(datasets[0])
	train_inputs, train_outputs, test_inputs, test_outputs = list(map(lambda x: encode(x, forward_dict), datasets))
	att = False
	type = ""
	if len(sys.argv) > 1:
		att = True
		type = sys.argv[1]
	m = model(vocab_size = len(forward_dict), hidden_dim = 128, useAttn = att, attnType = type)
	optimizer = optim.Adam(m.parameters())
	print(m.parameters)
	minibatch_size = 100
	num_minibatches = len(train_inputs) // minibatch_size

	# A minibatch is a group of examples for which make predictions for and then aggregate before
	# making a gradient update. This helps to make gradient updates more stable
	# as opposed to updating the model for every example (minibatch_size = 1). It also makes better use of the data
	# than performing a single gradient update after looking at the entire dataset (minibatch_size = dataset_size)

	for epoch in (range(2)):
		# Training
		print("Training")
		# Put the model in training mode
		m.train()
		start_train = time.time()

		for group in tqdm(range(num_minibatches)):
			total_loss = None
			#print(m.wa)
			optimizer.zero_grad()
			for i in range(group * minibatch_size, (group + 1) * minibatch_size):
				input_seq = train_inputs[i]
				gold_seq = torch.tensor(train_outputs[i])
				predictions, predicted_seq = m(input_seq, gold_seq)
				loss = m.compute_Loss(predictions, gold_seq)
				# On the first gradient update
				if total_loss is None:
					total_loss = loss
				else:
					total_loss += loss
			total_loss.backward()
			optimizer.step()
		print("Training time: {} for epoch {}".format(time.time() - start_train, epoch))

	# Evaluation
	print("Evaluation")
	# Put the model in evaluation mode
	m.eval()
	start_eval = time.time()

	file  = open("output/repeat_copy_" + type + ".txt", "w")
	predictions = 0
	correct = 0 # number of tokens predicted correctly
	for input_seq, gold_seq in zip(test_inputs, test_outputs):
		_, predicted_seq = m(input_seq)
		# Hint: why is this true? why is this assumption needed (for now)?
		assert len(predicted_seq) == len(gold_seq)
		correct += sum((predicted_seq[i] == gold_seq[i] for i in range(len(gold_seq))))
		predictions += len(gold_seq)
		# Hint: You might find the following useful for debugging.
		predicted_words = [backward_dict[index] for index in predicted_seq]
		predicted_sentence = " ".join(predicted_words)
		# for debugging: write the output in a file
		file.write(predicted_sentence)
		file.write("\n")
		gold_words = [backward_dict[index] for index in gold_seq]
		gold_sentence = " ".join(gold_words)
	accuracy = correct / predictions
	assert 0 <= accuracy <= 1
	print("Evaluation time: {} for epoch {}, Accuracy: {}".format(time.time() - start_eval, epoch, accuracy))
