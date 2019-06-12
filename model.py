import os
import random
import numpy as np
import math
import torch
from torch import nn
from torch.nn import init
from torch import optim

# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html

class model(nn.Module):

    def __init__(self, vocab_size, hidden_dim=64, useAttn=True, attnType="mul"):
        super(model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeds = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim)
        self.useAttn = useAttn
        self.attnType = attnType
        if (self.useAttn):
            self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim)
        self.loss = nn.CrossEntropyLoss()
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.wa = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim).fill_(0.00001)) #wa for mult
        self.w1 = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim).fill_(0.001)) #w1 for add
        self.w2 = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim).fill_(0.001)) #w2 for add
        self.va = nn.Parameter(torch.Tensor(hidden_dim, 1).fill_(0.001)) # va for add
        self.softmax = nn.Softmax()
        #self.wa = nn.Parameter(torch.tensor.new_full((hidden_dim, hidden_dim), 0.00001))

    def compute_Loss(self, pred_vec, gold_seq):
        return self.loss(pred_vec, gold_seq)

    """ Depending on the the attention type,
        either calls addhelper or multhelper to calculate aij
    """
    def getaij(self, i, j, type, hiddeni, encoder_outputs):
        if type == "add":
            result = self.addhelper(hiddeni[0], encoder_outputs[j])
        else:
            result = self.multhelper(hiddeni[0], encoder_outputs[j])
        return result

    """ Attention function that returns the context vector
    """
    def attn(self, hiddeni, i, encoder_outputs, type):
        ai = torch.Tensor(1, 10).fill_(0)
        for j in range(10):
            ai[0][j] = self.getaij(i, j, type, hiddeni, encoder_outputs)
        ai = self.softmax(ai)
        ci = torch.mm(ai.squeeze(1), encoder_outputs.squeeze(1))
        return ci

    """ helper function for additive attention
        Va^t * tanh ( W1 * hi + W2 * sj)
    """
    def addhelper(self, hi, sj):
        hi = torch.t(hi.squeeze(1))
        sj = torch.t(sj.squeeze(1))
        add_result = torch.mm(self.w1, hi).add(torch.mm(self.w2, sj))
        tan_result = torch.tanh(add_result)
        return torch.mm(torch.t(self.va), tan_result)


    """ helper function for multiplicative attention
        hi^t * Wa * sj
    """
    def multhelper(self, hi, sj):
        hi = hi.squeeze(1)
        sj = torch.t(sj.squeeze(1))
        return torch.mm(torch.mm(hi, self.wa), sj)



    def forward(self, input_seq, gold_seq=None):
        # input seq is one sentence
        input_vectors = self.embeds(torch.tensor(input_seq))
        input_vectors = input_vectors.unsqueeze(1)
        outputs, hidden = self.encoder(input_vectors)
        encoder_outputs = outputs

        # Technique used to train RNNs:
        # https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
        teacher_force = True

        # only implemented attention for non Teacher force
        # (there was a piazza post that said teacher force with attn was
        # not required.)
        if (self.useAttn):
            teacher_force = False

        # This condition tells us whether we are in training or inference phase
        if gold_seq is not None and teacher_force:
            gold_vectors = torch.cat([torch.zeros(1, self.hidden_dim), self.embeds(torch.tensor(gold_seq[:-1]))], 0)
            gold_vectors = gold_vectors.unsqueeze(1)
            gold_vectors = torch.nn.functional.relu(gold_vectors)

            outputs, hidden = self.decoder(gold_vectors, hidden)

            predictions = self.out(outputs)
            predictions = predictions.squeeze()
            vals, idxs = torch.max(predictions, 1)
            return predictions, list(np.array(idxs))
        else:
            prev = torch.zeros(1, 1, self.hidden_dim)
            predictions = []
            predicted_seq = []
            for i in range(len(input_seq)):
                prev = torch.nn.functional.relu(prev)
                if (self.useAttn):
                    context = self.attn(hidden, i, encoder_outputs, self.attnType)
                    rnn_input = torch.cat((prev.squeeze(1), context), 1).unsqueeze(1)
                    outputs, hidden = self.decoder(rnn_input, hidden)
                else:
                    outputs, hidden = self.decoder(prev, hidden)
                pred_i = self.out(outputs)
                pred_i = pred_i.squeeze()
                _, idx = torch.max(pred_i, 0)
                idx = idx.item()
                predictions.append(pred_i)
                predicted_seq.append(idx)
                prev = self.embeds(torch.tensor([idx]))
                prev = prev.unsqueeze(1)
            return torch.stack(predictions), predicted_seq
