# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/12 16:14
"""
import random
import torch
import torch.nn as nn
from dtsp.modules import Encoder, Decoder


class SimpleSeq2Seq(nn.Module):

    def __init__(self, target_size, hidden_size, rnn_type='LSTM', dropout=0.2, activation='Tanh', teacher=0.):
        super(SimpleSeq2Seq, self).__init__()
        self.encoder = Encoder(target_size, hidden_size, rnn_type)
        self.decoder = Decoder(target_size, hidden_size, target_size, rnn_type, dropout, activation)
        self.teacher = teacher

    def forward(self, enc_seqs, dec_input_seqs, dec_output_seqs):
        _, hidden = self.encoder(enc_seqs)

        if self.teacher > 0:
            outputs = []
            dec_input = enc_seqs[:, -1, :].unsqueeze(1)
            n_step = dec_output_seqs.shape[1]
            for i in range(n_step):
                dec_output, hidden = self.decoder(dec_input, hidden)
                outputs.append(dec_output)
                if self.teacher > random.random():
                    dec_input = dec_input_seqs[:, i, :].unsqueeze(1)
                else:
                    dec_input = dec_output
            outputs = torch.cat(outputs, dim=1)
        else:
            outputs, _ = self.decoder(dec_input_seqs, hidden)
        return outputs


class Seq2Seq:
    pass


class ConditionSeq2Seq:
    pass
