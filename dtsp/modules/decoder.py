# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/18 15:19
"""
import torch.nn as nn
from . attention import GeneralAttention


class SimpleRNNDecoder(nn.Module):
    def __init__(self, target_size, hidden_size, rnn_type,
                 dropout=0.2, activation='Tanh', **kwargs):
        super(SimpleRNNDecoder, self).__init__()
        self.rnn = getattr(nn, rnn_type)(target_size, hidden_size, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            getattr(nn, activation)(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, target_size))

    def forward(self, x, hidden=None):
        x, hidden = self.rnn(x, hidden)
        outputs = self.dense(x)
        return outputs, hidden


class AttentionDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, rnn_type,
                 dropout=0.2, activation='Tanh', attn_type='general'):
        super(AttentionDecoder, self).__init__()
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, batch_first=True)
        self.attn = GeneralAttention(hidden_size)
        self.dense = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            getattr(nn, activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, output_size))

    def forward(self, inputs, encoder_outputs, hidden=False):
        x, hidden = self.rnn(inputs, hidden)
        weight, x = self.attn(x, encoder_outputs)
        outputs = self.dense(x)
        return outputs, weight
