# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/19 14:04
"""
import torch.nn as nn


class SimpleRNNEncoder(nn.Module):

    def __init__(self, target_size, hidden_size, rnn_type, **kwargs):
        super(SimpleRNNEncoder, self).__init__()
        self.rnn = getattr(nn, rnn_type)(target_size, hidden_size, batch_first=True)

    def forward(self, x):
        outputs, hidden = self.rnn(x)
        return outputs, hidden


class RNNEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type, **kwargs):
        super(RNNEncoder, self).__init__()
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        outputs, hidden = self.rnn(x)
        return outputs, hidden
