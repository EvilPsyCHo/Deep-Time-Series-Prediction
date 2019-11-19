# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/18 15:58
"""
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type='LSTM'):
        super(Encoder, self).__init__()
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        outputs, hidden = self.rnn(x)
        return outputs, hidden
