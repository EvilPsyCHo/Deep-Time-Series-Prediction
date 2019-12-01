# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/18 15:19
"""
import torch
import torch.nn as nn
from . attention import MultiHeadAttention


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


class RNNDecoder(nn.Module):

    def __init__(self, input_size, target_size, hidden_size, rnn_type, dropout=0.2,
                 n_head=1, activation='Tanh', attn_type='general', use_attn=False, **kwargs):
        super(RNNDecoder, self).__init__()
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, batch_first=True)

        self.attn = MultiHeadAttention(n_head, hidden_size, attn_type) if use_attn else None
        self.use_attn = use_attn

        first_layer = (nn.Linear(hidden_size * (1 + n_head), hidden_size * 2) if use_attn
                       else nn.Linear(hidden_size, hidden_size * 2))
        self.dense = nn.Sequential(
            first_layer,
            getattr(nn, activation)(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, target_size))

    def forward(self, inputs, encoder_outputs, hidden=None):
        x, hidden = self.rnn(inputs, hidden)
        if self.use_attn:
            attn_context, weight = self.attn(x, encoder_outputs)
            concat = torch.cat([x, attn_context], dim=2)
            outputs = self.dense(concat)
            return outputs, hidden, weight
        else:
            outputs = self.dense(x)
            return outputs, hidden, None
