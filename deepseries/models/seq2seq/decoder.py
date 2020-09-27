# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/9/27 14:51
"""
import torch
import torch.nn as nn
from deepseries.nn import Attention


class RNNDecoder:

    pass


class AttnRNNDecoder:

    def __init__(self, input_size, output_size, rnn_type, hidden_size,
                 num_layers, dropout, attn_head, attn_size, activation="ReLU", residual=False):
        super().__init__()
        self.input_size = input_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.residual = residual

        self.input_dropout = nn.Dropout(dropout)
        self.rnn = getattr(nn, rnn_type)(input_size=input_size, batch_first=True,
                                         num_layers=num_layers, hidden_size=hidden_size, dropout=dropout)
        self.attention = Attention(attn_head, attn_size, hidden_size, hidden_size, hidden_size, dropout)
        self.activation = getattr(nn, activation)()
        regression_input_size = attn_size + hidden_size
        if residual:
            regression_input_size += input_size
        self.regression = nn.Linear(regression_input_size, output_size)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor, encoder_output):
        # step input -> (batch, seq, N); previous dec hidden (layer, batch, hidden_size)
        dec_rnn_output, dec_rnn_hidden = self.rnn(input, hidden)
        # attention
        attn_applied, attn_weights = self.attention(dec_rnn_output, encoder_output, encoder_output)
        # predict
        if self.residual:
            concat = self.activation(torch.cat([attn_applied, dec_rnn_hidden, input], dim=2))
        else:
            concat = self.activation(torch.cat([attn_applied, dec_rnn_hidden], dim=2))
        output = self.regression(concat)
        return output, hidden, attn_weights
