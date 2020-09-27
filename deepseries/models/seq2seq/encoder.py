# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/9/27 14:51
"""
import torch
import torch.nn as nn


class RNNEncoder(nn.Module):

    def __init__(self, input_size, rnn_type, hidden_size, bidirectional, num_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.num_direction = 2 if self.bidirectional else 1
        self.input_dropout = nn.Dropout(dropout)
        self.rnn = getattr(nn, rnn_type)(input_size=input_size, bidirectional=bidirectional, batch_first=True,
                                         num_layers=num_layers, hidden_size=hidden_size, dropout=dropout)

    def forward(self, input: torch.Tensor):
        batch_size = input.shape[0]
        output, hidden = self.rnn(self.input_dropout(input))

        def _reshape_hidden(hn):
            hn = hn.view(self.num_layers, 2, batch_size, self.hidden_size). \
                permute(0, 2, 1, 3).reshape(self.num_layers, batch_size, self.num_direction * self.hidden_size)
            return hn

        if self.bidirectional and self.rnn_type != "LSTM":
            hidden = _reshape_hidden(hidden)
        elif self.bidirectional and self.rnn_type == "LSTM":
            h, c = _reshape_hidden(hidden[0]), _reshape_hidden(hidden[1])
            hidden = (h, c)

        return output, hidden
