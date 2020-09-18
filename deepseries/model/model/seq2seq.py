# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/9/18 10:18
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from easydict import EasyDict
from collections import OrderedDict


class RNNEncoder(nn.Module):

    def __init__(self, input_size, rnn_type, hidden_size, bidirectional, num_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_dropout = nn.Dropout(dropout)
        self.rnn = getattr(nn, rnn_type)(input_size=input_size, bidirectional=bidirectional, batch_first=True,
                                         num_layers=num_layers, hidden_size=hidden_size, dropout=dropout)

    def forward(self, input: torch.Tensor):
        batch_size = input.shape[0]
        output, hidden = self.rnn(input)

        def _reshape_hidden(hn):
            hn = hn.view(self.num_layers, 2, batch_size, self.hidden_size). \
                permute(0, 2, 1, 3).reshape(self.num_layers, batch_size, 2 * self.hidden_size)
            return hn

        if self.bidirectional and self.rnn_type != "LSTM":
            hidden = _reshape_hidden(hidden)
        elif self.bidirectional and self.rnn_type == "LSTM":
            h, c = _reshape_hidden(hidden[0]), _reshape_hidden(hidden[1])
            hidden = (h, c)

        return output, hidden


class RNNDecoder(nn.Module):

    def __init__(self, input_size, output_size, rnn_type, hidden_size, num_layers, dropout, max_len=7):
        super().__init__()
        self.input_size = input_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

        self.input_dropout = nn.Dropout(dropout)
        self.rnn = getattr(nn, rnn_type)(input_size=input_size, batch_first=True,
                                         num_layers=num_layers, hidden_size=hidden_size, dropout=dropout)
        self.attn = nn.Linear(input_size + hidden_size, max_len)
        self.attn_combine = nn.Linear(self.input_size + hidden_size, hidden_size)
        self.regression = nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor, encoder_output):
        # single step
        # step input -> (batch, 1, N); previous dec hidden (layer, batch, hidden_size)
        batch_size = input.shape[0]
        attn_weights = F.softmax(self.attn(torch.cat([input, hidden.permute(1, 0, 2).view(batch_size, -1)])), dim=1)
        # attn_weights (batch, seq_len)
        # encoder_output (batch, seq_len, enc_hidden)
        attn_applied = torch.bmm(encoder_output.transpose(2, 1), attn_weights.unsqueeze(2)).transpose(2, 1)
        # attn_applied = (batch, 1, enc_hidden)
        concat = torch.cat([input, attn_applied], dim=2)
        concat = F.relu(self.attn_combine(concat))
        output, hidden = self.rnn(concat, hidden)
        return output, hidden


class Seq2Seq(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.encoder = RNNEncoder()


data_config = OrderedDict({
    "encode":
        {
            "categorical": [("month", 13, 2), ("weekday", 8, 2)],
         },
})


class MultipleEmbedding(nn.Module):

    def __init__(self, *variable_params):
        # example: *[(name, size, embed_size), ... ]
        super().__init__()
        self.params = variable_params
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(s, e) for (name, s, e) in variable_params
        })

    def forward(self, input):
        return torch.cat([self.embeddings[name](input[name]) for (name, _, _) in self.params], dim=2)


class PlaceHolder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ContinuousInput(nn.Module):

    def __init__(self, *vars_params):
        # {"name": xxx, "dtype": "", "size", "