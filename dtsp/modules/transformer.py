# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/25 15:15
"""
import torch.nn as nn
import torch
from dtsp.modules import Embeddings


class RNNTransformer(nn.Module):

    def __init__(self, trans_hidden_size, trans_continuous_var=None,
                 trans_category_var=None, trans_bidirectional=True,
                 trans_rnn_type='LSTM', trans_n_layer=1, trans_residual=True,
                 dropout=0.1, **kwargs):
        super().__init__()
        self.transformer_hidden_size = trans_hidden_size
        self.bidirectional = trans_bidirectional
        self.trans_residual = trans_residual
        self.n_continuous_var = 0 if trans_continuous_var is None else trans_continuous_var
        self.n_category_var = 0 if trans_category_var is None else len(trans_category_var)
        self.category_size = 0 if trans_category_var is None else sum([dim for _, dim in trans_category_var])
        self.rnn = getattr(nn, trans_rnn_type)(self.category_size + self.n_continuous_var, trans_hidden_size,
                                               batch_first=True, bidirectional=trans_bidirectional, num_layers=trans_n_layer)
        self.embed = None
        if trans_category_var is not None:
            self.embed = Embeddings(trans_category_var)
        self.dropout = nn.Dropout(dropout)

    def forward(self, continuous_x=None, category_x=None):
        # B x S x N
        if category_x is not None:
            category_x = self.embed(category_x)
        if category_x is not None and continuous_x is not None:
            x = torch.cat([continuous_x, category_x], dim=2)
        elif category_x is not None and continuous_x is None:
            x = category_x
        elif category_x is None and continuous_x is not None:
            x = continuous_x
        else:
            raise ValueError

        outputs, _ = self.rnn(x)
        if self.trans_residual:
            outputs = torch.cat([outputs, x], dim=-1)
        return outputs

    def transform_size(self):
        rnn_size = (int(self.bidirectional) + 1) * self.transformer_hidden_size
        residual_size = self.category_size + self.n_continuous_var
        if self.trans_residual:
            return rnn_size + residual_size
        else:
            return rnn_size
