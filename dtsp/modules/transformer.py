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
                 trans_category_var=None, trans_bidirectional=True, trans_rnn_type='LSTM', **kwargs):
        super().__init__()
        self.transformer_hidden_size = trans_hidden_size
        self.bidirectional = trans_bidirectional
        self.n_continuous_var = 0 if trans_continuous_var is None else trans_continuous_var
        self.n_category_var = 0 if trans_category_var is None else len(trans_category_var)
        self.category_size = 0 if trans_category_var is None else sum([dim for _, dim in trans_category_var])
        self.rnn = getattr(nn, trans_rnn_type)(self.category_size + self.n_continuous_var,
                                               trans_hidden_size, batch_first=True, bidirectional=trans_bidirectional)
        self.embed = None
        if trans_category_var is not None:
            self.embed = Embeddings(trans_category_var)

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

        x, _ = self.rnn(x)
        return x

    def transform_size(self):
        return (int(self.bidirectional) + 1) * self.transformer_hidden_size
