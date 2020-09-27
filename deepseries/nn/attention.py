# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/7 9:22
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Align(nn.Module):
    """
    Compute 'Scaled Dot Product Attention

    References:
        https://github.com/codertimo/BERT-pytorch/blob/d10dc4f9d5a6f2ca74380f62039526eb7277c671/bert_pytorch/model/attention/single.py#L8
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class Attention(nn.Module):
    """
    Take in model size and number of heads.
    general attention

    Args:
        query, key, value, mask. shape like (B, S, N)
    Returns:
        attention_value, (B, query_lens, N)
        attention_weight, (B, Head, query_lens, values_lens)
    References:
        https://github.com/codertimo/BERT-pytorch/blob/d10dc4f9d5a6f2ca74380f62039526eb7277c671/bert_pytorch/model/attention/single.py#L8
    """
    def __init__(self, heads, attn_size, query_size, key_size, value_size, dropout):
        super().__init__()
        assert attn_size % heads == 0

        # We assume d_v always equals d_k
        self.d_k = attn_size // heads
        self.h = heads

        self.linear_layers = nn.ModuleList([nn.Linear(s, attn_size) for s in [query_size, key_size, value_size]])
        self.output_linear = nn.Linear(attn_size, attn_size)
        self.align = Align()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """inputs shape (B, S, N)"""

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.align(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # (B, S, N), (B, H, S_q, S_k)
        return self.output_linear(x), attn


if __name__ == "__main__":

    q = torch.rand(4, 7, 30)
    k = torch.rand(4, 12, 9)

    attn = Attention(heads=3, attn_size=9, query_size=30, value_size=9, key_size=9, dropout=0.)
    values, weights = attn(q, k, k)
