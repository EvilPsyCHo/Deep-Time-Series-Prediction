# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/7 16:49
"""
import math
import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, size, bias=True, attn_type='general', use_scale=True, **kwargs):
        super(Attention, self).__init__()
        self.use_scale = use_scale
        self.attn_type = attn_type
        if self.attn_type == 'general':
            self.Wq = nn.Linear(size, size, bias=bias)
            self.scale_factor = math.sqrt(size)

        elif self.attn_type == 'general_v2':
            self.Wq = nn.Linear(size, size, bias=bias)
            self.Wk = nn.Linear(size, size, bias=bias)
            self.scale_factor = math.sqrt(size)

        elif self.attn_type == 'dot':
            self.scale_factor = math.sqrt(size)

    def forward(self, q, k):
        """General Attention for rnn-based seq2seq.

        Parameters
        ----------
        q: decoder i-th/all output, shape=(B, SQ/1, H)
        k: encoder all outputs, shape=(B, SK, H)

        Returns
        -------
        weight: attention weight
        attn_value: attention weight value
        """
        if self.attn_type == 'general':
            q = self.Wq(q)  # (B, SQ/1, H) -> (B, SQ/1, H)

            energy = torch.bmm(q, k.transpose(1, 2))  # (B, SQ/1, SK)

            if self.use_scale:
                energy /= self.scale_factor

            weight = torch.softmax(energy, dim=-1)  # (B, SQ/1, SK)
            # (B, SQ, SK) dot (B, SK, H) -> (B, SQ, H)
            attn_value = torch.bmm(weight, k)
            return attn_value, weight

        elif self.attn_type == 'general_v2':
            q = self.Wq(q)  # (B, SQ/1, H) -> (B, SQ/1, H)
            k_trans = self.Wk(k)

            energy = torch.bmm(q, k_trans.transpose(1, 2))  # (B, SQ/1, SK)
            if self.use_scale:
                energy /= self.scale_factor

            weight = torch.softmax(energy, dim=-1)  # (B, SQ/1, SK)
            # (B, SQ, SK) dot (B, SK, H) -> (B, SQ, H)
            attn_value = torch.bmm(weight, k)
            return attn_value, weight

        elif self.attn_type == 'dot':
            energy = torch.bmm(q, k.transpose(1, 2))
            if self.use_scale:
                energy /= self.scale_factor

            weight = torch.softmax(energy, dim=-1)  # (B, SQ/1, SK)
            # (B, SQ, SK) dot (B, SK, H) -> (B, SQ, H)
            attn_value = torch.bmm(weight, k)
            return attn_value, weight


class MultiHeadAttention(nn.Module):
    # TODO: Transformer multi-head-attention
    def __init__(self, n_head, size, bias=True, attn_type='general', use_scale=True, **kwargs):
        super().__init__()
        self.n_head = n_head
        self.heads = nn.ModuleList([Attention(size, bias, attn_type, use_scale, **kwargs) for i in range(n_head)])

    def forward(self, q, k):
        attn_values, attn_weights = [], []
        for layer in self.heads:
            v, w = layer(q, k)
            attn_values.append(v)
            attn_weights.append(w.unsqueeze(2))
        attn_values = torch.cat(attn_values, dim=2)  # (B, S, n_head x H)
        attn_weights = torch.cat(attn_weights, dim=2)  # (B, S n_head, H)
        return attn_values, attn_weights

    def output_size(self):
        pass
