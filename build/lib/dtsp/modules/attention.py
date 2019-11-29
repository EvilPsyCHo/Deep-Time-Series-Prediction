# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/7 16:49
"""
import math
import torch
import torch.nn as nn


class GeneralAttention(nn.Module):

    def __init__(self, size, bias=True):
        super(GeneralAttention, self).__init__()
        self.Wq = nn.Linear(size, size, bias=bias)
        self.scale_factor = math.sqrt(size)

    def forward(self, q, k):
        """General Attention for rnn-based seq2seq.

        Parameters
        ----------
        q: decoder i-th/all output, shape=(B, Sq/1, H)
        k: encoder all outputs, shape=(B, Sk, H)

        Returns
        -------
        weight: attention weight
        attn_value: attention weight value
        """
        q = self.Wq(q)
        energy = torch.bmm(q, k.transpose(1, 2)) / self.scale_factor  # (B, S1/1, S2)
        weight = torch.softmax(energy, dim=-1)  # (B, S1/1, S2)
        attn_value = torch.bmm(weight, k)  # (B, S1/1, H)
        return attn_value, weight
