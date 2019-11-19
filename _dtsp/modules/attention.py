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
        q: decoder i-th step output, shape=(1, B, H)
        k: encoder all steps outputs, shape=(S, B, H)

        Returns
        -------
        weight: attention weight
        attn_value: attention weight value
        """
        k = k.transpose(0, 1)
        q = q.transpose(0, 1)
        q = self.wq(q)
        energy = torch.bmm(q, k) / self.scale_factor  # (B, 1, S)
        weight = torch.softmax(energy, dim=-1)  # (B, 1, S)
        attn_value = torch.bmm(weight, k)  # (B, 1, H)
        return weight, attn_value
