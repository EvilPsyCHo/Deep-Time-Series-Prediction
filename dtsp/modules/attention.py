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

    # General Attention
    def __init__(self, q_size, k_size, attn_size, bias=True, use_scale=True, dropout=0., **kwargs):
        super(GeneralAttention, self).__init__()
        self.use_scale = use_scale
        self.Wq = nn.Linear(q_size, attn_size, bias=bias)
        self.Wk = nn.Linear(k_size, attn_size, bias=bias)
        self.scale_factor = math.sqrt(attn_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """General Attention
        Parameters
        ----------
        q:shape (B, S_q, H_q)
        k:shape (B, S_k, H_k)
        v:shape (B, S_v, H_v)  S_v = S_k
        mask
        Returns
        -------
        weight: attention weight
        attn_value: attention weight value
        """
        q = self.Wq(q)  # (B, S_q, H_q) -> (B, S_q, H_a)
        k = self.Wk(k)  # (B, S_k, H_k) -> (B, S_k, H_a)

        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, S_q, S_k)

        if mask is not None:
            # mask, 0-1 or bool matrix, 1 or true will be masked as input value
            scores = scores.masked_fill(mask, -1e9)

        if self.use_scale:
            scores /= self.scale_factor

        weight = self.dropout(torch.softmax(scores, dim=-1))  # (B, S_q, S_k)

        # (B, S_q, S_k) dot (B, S_k, S_v) -> (B, S_q, S_v)
        attn_value = torch.bmm(weight, v)
        return attn_value, weight


class DotAttention(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q shape: (batch, head, lens_q, dim)
        # k shape: (batch, head, lens_k, dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        # (batch, head, lens_q, lens_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        weight = self.dropout(torch.softmax(scores, dim=-1))
        attn_value = torch.matmul(weight, v)
        # (batch, head, lens_q, lens_k)  (batch, head, lens_k, dim) => (batch, head, lens_q, dim)
        return attn_value, weight


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, n_head, q_size, k_size, v_size, dropout=0.1):
        super().__init__()
        assert v_size % n_head == 0

        # We assume d_v always equals d_k
        self.d_k = v_size // n_head
        self.n_head = n_head

        self.linear_layers = nn.ModuleList([nn.Linear(size, v_size) for size in [q_size, k_size, v_size]])
        self.output_linear = nn.Linear(v_size, v_size)
        self.attention = DotAttention(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        # x: (batch, head, lens_q, dim) => (batch, lens_q, head * dim) = (batch, lens_q, value_size)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        # attn: (batch, head, lens_q, lens_k)

        return self.output_linear(x), attn
