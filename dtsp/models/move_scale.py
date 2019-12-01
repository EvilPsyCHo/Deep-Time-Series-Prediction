# coding: utf-8
"""
@Author: zhirui zhou
@Contact: evilpsycho42@gmail.com
@Time: 2019/11/30 下午3:39
"""


class MoveScale:

    def __init__(self, dim):
        self.dim = dim
        self.mu = None
        self.std = None

    def fit(self, x):
        self.mu = x.mean(self.dim).unsqueeze(self.dim)
        self.std = x.std(self.dim).unsqueeze(self.dim)

    def transform(self, *tensors):
        return [(t - self.mu.expand_as(t)) / self.std.expand_as(t) for t in tensors]

    def inverse(self, *tensors):
        return [t * self.std.expand_as(t) + self.mu.expand_as(t) for t in tensors]
