# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/25 15:15
"""
import torch.nn as nn


class RNNTransformer(nn.Module):

    def __init__(self, variables):
        self.vars = variables
