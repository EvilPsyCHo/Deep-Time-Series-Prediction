# encoding: utf-8
# Author: 周知瑞
# Mail: evilpsycho42@gmail.com

from ..modules.temporal_conv import TemporalBlock

import torch
from torch import nn


class TCNEncoder(nn.Module):

    def __init__(self, input_channels, hidden_channels, num_blocks, kernel_size, dropout=0.2):
        super().__init__()
        self.num_blocks = num_blocks
        self.conv_input = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(*[TemporalBlock(hidden_channels, hidden_channels, kernel_size, (kernel_size-1)**n, dropout) for n in range(self.num_blocks)])

    def encode(self, x):
        states = [x]
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < self.num_blocks - 1:
                states.append(x)
        return x
