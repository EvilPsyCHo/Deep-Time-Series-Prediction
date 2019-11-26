# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/25 15:53
"""
import torch.nn as nn
import torch
from .base_model import BaseModel
from dtsp.modules import DilationBlockV1


class SimpleWaveNet(BaseModel):

    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        self.conv_blocks = nn.ModuleList()

        for idx, d in enumerate(range(self.hp['dilation'])):
            if idx == 0:
                self.conv_blocks.append(DilationBlockV1(self.hp['target_size'], self.hp['residual_channels'],
                                                        kernel_size=2, dilation=d))
            else:
                self.conv_blocks.append(DilationBlockV1(self.hp['residual_channels'], self.hp['residual_channels'],
                                                        kernel_size=2, dilation=d))
        self.conv_out1 = nn.Conv1d(self.hp['residual_channels'], self.hp['residual_channels'], kernel_size=1)
        self.conv_out2 = nn.Conv1d(self.hp['residual_channels'], self.hp['target_size'], kernel_size=1)

    def forward(self, x):
        skips = torch.zeros(x.shape[0], self.hp['residual_channels'], x.shape[2])
        
        for layer in self.conv_blocks:
            x, skip = layer(x)
            skips += skip

        skips = torch.relu(skips)
        out1 = torch.relu(self.conv_out1(skips))
        out2 = self.conv_out2(out1)
        return out2

    def train_op(self):
        pass

    def predict(self, args, **kwargs):
        pass
