# coding: utf-8
"""
@Author: zhirui zhou
@Contact: evilpsycho42@gmail.com
@Time: 2019/11/7 上午1:37
"""
from torch import nn
import torch


class CausalConv1d(torch.nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        self.__padding = (kernel_size - 1) * dilation
        # 正常卷积pad：kernel_size=3, dilation=1, padding=2
        # 空洞卷积pad: kernel_size=3, dilation=2, padding=4
        self.__padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)

    def forward(self, inputs):
        result = super(CausalConv1d, self).forward(inputs)
        if self.padding != 0:
            return result[:, :, :-self.__padding]
        return result


class ResidualGatedBlock(nn.Module):

    """
    Notes:

        - Wavenet structure:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|
    """
    def __init__(self, residual_channels, skip_channels, dilation_channels, kernel_size, dilation, bias):
        super().__init__()

        self.conv_gate = nn.Sequential(
            CausalConv1d(residual_channels, dilation_channels, kernel_size, dilation=dilation, bias=bias),
            nn.Sigmoid())

        self.conv_filter = nn.Sequential(
            CausalConv1d(residual_channels, dilation_channels, kernel_size, dilation=dilation, bias=bias),
            nn.Tanh())

        self.conv_skip = nn.Conv1d(dilation_channels, skip_channels, kernel_size=1, bias=bias)
        self.conv_output = nn.Conv1d(dilation_channels, residual_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        gate = self.conv_gate(x)
        filters = self.conv_filter(x)
        concat = gate * filters
        skip = self.conv_skip(concat)
        outputs = self.conv_output(concat) + x
        return outputs, skip