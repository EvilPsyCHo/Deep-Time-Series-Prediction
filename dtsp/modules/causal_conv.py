# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/25 15:53
"""
from torch import nn
import torch


class CausalConv1d(torch.nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        self.__padding = (kernel_size - 1) * dilation
        # kernel_size=2, dilation=2, padding=2, outputs=[:, :, :-2]
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


class DilationBlockV1(nn.Module):
    """
    WAVENET A GENERATIVE MODEL FOR RAW AUDIO
    paper: https://arxiv.org/abs/1609.03499
    """

    def __init__(self, input_channels, residual_channels, kernel_size, dilation):
        super().__init__()
        self.conv_in = nn.Conv1d(input_channels, residual_channels, kernel_size=1)
        self.conv_left = CausalConv1d(residual_channels, residual_channels,
                                      kernel_size=kernel_size, dilation=dilation)
        self.conv_right = CausalConv1d(residual_channels, residual_channels,
                                       kernel_size=kernel_size, dilation=dilation)
        self.conv_out = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = torch.relu(x)

        x_left = torch.tanh(self.conv_left(x))
        x_right = torch.sigmoid(self.conv_right(x))
        skip = torch.mul(x_left, x_right)
        skip = self.conv_out(skip)
        # add residual connection
        x = x + skip

        return x, skip


# TODO condition wavenet
# TODO share condition Wcg Wcf weights?
class ConditionDilationBlock(nn.Module):

    def __init__(self, input_channels, residual_channels, condition_channels, kernel_size, dilation):
        super().__init__()
        self.conv_in = nn.Conv1d(input_channels, residual_channels, kernel_size=1)
        self.conv_x_f = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation=dilation)
        self.conv_x_g = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation=dilation)
        self.conv_c_f = CausalConv1d(condition_channels, residual_channels, kernel_size, dilation=dilation)
        self.conv_c_g = CausalConv1d(condition_channels, residual_channels, kernel_size, dilation=dilation)

    def forward(self, x, c):
        x = self.conv_in(x)
        x = torch.relu(x)

        left = torch.tanh(self.conv_x_f(x) + self.conv_x_g(c))
        right = torch.sigmoid(self.conv_c_g(x) + self.conv_c_g(c))
        skip = torch.mul(left, right)
        skip = self.conv_out(skip)
        # add residual connection
        x = x + skip

        return x, skip
