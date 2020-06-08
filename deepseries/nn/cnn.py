# encoding: utf-8
# Author: 周知瑞
# Mail: evilpsycho42@gmail.com
from typing import Dict, Any, Callable, Tuple

from torch import nn
from torch.nn import functional as F
import torch


class WaveQueue:

    def __init__(self, dilation):
        self.max_len = dilation + 1  # kernel size is 2
        self.values = None

    def enqueue(self, x):
        assert x.shape[2] == 1
        self.values = torch.cat([self.values[:, :, 1:], x], dim=2)

    def dequeue(self):
        return self.values[:, :, [-self.max_len, -1]]

    def clear_buffer(self):
        self.values = None

    def init(self, x):
        self.values = torch.zeros(x.shape[0], x.shape[1], self.max_len).to(x.device)
        self.values[:, :, -min(self.max_len, x.shape[2]):] = x[:, :, -min(self.max_len, x.shape[2]):]


class WaveStates:
    def __init__(self, num_blocks, num_layers):
        self.queues = [WaveQueue(2 ** j) for i in range(num_blocks) for j in range(num_layers)]

    def init(self, layer, x):
        self.queues[layer].init(x)

    def enqueue(self, layer, x):
        self.queues[layer].enqueue(x)

    def dequeue(self, layer):
        return self.queues[layer].dequeue()

    def clear_buffer(self):
        for q in self.queues:
            q.clear_buffer()


class CausalConv1d(nn.Conv1d):
    """1D Causal Convolution Layer

    Args:
        inputs, Tensor(batch, input_unit(kernel_size), sequence)

    Returns:
        Tensor(batch, output_unit(kernel_size), sequence)
    """

    def __init__(self, in_channels, out_channels, dilation=1, bias=True,
                 padding_mode='zeros'):
        kernel_size = 2
        self.shift = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=self.shift,
            dilation=dilation,
            groups=1,
            bias=bias,
            padding_mode=padding_mode)

    def forward(self, inputs):
        return super(CausalConv1d, self).forward(inputs)[:, :, :-self.shift]


class WaveLayer(nn.Module):

    def __init__(self, residual_channels, skip_channels, dilation):
        super(WaveLayer, self).__init__()
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilation = dilation
        self.conv_dilation = CausalConv1d(residual_channels, residual_channels, dilation=dilation)
        self.conv_filter = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.conv_gate = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.conv_skip = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
        self.conv_residual = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)

    def forward(self, x):
        """
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
        """
        x_dilation = self.conv_dilation(x)
        x_filter = self.conv_filter(x_dilation)
        x_gate = self.conv_gate(x_dilation)
        x_conv = torch.sigmoid(x_gate) * torch.tanh(x_filter)
        x_skip = self.conv_skip(x_conv)
        x_res = self.conv_residual(x_conv) + x_dilation
        return x_res, x_skip

    def last_forward(self, x):
        x_dilation = F.conv1d(x, self.conv_dilation.weight, self.conv_dilation.bias,
                              self.conv_dilation.stride, 0, dilation=1, groups=self.conv_dilation.groups)
        x_filter = self.conv_filter(x_dilation)
        x_gate = self.conv_gate(x_dilation)
        x_conv = torch.sigmoid(x_gate) * torch.tanh(x_filter)
        x_skip = self.conv_skip(x_conv)
        x_res = self.conv_residual(x_conv) + x_dilation
        return x_res, x_skip


class WaveNet(nn.Module):

    def __init__(self, input_channels, residual_channels, skip_channels, num_blocks, num_layers, mode="add"):
        super(WaveNet, self).__init__()
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.input_conv = nn.Conv1d(input_channels, residual_channels, kernel_size=1)
        self.mode = mode
        self.wave_layers = nn.ModuleList([WaveLayer(residual_channels, skip_channels, 2 ** i)
                                          for _ in range(num_blocks) for i in range(num_layers)])

    def encode(self, x):
        state = WaveStates(self.num_blocks, self.num_layers)
        x = self.input_conv(x)
        skips = 0.
        for i, layer in enumerate(self.wave_layers):
            state.init(i, x)
            x, skip = layer(x)
            skips += skip
        return skips, state

    def decode(self, x, state):
        x = self.input_conv(x)
        skips = 0.
        for i, layer in enumerate(self.wave_layers):
            state.enqueue(i, x)
            x = state.dequeue(i)
            x, skip = layer.last_forward(x)
            skips += skip
        return skips, state
