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


if __name__ == "__main__":
    layers = nn.ModuleList([
        DilationBlockV1(1, residual_channels=12, kernel_size=2, dilation=1),
        DilationBlockV1(12, residual_channels=12, kernel_size=2, dilation=2),
        DilationBlockV1(12, residual_channels=12, kernel_size=2, dilation=4),
        DilationBlockV1(12, residual_channels=12, kernel_size=2, dilation=8),
        DilationBlockV1(12, residual_channels=12, kernel_size=2, dilation=16),
    ])
    from torch.nn import init
    # layer = DilationBlockV1(12, residual_channels=12, kernel_size=2, dilation=16)
    x = torch.normal(0, 0.05, (1, 1, 196))
    skips = torch.zeros((1, 12, 196))

    conv_out = nn.Conv1d(12, 1, kernel_size=1)

    print(x.shape)

    x_in = x
    for layer in layers:
        x_in, skip = layer(x_in)
        skips += skip
        print(x_in.shape, skips.shape)

    y = conv_out(skips)
    print(y.shape)
    import matplotlib.pyplot as plt

    plt.plot(x.detach().numpy().reshape(-1), label='x')
    plt.plot(y.detach().numpy().reshape(-1), label='y')
    plt.legend()
    plt.show()
