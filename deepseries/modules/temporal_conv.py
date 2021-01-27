# Mail: evilpsycho42@gmail.com

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn.modules.conv import _ConvNd, _single


class TemporalConv1D(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 padding_mode="zeros", stride=1, groups=1, bias=True):
        padding = (kernel_size - 1) * dilation
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(TemporalConv1D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)

    def forward(self, x):
        if self.padding_mode != 'zeros':
            x = F.conv1d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                         self.weight, self.bias, self.stride,
                         _single(0), self.dilation, self.groups)
        else:
            x = F.conv1d(x, self.weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)
        return x[:, :, :-self.padding[0]]


class TemporalBlock(nn.Module):

    """
    An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
    https://arxiv.org/pdf/1803.01271.pdf?source=post_page---------------------------
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv_block = nn.Sequential(
            weight_norm(TemporalConv1D(in_channels, out_channels, kernel_size, dilation)),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(TemporalConv1D(in_channels, out_channels, kernel_size, dilation)),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.conv_res = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x_block = self.conv_block(x)
        x_res = self.conv_res(x)
        return x_block + x_res


if __name__ == "__main__":
    # test causal conv
    x = torch.rand(4, 12, 100)
    conv = TemporalConv1D(12, 4, kernel_size=3, dilation=3)
    print(f"conv input shape: {x.shape}, output shape: {conv(x).shape}")
