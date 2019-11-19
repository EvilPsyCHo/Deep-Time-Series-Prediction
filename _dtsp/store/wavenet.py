# encoding: utf-8
# Author: 周知瑞
# Mail: evilpsycho42@gmail.com
import torch
from torch import nn

from dtsp.store.mudules import ResidualGatedBlock


class SimpleWaveNet(nn.Module):

    """

    Notes:

        inputs -> convolutions -> ReLU -> conv 1x1 -> ReLU -> conv 1x1
    """

    def __init__(self, in_channels, skip_channels, dilation_channels, residual_channels,
                 outputs_channels, dilation, activation="ReLU", kernel_size=2, bias=True,
                 loss_fn='MSELoss'):
        super().__init__()
        self.skip_channels = skip_channels
        self.loss_fn = getattr(nn, loss_fn)

        # 统一inputs_channels和residual_channels
        self.processing = nn.Conv1d(in_channels, residual_channels, kernel_size=1)

        self.residual_gated_blocks = nn.ModuleList()
        for n_layer, d in enumerate(dilation):
            block = ResidualGatedBlock(residual_channels, skip_channels, dilation_channels,
                                       kernel_size, d, bias)
            self.residual_gated_blocks.append(block)

        hidden_channels = skip_channels * 2

        self.out = nn.Sequential(
            # nn.BatchNorm1d(skip_channels),
            getattr(nn, activation)(),
            nn.Conv1d(skip_channels, hidden_channels, kernel_size=1, bias=True),
            getattr(nn, activation)(),
            nn.Conv1d(hidden_channels, outputs_channels, kernel_size=1, bias=True),
        )

    def forward(self, x):
        batch, _, lens = x.size()
        skips = torch.zeros([batch, self.skip_channels, lens])
        x = self.processing(x)
        for layer in self.residual_gated_blocks:
            x, skip = layer(x)
            skips = skip + skips
        outputs = self.out(skips)
        return outputs

    def predict_seqs(self, x, lens):
        batch = x.size()[0]
        predict_all = torch.zeros((batch, 1, lens))

        for step in range(lens):
            predict = self(x)[:, :, -1].detach()
            predict_all[:, :, step] = predict
            x = torch.cat([x[:, :, :-1], predict.reshape(batch, -1, 1)], dim=2)
        return predict_all

    def init_weights(self):

        def _init_func(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.00)

        self.apply(_init_func)
