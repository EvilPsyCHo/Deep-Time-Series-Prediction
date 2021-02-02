# encoding: utf-8
# Mail: evilpsycho42@gmail.com
from deepseries.modules.temporal_conv import TemporalConv1D

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class State:

    def __init__(self, num_layers):
        self.num_layers = num_layers


class TCN(nn.Module):

    def __init__(self, hidden_size, activation="ReLU", return_state=True,
                 dropout=0.2, block_layers=2, num_layers=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.return_state = return_state
        self.activation = activation
        self.dropout = dropout
        self.block_layers = block_layers
        self.num_layers = num_layers

        assert self.block_layers >= 1
        for n in range(1, num_layers+1):
            for b in range(1, block_layers+1):
                conv = weight_norm(TemporalConv1D(hidden_size, hidden_size, kernel_size=2, dilation=2**(n-1)))
                setattr(self, f"conv_{n}{b}", conv)
                setattr(self, f"dropout_{n}{b}", nn.Dropout(dropout))
                setattr(self, f"{activation}_{n}{b}", getattr(nn, activation)())

    def forward(self, x):
        if self.return_state:
            state = [x]
        for n in range(1, self.num_layers+1):
            for b in range(1, self.block_layers+1):
                x_conv = getattr(self, f"dropout_{n}{b}")(x)
                x_conv = getattr(self, f"conv_{n}{b}")(x_conv)
                x_conv = getattr(self, f"{self.activation}_{n}{b}")(x_conv)
            x = x_conv + x
            if self.return_state:
                state.append(x)
        if self.return_state:
            return x, state
        return x


class TCN2TCN(nn.Module):

    def __init__(self):
        super(TCN2TCN, self).__init__()
        self.encoder = TCN
        self.decoder = TCN

    def forward(self):
        pass

    def predict(self):
        pass



if __name__ == "__main__":
    x = torch.rand(4, 12, 64)
    net = TCN(12, 24, 12)
    print(net)
    print(net(x).shape)
