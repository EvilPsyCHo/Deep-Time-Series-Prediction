# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/25 15:53
"""
import torch.nn as nn
import torch
from torch import optim
from torch.optim import lr_scheduler
from .base import BaseModel
from dtsp.modules import DilationBlockV1
from dtsp import metrics


class SimpleWaveNet(BaseModel):

    def __init__(self, hp):
        super(SimpleWaveNet, self).__init__()
        self.hp = hp

        self.conv_blocks = nn.ModuleList()

        for idx, d in enumerate(self.hp['dilation']):
            if idx == 0:
                self.conv_blocks.append(DilationBlockV1(self.hp['target_size'], self.hp['residual_channels'],
                                                        kernel_size=2, dilation=d))
            else:
                self.conv_blocks.append(DilationBlockV1(self.hp['residual_channels'], self.hp['residual_channels'],
                                                        kernel_size=2, dilation=d))

        self.conv_out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.hp['dropout']),
            nn.Conv1d(self.hp['residual_channels'], self.hp['residual_channels'], kernel_size=1),
            nn.ReLU(),
            nn.Dropout(self.hp['dropout']),
            nn.Conv1d(self.hp['residual_channels'], self.hp['target_size'], kernel_size=1),
        )
        self.loss_fn = getattr(nn, hp['loss_fn'])()
        self.optimizer = getattr(optim, hp['optimizer'])(self.parameters(), lr=hp['learning_rate'])
        if hp['lr_scheduler'] is not None:
            self.lr_scheduler = getattr(lr_scheduler, hp.get('lr_scheduler'))(self.optimizer,
                                                                              **hp.get('lr_scheduler_kw'))
        self.metric = getattr(metrics, hp['metric'])()

    def forward(self, x):
        skips = torch.zeros(x.shape[0], self.hp['residual_channels'], x.shape[2])
        for layer in self.conv_blocks:
            x, skip = layer(x)
            skips += skip

        skips = torch.relu(skips)
        out = self.conv_out(skips)
        return out

    def predict(self, x, n_steps):
        """

        Parameters
        ----------
        x (Tensor): shape B x C x S
        n_steps (int): num of predict step

        Returns
        -------
        y_pred (Tensor): prediction
        """
        y_pred = []

        for step in range(n_steps):
            y_step = self(x)[:, :, -1].unsqueeze(2)
            y_pred.append(y_step)
            x = torch.cat([x[:, :, 1:], y_step], dim=2)
        y_pred = torch.cat(y_pred, dim=2)

        return y_pred

    def train_batch(self, enc_inputs, dec_outputs):
        # TODO: only teacher forcing learning , add self learning
        self.optimizer.zero_grad()
        dec_lens = dec_outputs.shape[-1]
        x = torch.cat([enc_inputs, dec_outputs[:, :, :-1]], dim=2)
        y_pred = self(x)[:, :, -dec_lens:]
        loss = self.loss_fn(y_pred, dec_outputs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate_batch(self, enc_inputs, dec_outputs):
        n_steps = dec_outputs.shape[-1]
        y_pred = self.predict(enc_inputs, n_steps)
        loss = self.loss_fn(y_pred, dec_outputs)
        score = self.metric(y_pred, dec_outputs)
        return loss.item(), score
