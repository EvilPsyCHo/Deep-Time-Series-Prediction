# coding: utf-8
"""
@Author: zhirui zhou
@Contact: evilpsycho42@gmail.com
@Time: 2019/11/30 下午10:53
"""
from torch import nn
import torch
from .base_model import BaseModel
from .move_scale import MoveScale
from dtsp.modules import DilationBlockV1, ConditionDilationBlock, RNNTransformer


class WaveNet(nn.Module, BaseModel):

    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.trans = RNNTransformer(**hp)
        self.conv_blocks = nn.ModuleList()
        self.hp['condition_channels'] = self.trans.transform_size()

        self.conv_first = ConditionDilationBlock(self.hp['target_size'],
                                                 self.hp['residual_channels'],
                                                 self.trans.transform_size(),
                                                 kernel_size=2,
                                                 dilation=self.hp['dilation'][0]
                                                 )
        for idx, d in enumerate(self.hp['dilation'][1:]):
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
        self.move_scale = MoveScale(2) if hp['use_move_scale'] else None

    def forward(self, x, c):
        skips = torch.zeros(x.shape[0], self.hp['residual_channels'], x.shape[2])
        x, skip = self.conv_first(x, c)
        skips += skip
        for layer in self.conv_blocks:
            x, skip = layer(x)
            skips += skip

        skips = torch.relu(skips)
        out = self.conv_out(skips)
        return out

    def predict(self, x, n_steps, continuous_x=None, category_x=None, use_move_scale=False):
        """

        Parameters
        ----------
        x (Tensor): shape B x C x S
        n_steps (int): num of predict step

        Returns
        -------
        y_pred (Tensor): prediction
        """
        trans_outputs = self.trans(continuous_x, category_x)[: 1:, :].transpose(1, 2)
        use_move_scale = use_move_scale and self.move_scale is not None
        if use_move_scale:
            self.move_scale.fit(x)
            x = self.move_scale.transform(x)

        y_pred = []

        for step in range(n_steps):
            y_step = self(x, trans_outputs[:, :, step: self.hp['enc_lens'] + step])[:, :, -1].unsqueeze(2)
            y_pred.append(y_step)
            x = torch.cat([x[:, :, 1:], y_step], dim=2)
        y_pred = torch.cat(y_pred, dim=2)

        if use_move_scale:
            y_pred = self.move_scale.inverse(y_pred)

        return y_pred

    def train_batch(self, enc_inputs, dec_outputs, continuous_x=None, category_x=None):
        # TODO: only teacher forcing learning , add self learning
        if self.move_scale is not None:
            self.move_scale.fit(enc_inputs)
            enc_inputs, dec_outputs = self.move_scale.transform(enc_inputs, dec_outputs)
        self.optimizer.zero_grad()
        dec_lens = dec_outputs.shape[-1]
        x = torch.cat([enc_inputs, dec_outputs[:, :, :-1]], dim=2)
        c = self.trans(continuous_x, category_x)[:, 1:, :].transpose(0, 1)
        y_pred = self(x, c)[:, :, -dec_lens:]
        loss = self.loss_fn(y_pred, dec_outputs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_batch(self, enc_inputs, dec_outputs):
        if self.move_scale is not None:
            self.move_scale.fit(enc_inputs)
            enc_inputs, dec_outputs = self.move_scale.transform(enc_inputs, dec_outputs)
        n_steps = dec_outputs.shape[-1]
        y_pred = self.predict(enc_inputs, n_steps)
        loss = self.loss_fn(y_pred, dec_outputs)

        if self.move_scale is not None:
            y_pred, dec_outputs = self.move_scale.inverse(y_pred, dec_outputs)
        return loss.item(), y_pred, dec_outputs