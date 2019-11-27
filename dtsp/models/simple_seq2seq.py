# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/19 14:12
"""
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch
import random
import os
import numpy as np
from tqdm import tqdm
from .base_model import BaseModel
from dtsp.modules import SimpleRNNDecoder, SimpleRNNEncoder


class SimpleSeq2Seq(BaseModel):

    def __init__(self, hp):
        """
        Parameters
        ----------
        hp (dict): HyperParams of model. Import and Modify it from dtsp.param.BasicSeq2SeqHP.

        Notes
        -----
        Only support epoch based lr_scheduler.
        """
        super(SimpleSeq2Seq, self).__init__()
        self.hp = hp

        self.encoder = SimpleRNNEncoder(**hp)
        self.decoder = SimpleRNNDecoder(**hp)

        self.loss_fn = getattr(nn, hp['loss_fn'])()
        self.optimizer = getattr(optim, hp['optimizer'])(self.parameters(), lr=hp['learning_rate'])
        if hp['lr_scheduler'] is not None:
            self.lr_scheduler = getattr(lr_scheduler, hp.get('lr_scheduler'))(self.optimizer, **hp.get('lr_scheduler_kw'))

    def train_batch(self, **batch):
        enc_inputs = batch['enc_inputs']
        dec_inputs = batch['dec_inputs']
        dec_outputs = batch['dec_outputs']

        self.optimizer.zero_grad()
        outputs = []
        _, hidden = self.encoder(enc_inputs)
        dec_inputs_i = dec_inputs[:, 0].unsqueeze(1)
        n_steps = dec_outputs.shape[1]
        for i in range(n_steps):
            output, hidden = self.decoder(dec_inputs_i, hidden)
            outputs.append(output)
            if i < n_steps - 1:
                if random.random() < self.hp['teacher_forcing_rate']:
                    dec_inputs_i = dec_inputs[:, i+1].unsqueeze(1)
                else:
                    dec_inputs_i = output
        outputs = torch.cat(outputs, dim=1)
        loss = self.loss_fn(outputs, dec_outputs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate_batch(self, **batch):
        enc_inputs = batch['enc_inputs']
        dec_outputs = batch['dec_outputs']
        y_pred = self.predict(enc_inputs, dec_outputs.shape[1])
        loss = self.loss_fn(y_pred, dec_outputs)
        return loss.item()

    def predict(self, enc_seqs, n_step):
        _, hidden = self.encoder(enc_seqs)
        inputs = enc_seqs[:, -1].unsqueeze(1)
        outputs = []
        for i in range(n_step):
            inputs, hidden = self.decoder(inputs, hidden)
            outputs.append(inputs)
        return torch.cat(outputs, dim=1)
