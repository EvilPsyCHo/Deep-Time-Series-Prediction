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

        self.check_path()

    def check_path(self):
        if not os.path.exists(self.hp['path']):
            os.mkdir(self.hp['path'])
            print(f'create model path: {self.hp["path"]}')
        else:
            print(f'path {self.hp["path"]} already exists')

    def train_op(self, **batch):
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
            if random.random() < self.hp['teacher_forcing_rate']:
                dec_inputs_i = dec_inputs[:, i].unsqueeze(1)
            else:
                dec_inputs_i = output
        outputs = torch.cat(outputs, dim=1)
        loss = self.loss_fn(outputs, dec_outputs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit_one_cycle(self, trn_ld, val_ld=None):
        self.train()
        trn_loss = []

        with tqdm(trn_ld) as bar:
            for i, batch in enumerate(bar):
                _loss = self.train_op(**batch)
                trn_loss.append(_loss)
                bar.set_description_str(desc=f'batch {i} / {len(trn_ld)}, loss {_loss:.3f}', refresh=True)
        trn_loss = np.mean(trn_loss)

        if val_ld is not None:
            val_loss = self.evaluate(val_ld)
        else:
            val_loss = None

        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.step()

        return trn_loss, val_loss

    def fit(self, n_epochs, trn_ld, val_ld=None):
        for epoch in range(n_epochs):
            trn_loss, val_loss = self.fit_one_cycle(trn_ld, val_ld)
            if val_loss is not None:
                print(f'epoch {epoch} / {n_epochs}, loss {trn_loss:.3f}, val loss {val_loss:.3f}')
            else:
                print(f'epoch {epoch} / {n_epochs}, loss {trn_loss:.3f}')

    def evaluate(self, val_ld):
        self.eval()

        y = []
        y_pred = []

        with torch.no_grad():
            for batch in val_ld:
                _enc_input = batch['enc_inputs']
                _dec_output = batch['dec_outputs']
                _dec_pred = self.predict(_enc_input, _dec_output.shape[1])
                y.append(_dec_output)
                y_pred.append(_dec_pred)
            y = torch.cat(y, dim=0)
            y_pred = torch.cat(y_pred, dim=0)
            loss = self.loss_fn(y_pred, y)
        return loss.item()

    def predict(self, enc_seqs, n_step):
        _, hidden = self.encoder(enc_seqs)
        inputs = enc_seqs[:, -1].unsqueeze(1)
        outputs = []
        for i in range(n_step):
            inputs, hidden = self.decoder(inputs, hidden)
            outputs.append(inputs)
        return torch.cat(outputs, dim=1)
