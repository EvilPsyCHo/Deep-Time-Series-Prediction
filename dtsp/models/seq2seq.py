# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/27 15:35
"""
from .base_model import BaseModel
from dtsp.modules import AttentionDecoder, RNNEncoder, RNNTransformer
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch
import random


class Seq2Seq(BaseModel):

    def __init__(self, hp):
        super(Seq2Seq, self).__init__()
        self.hp = hp

        if self.hp['trans_hidden_size'] is not None:
            self.trans = RNNTransformer(**self.hp)
            self.hp['input_size'] = self.hp['target_size'] + self.trans.transform_size()
        else:
            self.trans = None
            self.hp['input_size'] = self.hp['target_size']

        self.enc = RNNEncoder(**self.hp)
        self.dec = AttentionDecoder(**self.hp)

        self.loss_fn = getattr(nn, hp['loss_fn'])()
        self.optimizer = getattr(optim, hp['optimizer'])(self.parameters(), lr=hp['learning_rate'])
        if hp['lr_scheduler'] is not None:
            self.lr_scheduler = getattr(lr_scheduler, hp.get('lr_scheduler'))(self.optimizer,
                                                                              **hp.get('lr_scheduler_kw'))

    def train_batch(self, enc_inputs, dec_inputs, dec_outputs, continuous_x=None, category_x=None):
        enc_lens = enc_inputs.shape[1]
        dec_lens = dec_outputs.shape[1]
        if self.trans is not None:
            trans_x = self.trans(continuous_x, category_x)
            enc_inputs = torch.cat([enc_inputs, trans_x[:, :enc_lens, :]], dim=2)
            dec_inputs = torch.cat([dec_inputs, trans_x[:, enc_lens:, :]], dim=2)

        # TODO implement teacher forcing learning
        enc_outputs, hidden = self.enc(enc_inputs)
        preds, hidden, attns = self.dec(dec_inputs, enc_outputs, hidden)
        loss = self.loss_fn(preds, dec_outputs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, enc_inputs, dec_lens, continuous_x=None, category_x=None):
        enc_lens = enc_inputs.shape[1]
        if self.trans is not None:
            trans_x = self.trans(continuous_x, category_x)
            enc_inputs = torch.cat([enc_inputs, trans_x[:, :enc_lens, :]], dim=2)
            dec_inputs = trans_x[:, :enc_lens, :]