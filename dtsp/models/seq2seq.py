# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/27 15:35
"""
from .base import BaseModel
from dtsp.modules import RNNDecoder, RNNEncoder, RNNTransformer
from dtsp import metrics
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
        self.dec = RNNDecoder(**self.hp)

        self.loss_fn = getattr(nn, hp['loss_fn'])()
        self.optimizer = getattr(optim, hp['optimizer'])(self.parameters(), lr=hp['learning_rate'])
        if hp['lr_scheduler'] is not None:
            self.lr_scheduler = getattr(lr_scheduler, hp.get('lr_scheduler'))(self.optimizer,
                                                                              **hp.get('lr_scheduler_kw'))
        self.metric = getattr(metrics, hp['metric'])()

    def train_batch(self, enc_inputs, dec_inputs, dec_outputs, continuous_x=None, category_x=None):
        self.optimizer.zero_grad()
        enc_lens = enc_inputs.shape[1]
        dec_lens = dec_outputs.shape[1]

        use_teacher_forcing = random.random() < self.hp['teacher_forcing_rate']
        if use_teacher_forcing:
            if self.trans is not None:
                trans_x = self.trans(continuous_x, category_x)
                enc_inputs = torch.cat([enc_inputs, trans_x[:, :enc_lens, :]], dim=2)
                dec_inputs = torch.cat([dec_inputs, trans_x[:, enc_lens:, :]], dim=2)
            enc_outputs, hidden = self.enc(enc_inputs)
            preds, hidden, attns = self.dec(dec_inputs, enc_outputs, hidden)
        else:
            preds = self.predict(enc_inputs, dec_lens, continuous_x, category_x, return_attns=False)
        loss = self.loss_fn(preds, dec_outputs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate_batch(self, enc_inputs, dec_inputs, dec_outputs, continuous_x=None, category_x=None):
        dec_lens = dec_outputs.shape[1]
        preds = self.predict(enc_inputs, dec_lens, continuous_x, category_x)
        loss = self.loss_fn(preds, dec_outputs)
        score = self.metric(preds, dec_outputs)
        return loss, score

    def predict(self, enc_inputs, dec_lens, continuous_x=None, category_x=None, return_attns=False):
        if not self.hp['use_attn']:
            return_attns = False

        enc_lens = enc_inputs.shape[1]
        if self.trans is not None:
            trans_x = self.trans(continuous_x, category_x)
            enc_inputs = torch.cat([enc_inputs, trans_x[:, :enc_lens, :]], dim=2)
            dec_input_i = enc_inputs[:, -1, :].unsqueeze(1)
        else:
            dec_input_i = enc_inputs[:, -1, :].unsqueeze(1)
        enc_outputs, hidden = self.enc(enc_inputs)

        preds = []
        attns = []

        for i in range(dec_lens):
            pred_i, hidden, attn_i = self.dec(dec_input_i, enc_outputs, hidden)
            preds.append(pred_i)
            attns.append(attn_i)
            if self.trans is not None:
                dec_input_i = torch.cat([pred_i, trans_x[:, enc_lens+i, :].unsqueeze(1)], dim=2)
            else:
                dec_input_i = pred_i
        preds = torch.cat(preds, dim=1)

        if return_attns:
            attns = torch.cat(attns, dim=1)
            return preds, attns
        else:
            return preds
