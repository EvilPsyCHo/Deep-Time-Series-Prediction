# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/6 13:49
"""
import torch
from torch import nn


class SimpleSeq2Seq(nn.Module):

    def __init__(self, target_dim, hidden_size, activation='Tanh', dropout=0.0):
        super(SimpleSeq2Seq, self).__init__()
        self.target_dim = target_dim
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(target_dim, hidden_size, num_layers=1, bias=True, batch_first=True)
        self.decoder = nn.LSTM(target_dim, hidden_size, num_layers=1, bias=True, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, target_dim),
            getattr(nn, activation)(),
            nn.Dropout(dropout))

    def forward(self, enc_inputs, dec_inputs):
        batch, dec_lens, _ = dec_inputs.shape
        enc_outputs, enc_hidden = self.encoder(enc_inputs)
        dec_hidden = enc_hidden
        dec_outputs, _ = self.decoder(dec_inputs, dec_hidden)
        preds = self.out(dec_outputs)
        return preds

    def predict(self, enc_inputs, predict_steps):
        with torch.no_grad():
            batch, _, _ = enc_inputs.shape
            enc_outputs, enc_hidden = self.encoder(enc_inputs)
            dec_inputs = enc_inputs[:, -1, :].unsqueeze(1)
            dec_hidden = enc_hidden
            preds = torch.zeros(batch, predict_steps, 1)
            for i in range(predict_steps):
                dec_outputs, dec_hidden = self.decoder(dec_inputs, dec_hidden)
                dec_inputs = self.out(dec_outputs)
                preds[:, i] = dec_inputs
            return preds
