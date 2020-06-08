# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/12 16:33
"""
from deepseries.models import RNN2RNN, Wave2WaveV1
from deepseries.train import Learner
from deepseries.dataset import Values, create_seq2seq_data_loader, forward_split
import numpy as np
from torch.optim import Adam
import torch


batch_size = 16
enc_len = 36
dec_len = 12
series = np.sin(np.arange(0, 1000))
series = series.reshape(1, 1, -1)
train_idx, valid_idx = forward_split(np.arange(series.shape[2]), enc_len=14, valid_size=200)


def test_rnn2rnn():
    train_dl = create_seq2seq_data_loader(series, enc_len=14, dec_len=7, time_idx=train_idx,
                                          batch_size=12, num_iteration_per_epoch=12, seq_last=False)
    valid_dl = create_seq2seq_data_loader(series, enc_len=14, dec_len=7, time_idx=valid_idx,
                                          batch_size=12, num_iteration_per_epoch=12, seq_last=False)
    model = RNN2RNN(1, 256, 64, num_layers=1, attn_heads=1, attn_size=12, rnn_type='LSTM')
    model.cuda()
    opt = Adam(model.parameters(), 0.001)
    learner = Learner(model, opt, ".")
    learner.fit(10, train_dl, valid_dl, early_stopping=False)


def test_wave2wave_v1():
    train_dl = create_seq2seq_data_loader(series, enc_len=14, dec_len=7, time_idx=train_idx,
                                          batch_size=12, num_iteration_per_epoch=12, seq_last=True)
    valid_dl = create_seq2seq_data_loader(series, enc_len=14, dec_len=7, time_idx=valid_idx,
                                          batch_size=12, num_iteration_per_epoch=12, seq_last=True)
    model = Wave2WaveV1(1)
    model.cuda()
    opt = Adam(model.parameters(), 0.001)
    learner = Learner(model, opt, ".")
    learner.fit(100, train_dl, valid_dl, early_stopping=False)
