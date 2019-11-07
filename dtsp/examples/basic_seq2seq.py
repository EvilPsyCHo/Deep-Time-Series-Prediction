# coding: utf-8
"""
@Author: zhirui zhou
@Contact: evilpsycho42@gmail.com
@Time: 2019/11/6 下午10:34
"""

from dtsp.dataset import log_sin_curve, create_dataset
from dtsp.torch_models.seq2seq import SimpleSeq2Seq
from torch import nn
from torch.utils.data import DataLoader
from fastai.basic_data import DataBunch, AdamW
from fastai.basic_train import Learner
from fastai.callbacks import EarlyStoppingCallback

# config
data_lens = 200
enc_lens = 50
dec_lens = 30
n_valid = 10
n_test = 10
target_dim = 1
hidden_size = 20
activation = 'Tanh'
dropout = 0.0
lr = 0.001
batch_size = 12
epochs = 10

x, source = log_sin_curve(data_lens)
train, valid, test = create_dataset(x, enc_lens, dec_lens, n_valid, n_test, normalization=True)
train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid, batch_size, shuffle=False)
test_dl = DataLoader(test, batch_size, shuffle=False)

bunch = DataBunch(train_dl, valid_dl, test_dl, device='cpu')

model = SimpleSeq2Seq(target_dim, hidden_size, activation, dropout)
optimizer = AdamW
loss_fn = nn.MSELoss()

learner = Learner(bunch, model, optimizer, loss_fn)
early_stop = EarlyStoppingCallback(learner, patience=10)
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle()
learner.recorder.plot_lr(show_moms=True)
learner.recorder.plot_losses()

learner.fit(10, lr=lr, callbacks=[early_stop])
learner.validate(test_dl)


import torch
import matplotlib.pyplot as plt


def plot(x):
    (enc, _), y_true = test[x]
    step = y_true.shape[0]
    y_pred = learner.model.predict(torch.tensor(enc.reshape(1, -1, 1)), step)
    y_true = y_true.reshape(-1)
    enc = enc.reshape(-1)
    y_pred = y_pred.numpy().reshape(-1)
    plt.plot(enc)
    plt.plot(range(len(enc), len(enc)+step), y_pred, label='pred')
    plt.plot(range(len(enc), len(enc) + step), y_true, label='true')
    plt.legend()

plot(5)