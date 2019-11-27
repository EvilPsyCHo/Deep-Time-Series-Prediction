# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/19 16:00
"""
from dtsp.dataset import example_simple_wavenet_arima
from dtsp.param import SIMPLE_WAVENET_HP
from dtsp.models import SimpleWaveNet
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import matplotlib.pyplot as plt


seqs_lens = 500
enc_lens = 80
dec_lens = 40
batch_size = 32
n_test = 100

hp = SIMPLE_WAVENET_HP
hp['path'] = r'C:\Users\evilp\project\Deep-Time-Series-Prediction\dtsp\examples\logs'
hp['dilation'] = [1, 2, 4, 8, 16]


train, valid = example_simple_wavenet_arima(seqs_lens, enc_lens, dec_lens, n_test)
plt.plot(train.dataset.seqs)
train_dataloader = DataLoader(train, batch_size, shuffle=True)
valid_dataloader = DataLoader(valid, batch_size, shuffle=False)


model = SimpleWaveNet(hp)

model.fit(10, train_dataloader, valid_dataloader, early_stopping=2)
path = r"C:\Users\evilp\project\Deep-Time-Series-Prediction\dtsp\examples\logs\test1.pkl"
model.save(path)
model = SimpleWaveNet.load(path)
# os.remove(path)
