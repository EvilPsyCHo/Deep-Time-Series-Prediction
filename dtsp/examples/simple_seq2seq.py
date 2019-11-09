# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/7 14:42
"""
from dtsp.dataset import arima, create_simple_seq2seq_dataset, log_sin
from dtsp.models import SimpleSeq2Seq
import matplotlib.pyplot as plt
import numpy as np
import time

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

start = time.time()

target_dim = 1
enc_lens = 80
dec_lens = 40
data_lens = 1000
ar = {1: 0.51, 3: 0.39, 12: 0.1}
ma = {1: 0.62, 2: 0.20, 6: 0.18}
var = 1.
n_test = 100
batch_size = 64

series = arima(data_lens, ar=ar, ma=ar, var=var)
# series = log_sin(data_lens)
mu = series[:-(n_test+dec_lens)].mean()
std = series[:n_test+dec_lens].std()
series = (series - mu) / std
plt.plot(series)


trainset, validset = create_simple_seq2seq_dataset(series, enc_lens, dec_lens, n_test, batch_size)
model = SimpleSeq2Seq(1, 32, "/home/zhouzr/test", dropout=0.1, verbose=1)
history = model.fit_generator(trainset, validset, epochs=100, verbose=2, shuffle=True)

model.plot_loss()


def plot_prediction(idx):
    f = plt.figure()
    x, y_true = validset.get(idx)
    y_pred = model.predict(x['enc_input'].reshape(1, enc_lens, target_dim), dec_lens)
    y_pred = np.concatenate(y_pred, axis=1).reshape(-1)
    y_true = y_true.reshape(-1)
    plt.plot(x['enc_input'].reshape(-1))
    plt.plot(range(enc_lens, enc_lens+dec_lens), y_pred, label='pred')
    plt.plot(range(enc_lens, enc_lens+dec_lens), y_true, label='true')
    plt.legend()
    return f


plot_prediction(99)
end = time.time()

print(f'finished, use time {(start-end) / 60}')
