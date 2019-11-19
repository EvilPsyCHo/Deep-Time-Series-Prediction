# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/7 14:42
"""
from dtsp.dataset import arima, create_simple_seq2seq_dataset, log_sin
from dtsp.models import Seq2Seq
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


