# coding: utf-8
"""
@Author: zhirui zhou
@Contact: evilpsycho42@gmail.com
@Time: 2019/11/9 上午12:29
"""
from dtsp.dataset import arima, create_simple_wavenet_dataset, log_sin
from dtsp.models import SimpleWaveNet
import matplotlib.pyplot as plt
import time

start = time.time()

target_dim = 1
enc_lens = 60
dec_lens = 20
data_lens = 150
ar = {1: 0.51, 3: 0.39, 12: 0.1}
ma = {1: 0.62, 2: 0.20, 6: 0.18}
var = 1.
n_test = 20
batch_size = 12

series = arima(data_lens, ar=ar, ma=ar, var=var)
# series = log_sin(data_lens)
mu = series[:-(n_test+dec_lens)].mean()
std = series[:n_test+dec_lens].std()
series = (series - mu) / std
plt.plot(series)


trainset, validset = create_simple_wavenet_dataset(series, enc_lens, dec_lens, n_test, batch_size)
model = SimpleWaveNet(1, dec_lens, 36, 8, "/home/zhouzr/test", dropout=0.1, verbose=1, wavenet_mode='v2')
history = model.fit_generator(trainset, validset, epochs=100, verbose=2, shuffle=True)

model.plot_loss()


def plot_prediction(idx):
    f = plt.figure()
    x, y_true = validset.get(idx)
    y_pred = model.predict(x, dec_lens).reshape(-1)
    y_true = y_true.reshape(-1)
    plt.plot(x.reshape(-1))
    plt.plot(range(enc_lens, enc_lens+dec_lens), y_pred, label='pred')
    plt.plot(range(enc_lens, enc_lens+dec_lens), y_true, label='true')
    plt.legend()
    return f

plot_prediction(10)
end = time.time()
