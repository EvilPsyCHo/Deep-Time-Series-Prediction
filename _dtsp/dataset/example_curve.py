# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/7 14:42
"""
import numpy as np


def log_sin(lens, noise_var=0):
    source = np.sin(np.arange(lens)) + np.log(np.arange(10, lens + 10))
    noise = np.random.normal(0, noise_var, size=lens)
    return source + noise


def arima(lens, ar, ma, var=1, c=0, period=12):
    get_param = lambda x: np.array([[lag, auto_regress_coff] for lag, auto_regress_coff in x.items()])
    ar_param = get_param(ar)
    ar_lag = ar_param[:, 0].astype(int)

    ma_param = get_param(ma)
    ma_lag = ma_param[:, 0].astype(int)

    max_lag = max(max(ar_lag), max(ma_lag))
    y = np.zeros(lens + max_lag + 1)
    epsilon = np.random.normal(0, var, lens + max_lag + 1)
    auto_coff, mov_coff = ar_param[:, 1], ma_param[:, 1]

    for i in range(max_lag + 1, len(y)):
        auto_index = i - ar_lag
        mov_index = i - ma_lag  # + epsilon[i]
        y[i] = c + np.dot(y[auto_index], auto_coff) + np.dot(epsilon[mov_index], mov_coff) + epsilon[
            i] + 0.5 * np.sin(np.pi * i / period)
    return y[max_lag + 1:]
