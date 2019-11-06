# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/6 15:02
"""
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TorchSimpleSeriesDataSet(Dataset):

    def __init__(self, series, enc_lens, dec_lens):
        self.s = series.astype('float32')
        self.el = enc_lens
        self.dl = dec_lens

    def __len__(self):
        return len(self.s) - self.el - self.dl + 1

    def __getitem__(self, item):
        enc_inputs = self.s[item: item + self.el].reshape(-1, 1)
        dec_inputs = self.s[item + self.el - 1: item + self.el + self.dl - 1].reshape(-1, 1)
        dec_outputs = self.s[item + self.el: item + self.el + self.dl].reshape(-1, 1)
        return (enc_inputs, dec_inputs), dec_outputs


def walk_forward_split(series_index, n_test, enc_lens, dec_lens):
    train_index = series_index[: -n_test]
    valid_index = series_index[-(dec_lens + n_test - 1 + enc_lens):]
    return train_index, valid_index


def log_sin_curve(total_lens):
    source = np.sin(np.arange(total_lens)) + np.log(np.arange(1, total_lens + 1))
    noise = np.random.normal(0, 0.5, size=total_lens)
    x = source + noise
    return x, source


def create_dataset(x, enc_lens, dec_lens, n_valid, n_test, normalization=True):
    idxes = np.arange(len(x))
    train_idx, tmp_idx = walk_forward_split(idxes, n_test+n_valid, enc_lens, dec_lens)
    valid_idx, test_idx = walk_forward_split(tmp_idx, n_test, enc_lens, dec_lens)
    x_train, x_valid, x_test = x[train_idx], x[valid_idx], x[test_idx]

    if normalization:
        mu = x_train.mean()
        std = x_train.std()
        x_train = (x_train - mu) / std
        x_valid = (x_valid - mu) / std
        x_test = (x_test - mu) / std

    train = TorchSimpleSeriesDataSet(x_train, enc_lens, dec_lens)
    valid = TorchSimpleSeriesDataSet(x_valid, enc_lens, dec_lens)
    test = TorchSimpleSeriesDataSet(x_test, enc_lens, dec_lens)
    return train, valid, test


from fastai.basic_train import Learner

