# encoding: utf-8
# Author: 周知瑞
# Mail: evilpsycho42@gmail.com
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SimpleSeriesDataset(Dataset):

    def __init__(self, series, enc_lens, dec_lens):
        self.series = series
        self.enc_lens = enc_lens
        self.dec_lens = dec_lens

    def __len__(self):
        return len(self.series) - self.enc_lens - self.dec_lens

    def __getitem__(self, item):
        l = self.enc_lens + self.dec_lens
        return (self.series[item: item + l].reshape(1, -1),
                self.series[item + 1: item + l + 1].reshape(1, -1))


def log_sin_series(lens):
    source = np.log(list(range(1, lens + 1))) + np.sin(list(range(lens)))
    noise = np.random.normal(0, 1, size=lens)
    series = source + noise
    return series.astype('float32'), source.astype('float32')


def log_sin_dataset(train_size, test_size, enc_lens, preds_lens):
    n = train_size + test_size + enc_lens + preds_lens - 1
    series, source = log_sin_series(n)
    train_series = series[: train_size + enc_lens + preds_lens - 1]
    trainset = SimpleSeriesDataset(train_series, enc_lens, preds_lens)
    valid_series = series[-test_size - enc_lens - preds_lens + 1:]
    testset = SimpleSeriesDataset(valid_series, enc_lens, preds_lens)
    return trainset, testset

from fastai.train import Learner
from fastai.basic_data import DataBunch
from dtsp.models.wavenet import SimpleWaveNet
import torch.nn as nn
import torch


trainset, validset = log_sin_dataset(5000, 1000, 100, 50)
traindl = DataLoader(trainset, batch_size=12, shuffle=True, drop_last=True)
validdl = DataLoader(validset, batch_size=128)

model = SimpleWaveNet(1, 32, 32, 32, 1, [2**i for i in range(8)])
bunch = DataBunch(traindl, validdl)
learner = Learner(bunch, model, loss_func=nn.MSELoss())
learner.fit(10, lr=0.001)

model.train()

model.predict_seqs(torch.tensor(validset[0][0].reshape(1, 1, -1)), 50)
