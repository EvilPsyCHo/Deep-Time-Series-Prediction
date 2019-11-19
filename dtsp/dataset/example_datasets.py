# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/19 16:02
"""
from .curve import arima
from .dataset import SimpleSeq2SeqDataSet
from .utils import walk_forward_split
import numpy as np
from torch.utils.data import Subset


def example_simple_seq2seq_arima(seqs_lens, enc_lens, dec_lens, n_test):
    ar = {1: 0.51, 3: 0.39, 12: 0.1}
    ma = {1: 0.62, 2: 0.20, 6: 0.18}
    var = 1.

    series = arima(seqs_lens, ar=ar, ma=ma, var=var)
    mu = series[:-(n_test+dec_lens)].mean()
    std = series[:n_test+dec_lens].std()
    series = (series - mu) / std

    dset = SimpleSeq2SeqDataSet(series, enc_lens, dec_lens)
    idxes = np.arange(len(dset))
    train_idx, valid_idx = walk_forward_split(idxes, enc_lens, dec_lens, n_test)
    train = Subset(dset, train_idx)
    valid = Subset(dset, valid_idx)
    return train, valid
