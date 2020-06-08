# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/20 17:17
"""
import pytest
from deepseries.dataset import create_seq2seq_data_loader
import numpy as np


def test_create_seq2seq_data_loader():
    x = np.random.rand(30, 1, 24)
    dl = create_seq2seq_data_loader(x, 12, 12, np.arange(x.shape[-1]), batch_size=4, num_iteration_per_epoch=30, seq_last=True)
    for i, batch in enumerate(dl):
        pass
    assert i == 30-1



