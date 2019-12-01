# coding: utf-8
"""
@Author: zhirui zhou
@Contact: evilpsycho42@gmail.com
@Time: 2019/11/30 下午2:27
"""
from dtsp.metrics import RMSE
import numpy as np


def test_rmse():
    y_true = [1, 2]
    y_pred = [2, 0]
    metric = RMSE()
    assert metric(y_true, y_pred) == np.sqrt(2.5)

    y_true = np.zeros([4, 12, 1])
    y_pred = np.zeros([4, 12, 1])
    sample_weight = np.random.rand(4)
    metric = RMSE()
    assert metric(y_true, y_pred, sample_weight) == 0
