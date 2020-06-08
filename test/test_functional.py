# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/27 14:04
"""
import pytest
import numpy as np
from deepseries.functional import *


def test_smooth():
    x = np.random.rand(3, 100)
    center = smooth(x, window=3, ratio=0.5, mode='center')
    causal = smooth(x, window=3, ratio=0.5, mode='causal')
    weights = np.array([0.25, 0.5, 0.25])
    # assert np.sum(np.array(x[0, -2:].tolist() + [x[0, -1]]) * weights) == center[0, -1]
    # assert np.sum(np.array(x[1, -2:].tolist() + [x[1, -1]]) * weights) == center[1, -1]
    # assert np.sum(np.array(x[1, -3:] * weights)) == center[1, -2]
    # assert np.sum(np.array(x[0, -3:] * weights)) == causal[0, -1]

def test_make_lags():
    power = np.random.rand(3, 10)
    p1 = make_lags(power, [1, 2])
    p1s = make_lags(power, [2, 3], True)
    assert p1.shape == p1s.shape
    assert p1.shape == (3, 2, 10)


def test_get_trend():
    power = np.random.rand(3, 400)
    trend = get_trend(power, 365, False)
    assert trend.shape == (3, 400)

