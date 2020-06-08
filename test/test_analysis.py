# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/27 15:11
"""
import pytest
from deepseries.analysis import SeriesAnalysisModel
import numpy as np


def test_analysis_model():
    x = np.random.rand(4, 500) + 1e-3
    x[0][0] = np.nan
    x[0][1] = 0
    model = SeriesAnalysisModel(x)
    model.plot_valid()
    model.get_trend(365).plot_trend()
    model.plot_trend(0)

    model.get_autocorr(np.arange(1, 300)).plot_autocorr()

    assert model.mask.sum() == 2
    assert model.valid_lens[0] == 498


if __name__ == "__main__":
    test_analysis_model()
