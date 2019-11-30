# coding: utf-8
"""
@Author: zhirui zhou
@Contact: evilpsycho42@gmail.com
@Time: 2019/11/30 下午2:14
"""
import torch
import numpy as np


class Metric(object):

    @staticmethod
    def transform_inputs(*args):
        result = []
        for x in args:
            if isinstance(x, torch.Tensor):
                result.append(x.detach().numpy())
            elif isinstance(x, list):
                result.append(np.array(x))
            else:
                result.append(x)
        return result

    def __call__(self, y_pred, y_true, sample_weight=None, **kwargs):
        y_true, y_pred, sample_weight = self.transform_inputs(y_true, y_pred, sample_weight)
        score = self.score(y_true, y_pred, sample_weight, **kwargs)
        return score

    def score(self, *args, **kwargs):
        raise NotImplemented

    @property
    def name(self):
        return str(self.__class__.__name__)


class RMSE(Metric):

    def __init__(self):
        pass

    def score(self, y_pred, y_true, sample_weight=None):
        return np.sqrt(np.mean(np.average((y_true - y_pred) ** 2, axis=0, weights=sample_weight)))
