# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/16 10:58
"""
import torch
from torch.nn import functional as F
from deepseries.log import get_logger


logger = get_logger(__name__)


class MSE:

    def __call__(self, input, target, weight=None):
        if weight is None:
            loss = F.mse_loss(input, target, reduction='mean')
        else:
            ret = F.mse_loss(input, target, reduction='none')
            loss = torch.mean(ret * weight)
        return loss


class RMSE:

    def __call__(self, input, target, weight=None):
        if weight is None:
            ret = F.mse_loss(input, target, reduction='mean')
        else:
            ret = F.mse_loss(input, target, reduction='none') * weight
        ret[ret == 0.] = 1e-6
        loss = torch.sqrt(torch.mean(ret))
        return loss


class SMAPE:

    def __call__(self, input, target, weight=None):
        mae = torch.abs(input - target)
        divide = input + target
        divide[divide == 0.] = 1e-6
        smape = mae / divide
        if weight is not None:
            smape *= weight
        return torch.mean(smape)


class MAPE:

    def __call__(self, input, target, weight=None):
        mae = torch.abs(input - target)
        divide = target
        divide[divide == 0.] = 1e-6
        smape = mae / divide
        if weight is not None:
            smape *= weight
        return torch.mean(smape)


class RNNStabilityLoss:
    """

    RNN outputs -> loss

    References:
        https://arxiv.org/pdf/1511.08400.pdf
    """

    def __init__(self, beta=1e-5):
        self.beta = beta

    def __call__(self, rnn_output):
        if self.beta == .0:
            return .0
        l2 = torch.sqrt(torch.sum(torch.pow(rnn_output, 2), dim=-1))
        l2 = self.beta * torch.mean(torch.pow(l2[:, 1:] - l2[:, :-1], 2))
        return l2


class RNNActivationLoss:

    """
    RNN outputs -> loss
    """

    def __init__(self, beta=1e-5):
        self.beta = beta

    def __call__(self, rnn_output):
        if self.beta == .0:
            return .0
        return torch.sum(torch.norm(rnn_output)) * self.beta
