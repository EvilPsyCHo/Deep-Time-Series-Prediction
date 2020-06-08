# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/17 11:34
"""
from torch.optim.lr_scheduler import _LRScheduler
import math


class ReduceCosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi)  * gamma ** epoch)

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, gamma=0.998, eta_min=5e-5, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.gamma = gamma
        super(ReduceCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [(self.eta_min + self.gamma ** self.last_epoch * (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2)
                for base_lr in self.base_lrs]


if __name__ == "__main__":
    import torch
    net = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(net.parameters(), 0.01)
    lr_scheduler = ReduceCosineAnnealingLR(optimizer, 64)

    record = []
    for i in range(1000):
        record.append(lr_scheduler.get_lr())
        lr_scheduler.step()
    import matplotlib.pyplot as plt
    plt.plot(record)
