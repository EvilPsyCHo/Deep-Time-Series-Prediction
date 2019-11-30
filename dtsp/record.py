# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/27 13:21
"""
import numpy as np


class Record(object):

    def __init__(self):
        self.trn_loss = []
        self.val_loss = []
        self.lr = []
        self.epochs = 0
        self.best_model_epoch = None
        self.best_model_loss = np.inf
        self.epsilon = 0.0005

    def update(self, trn_loss, val_loss, lr):
        """Update Per Epoch

        :param trn_loss:
        :param val_loss:
        :param lr:
        :return:
        """
        self.trn_loss.append(trn_loss)
        self.val_loss.append(val_loss)
        self.lr.append(lr)
        self.epochs += 1
        if val_loss + self.epsilon <= self.best_model_loss:
            self.best_model_epoch = self.epochs
            self.best_model_loss = val_loss

    def use_early_stop(self, patient):
        if self.best_model_epoch is None:
            return True
        if self.epochs - self.best_model_epoch <= patient:
            return False
        else:
            return True
