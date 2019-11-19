# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/8 15:40
"""
import os
import matplotlib.pyplot as plt


class BaseKerasModel(object):

    def __init__(self, *args, **kwargs):
        self.history = None

    def fit(self, *args, **kwargs):
        # update self.history in the end
        raise NotImplemented

    def fit_generator(self, *args, **kwargs):
        # update self.history in the end
        raise NotImplemented

    def load_weights(self, file_path):
        if os.path.exists(file_path):
            self.model.load_weights(file_path)

    def save_weights(self, file_path):
        self.model.load_weights(file_path)

    def plot_loss(self, figsize=(12, 6)):
        f = plt.figure(figsize=figsize)
        h = self.history.history
        plt.plot(h['loss'], label='train_loss')
        plt.plot(h['val_loss'], label='val_loss')
        plt.legend()
        return f
