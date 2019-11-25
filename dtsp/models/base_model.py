# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/25 15:03
"""
import torch.nn as nn
import torch
import os


class BaseModel(nn.Module):

    def train_op(self, args, **kwargs):
        raise NotImplemented

    def predict(self, args, **kwargs):
        raise NotImplemented

    def save(self, name):
        checkpoint = {
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'hp': self.hp,
        }

        if hasattr(self, "lr_scheduler"):
            checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()
        torch.save(checkpoint, os.path.join(self.hp["path"], name))

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        model = cls(checkpoint['hp'])
        model.load_state_dict(checkpoint['model'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        if hasattr(model, "lr_scheduler"):
            model.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        return model
