# coding: utf-8
"""
@Author: zhirui zhou
@Contact: evilpsycho42@gmail.com
@Time: 2019/12/1 上午7:17
"""
import torch
from torch import optim
from torch import nn
from dtsp.record import Record
from dtsp import metrics
from tqdm import tqdm
import numpy as np
import os


class BaseModel:

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.optimizer = None
        self.metric = None
        self.lr_scheduler = None
        self.compile_params = None
        self.hp = None
        self.record = None
        self.loss_fn = None

    def set_optimizer(self, optimizer, lr, kwargs):
        kwargs = {} if kwargs is None else kwargs
        self.optimizer = getattr(optim, optimizer)(self.parameters(), lr, **kwargs)

    def set_loss_fn(self, loss_fn):
        self.loss_fn = getattr(nn, loss_fn)()

    def set_lr_scheduler(self, lr_scheduler, lr_scheduler_kw):
        if lr_scheduler is not None:
            lr_scheduler_kw = {} if lr_scheduler_kw is None else lr_scheduler_kw
            self.lr_scheduler = getattr(optim.lr_scheduler, lr_scheduler)(
                self.optimizer, **lr_scheduler_kw)

    def set_metric(self, metric):
        if metric is not None:
            self.metric = getattr(metrics, metric)()

    def compile(self, optimizer, loss_fn, lr=0.001, metric="RMSE",
                lr_scheduler=None, lr_scheduler_kw=None, optimizer_kw=None):
        self.compile_params = {
            'optimizer': optimizer,
            'loss_fn': loss_fn,
            'lr': lr,
            'metric': metric,
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_kw': lr_scheduler_kw,
            'optimizer_kw': optimizer_kw,
        }
        self.set_optimizer(optimizer, lr, optimizer_kw)
        self.set_loss_fn(loss_fn)
        self.set_metric(metric)
        self.set_lr_scheduler(lr_scheduler, lr_scheduler_kw)
        self.record = Record()

    def train_batch(self, *args, **kwargs):
        # do batch gradient and return loss.item()
        raise NotImplementedError

    def eval_batch(self, *args, **kwargs):
        # do batch prediction and return loss, y_pred, y_true
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        # do batch prediction
        raise NotImplementedError

    def train_cycle(self, trn_ld):
        self.train()
        trn_loss = []
        with tqdm(trn_ld) as bar:
            for i, batch in enumerate(bar):
                _loss = self.train_batch(**batch)
                trn_loss.append(_loss)
                bar.set_description_str(desc=f'batch {i + 1} / {len(trn_ld)}, loss {_loss:.3f}', refresh=True)
                if hasattr(self, "lr_scheduler"):
                    self.lr_scheduler.step(self.record.epochs + i / len(bar))
        trn_loss = np.mean(trn_loss)
        return trn_loss

    def eval_cycle(self, val_ld):
        self.eval()
        val_loss = val_score = 0
        with torch.no_grad():
            for batch in val_ld:
                batch_loss, y_pred, y_true = self.eval_batch(**batch)
                val_loss += batch_loss / len(val_ld)
                val_score += self.metric(y_pred, y_true) / (len(val_ld))
        return val_loss, val_score

    def fit(self, n_epochs, trn_ld, val_ld, early_stopping=5, save_every_n_epochs=1, save_best_model=True):
        total_epochs = self.record.epochs + n_epochs
        init_epochs = self.record.epochs + 1
        best_model_path = None
        for epoch in range(n_epochs):
            trn_loss = self.train_cycle(trn_ld)
            val_loss, val_score = self.eval_cycle(val_ld)
            print(f'epoch {epoch + init_epochs} / {total_epochs}: '
                  f'train loss {trn_loss:.3f} '
                  f'val loss {val_loss:.3f} {self.metric.name} {val_score:.3f}')
            self.record.update(trn_loss, val_loss, self.optimizer.param_groups[0]['lr'])
            save_every = (epoch - 1) % save_every_n_epochs == 0 if isinstance(save_every_n_epochs, int) else False
            save_best = (self.record.best_model_epoch == self.record.epochs) and save_best_model

            if save_every or save_best:
                model_name = f'{self.__class__.__name__}_epoch_{self.record.epochs}'
                model_info = f'{self.record.best_model_loss:.3f}'
                save_path = self.save(f'{model_name}_{model_info}.pkl')
                if save_best:
                    best_model_path = save_path

            if isinstance(early_stopping, int):
                if self.record.use_early_stop(early_stopping):
                    print(f'early_stopping ! '
                          f'current epochs: {self.record.epochs}, '
                          f'best epochs: {self.record.best_model_epoch}.')
                    print(f'model save in {best_model_path}')
                    return
        print(f'best model save in {best_model_path}')

    def check_path(self):
        if not os.path.exists(self.hp['path']):
            os.makedirs(self.hp['path'], exist_ok=True)
            print(f'create model path: {self.hp["path"]}')

    def save(self, name):
        self.check_path()
        checkpoint = {
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'hp': self.hp,
            'record': self.record,
            'compile_params': self.compile_params
        }

        if hasattr(self, "lr_scheduler"):
            checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()
        save_path = os.path.join(self.hp["path"], name)
        torch.save(checkpoint, save_path)
        return save_path

    def reload(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.record = checkpoint['record']
        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    def best_model_path(self):
        if self.record.best_model_epoch is None:
            return None

        model_name = f'{self.__class__.__name__}_epoch_{self.record.best_model_epoch}'
        model_info = f'{self.record.best_model_loss:.3f}'
        path = os.path.join(self.hp["path"], f'{model_name}_{model_info}.pkl')
        return path

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        model = cls(checkpoint['hp'])
        model.compile(**checkpoint['compile_params'])
        model.load_state_dict(checkpoint['model'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        model.record = checkpoint['record']
        if hasattr(model, "lr_scheduler"):
            model.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        return model
