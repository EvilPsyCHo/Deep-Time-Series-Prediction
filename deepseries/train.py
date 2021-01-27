# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/1 17:03
"""
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from deepseries.log import get_logger
import numpy as np
import time
import copy


class EarlyStopping(Exception):
    pass


logger = get_logger(__name__)


class Learner:

    def __init__(self, model, optimizer, root_dir, verbose=32, lr_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.root_dir = root_dir
        self.log_dir = os.path.join(root_dir, 'logs')
        self.model_dir = os.path.join(root_dir, 'checkpoints')
        for i in [self.root_dir, self.log_dir, self.model_dir]:
            if not os.path.exists(i):
                os.mkdir(i)
        self.epochs = 0
        self.best_epoch = -1
        self.best_loss = np.inf
        self.global_steps = 0
        self.use_patient = 0
        self.losses = []
        self.verbose = verbose

    def finish_info(self, message=None):
        if message is not None:
            logger.info(message)
        logger.info(f"training finished, best epoch {self.best_epoch}, best valid loss {self.best_loss:.4f}")

    def eval_cycle(self, data_ld):
        self.model.eval()

        with torch.no_grad():
            valid_loss = 0.
            for x, y in data_ld:
                loss = self.model.batch_loss(x, y).item()
                valid_loss += loss / len(data_ld)
        return valid_loss

    def fit(self, max_epochs, train_dl, valid_dl, early_stopping=True, patient=10, start_save=-1):
        with SummaryWriter(self.log_dir) as writer:
            # writer.add_graph(self.model)
            logger.info(f"start training >>>>>>>>>>>  "
                        f"see log: tensorboard --logdir {self.log_dir}")
            start_epoch = copy.copy(self.epochs)

            try:
                for i in range(max_epochs):
                    self.epochs += 1
                    time_start = time.time()
                    self.model.train()
                    train_loss = 0
                    for j, (x, y) in enumerate(train_dl):
                        self.optimizer.zero_grad()
                        loss = self.model.batch_loss(x, y)
                        loss.backward()
                        self.optimizer.step()
                        loss = loss.item()
                        writer.add_scalar("Loss/train", loss, self.global_steps)
                        self.global_steps += 1
                        train_loss += loss
                        if self.verbose > 0 and self.global_steps % self.verbose == 0:
                            logger.info(f"epoch {self.epochs} / {max_epochs + start_epoch}, "
                                        f"batch {j / len(train_dl) * 100:3.0f}%, "
                                        f"train loss {train_loss / (j + 1):.4f}")
                    valid_loss = self.eval_cycle(valid_dl)
                    writer.add_scalar("Loss/valid", valid_loss, self.global_steps)
                    epoch_use_time = (time.time() - time_start) / 60
                    logger.info(f"epoch {self.epochs} / {max_epochs + start_epoch}, batch 100%, "
                                f"train loss {train_loss / len(train_dl):.4f}, valid loss {valid_loss:.4f}, "
                                f"cost {epoch_use_time:.1f} min")

                    self.losses.append(valid_loss)
                    writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.global_steps)

                    if self.epochs >= start_save:
                        self.save()

                    if early_stopping:
                        if self.epochs > 1:
                            if valid_loss > self.best_loss:
                                self.use_patient += 1
                            else:
                                self.use_patient = 0
                            if self.use_patient >= patient:
                                raise EarlyStopping
                    if valid_loss <= self.best_loss:
                        self.best_loss = valid_loss
                        self.best_epoch = self.epochs
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
            except KeyboardInterrupt:
                self.finish_info("KeyboardInterrupt")
                return
            except EarlyStopping:
                self.finish_info("EarlyStopping")
                return
            self.finish_info()

    def load(self, epoch, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = self.model_dir
        checkpoint = torch.load(os.path.join(checkpoint_dir, f"model-epoch-{epoch}.pkl"))
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epochs = checkpoint['epochs']
        self.lr_scheduler = checkpoint['lr_scheduler']
        self.epochs = epoch
        self.losses = checkpoint['losses']
        self.best_loss = checkpoint['best_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.global_steps = checkpoint['global_steps']

    def save(self):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epochs": self.epochs,
            'lr_scheduler': self.lr_scheduler,
            'losses': self.losses,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'use_patient': self.use_patient,
            'global_steps': self.global_steps,
        }

        name = f"model-epoch-{self.epochs}.pkl"
        torch.save(checkpoint, os.path.join(self.model_dir, name))
