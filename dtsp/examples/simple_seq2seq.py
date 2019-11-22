# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/19 16:00
"""
from dtsp.dataset import example_simple_seq2seq_arima
from dtsp.hp_params import BASIC_SEQ2SEQ_HP
from dtsp.models import BasicSeq2Seq
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import torch
import matplotlib.pyplot as plt


seqs_lens = 2000
enc_lens = 80
dec_lens = 40
batch_size = 32
n_test = 200

hp = BASIC_SEQ2SEQ_HP


train, valid = example_simple_seq2seq_arima(seqs_lens, enc_lens, dec_lens, n_test)
plt.plot(train.dataset.seqs)
train_dataloader = DataLoader(train, batch_size, shuffle=True)
valid_dataloader = DataLoader(valid, batch_size, shuffle=False)
hp['train_dataloader'] = train_dataloader
hp['valid_dataloader'] = valid_dataloader


model = BasicSeq2Seq(hp)
path = r'C:\Users\evilp\project\Deep-Time-Series-Prediction\dtsp\examples\logs'
checkpoint = ModelCheckpoint(path, save_best_only=False, prefix='v2')
early = EarlyStopping(patience=5, min_delta=0.001)

trainer = Trainer(early_stop_callback=early, max_nb_epochs=100, checkpoint_callback=checkpoint,
                  check_val_every_n_epoch=1, show_progress_bar=True)
trainer.fit(model)


def plot_validation(item):
    enc = valid[item]['enc_inputs'].reshape(-1)
    pred = model(torch.tensor(valid[item]['enc_inputs']).float().unsqueeze(0), dec_lens).detach().numpy().reshape(-1)
    true = valid[item]['dec_outputs'].reshape(-1)
    plt.plot(enc, label='enc')
    plt.plot(range(len(enc), len(enc) + dec_lens), pred, label='pred')
    plt.plot(range(len(enc), len(enc) + dec_lens), true, label='true')
    plt.legend()


torch.load(r'C:\Users\evilp\project\Deep-Time-Series-Prediction\dtsp\examples\v2_ckpt_epoch_5.ckpt')['state_dict']
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint

trainer = Engine(lambda x: x)
trainer.add_event_handler(Events.EPOCH_COMPLETED)
trainer.run