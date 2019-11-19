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

seqs_lens = 1000
enc_lens = 80
dec_lens = 20
batch_size = 32
n_test = 40

hp = BASIC_SEQ2SEQ_HP


train, valid = example_simple_seq2seq_arima(seqs_lens, enc_lens, dec_lens, n_test)
train_dataloader = DataLoader(train, batch_size, shuffle=True)
valid_dataloader = DataLoader(valid, batch_size, shuffle=False)
hp['train_dataloader'] = train_dataloader
hp['valid_dataloader'] = valid_dataloader


model = BasicSeq2Seq(hp)
trainer = Trainer(min_nb_epochs=10, early_stop_callback=False, max_nb_epochs=50, check_val_every_n_epoch=1)
trainer.fit(model)

import ignite

ignite.engine.Engine
ignite.handlers.EarlyStopping