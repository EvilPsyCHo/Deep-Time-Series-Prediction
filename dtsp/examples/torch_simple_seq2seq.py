# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/18 16:28
"""
import time
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from dtsp.models import SimpleSeq2Seq
from dtsp.dataset import SimpleDataSet, arima, walk_forward_split


start = time.time()

target_dim = 1
enc_lens = 80
dec_lens = 40
data_lens = 1000
ar = {1: 0.51, 3: 0.39, 12: 0.1}
ma = {1: 0.62, 2: 0.20, 6: 0.18}
var = 1.
n_test = 100
batch_size = 64

series = arima(data_lens, ar=ar, ma=ar, var=var)
# series = log_sin(data_lens)
mu = series[:-(n_test+dec_lens)].mean()
std = series[:n_test+dec_lens].std()
series = (series - mu) / std
plt.plot(series)

dset = SimpleDataSet(series, enc_lens, dec_lens)
train_idx, valid_idx = walk_forward_split(list(range(len(dset))), enc_lens, dec_lens, n_test)
train_set = Subset(dset, train_idx)
valid_set = Subset(dset, valid_idx)



from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader

train_ld = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_ld = DataLoader(valid_set, batch_size=batch_size, shuffle=False)


model = SimpleSeq2Seq(1, 12)
loss_fn = MSELoss()
opt = Adam(model.parameters(), 0.001)

for i in range(10):
    for batch in train_ld:
        opt.zero_grad()
        preds = model(*batch)
        loss = loss_fn(preds, batch[-1])
        loss.backward()
        opt.step()
        print(loss.item())


from fastai.data_block import DataBunch
from fastai.basic_train import Learner

from torch.optim.lr_scheduler import *

from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar

import pytorch_lightning as pl

from pytorch_lightning import Trainer