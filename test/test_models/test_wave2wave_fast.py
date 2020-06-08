# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/27 14:25
"""
# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/12 16:33
"""
from deepseries.models import Wave2Wave
from deepseries.train import Learner
from deepseries.data import Value, create_seq2seq_data_loader, forward_split
import numpy as np
from torch.optim import Adam


batch_size = 16
enc_len = 36
dec_len = 12
series = np.sin(np.arange(0, 1000))
series = series.reshape(1, 1, -1)
train_idx, valid_idx = forward_split(np.arange(series.shape[2]), enc_len=14, valid_size=200)

train_dl = create_seq2seq_data_loader(series, enc_len=14, dec_len=7, time_idx=train_idx,
                                      batch_size=12, sampling_rate=1., seq_last=True)
valid_dl = create_seq2seq_data_loader(series, enc_len=14, dec_len=7, time_idx=valid_idx,
                                      batch_size=12, sampling_rate=1., seq_last=True)
model = Wave2Wave(1, debug=False, num_layers=5, num_blocks=1)
model.cuda()
opt = Adam(model.parameters(), 0.001)
learner = Learner(model, opt, ".")
learner.fit(100, train_dl, valid_dl, early_stopping=False)
