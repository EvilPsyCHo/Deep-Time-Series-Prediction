# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/6/1 16:37
"""
from deepseries.data import create_seq2seq_data_loader
import numpy as np

enc_len = 12
dec_len = 8
series = np.random.rand(1000, 8, 100)

data_loader = create_seq2seq_data_loader(series, enc_len, dec_len, time_idx=np.arange(series.shape[2]), batch_size=32,
                                         seq_last=True)

for i in data_loader:
    pass

i[0]['enc_x'].shape
i[1].shape