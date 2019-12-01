# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/25 11:00
"""
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
# support iteration based lr
WaveNet_HP ={
    'path': None,
    'target_size': 20,
    'dilation': [1, 2, 4, 8, 16, 32, 64],
    'dropout': 0.2,
    'residual_channels': 36,
    'teacher_forcing_rate': 0.5,
    'use_move_scale': True,
}


SIMPLE_WAVENET_HP = {
    'path': None,
    'target_size': 1,
    'dilation': None,
    'residual_channels': 12,
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'loss_fn': 'MSELoss',
    'teacher_forcing_rate': 0.5,
    'lr_scheduler': 'CosineAnnealingWarmRestarts',
    'lr_scheduler_kw': {'T_0': 5, 'T_mult': 2},
    'dropout': 0.2,
}

SIMPLE_SEQ2SEQ_HP = {
    'path': None,
    'target_size': 1,
    'hidden_size': 12,
    'rnn_type': 'LSTM',
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'loss_fn': 'MSELoss',
    'teacher_forcing_rate': 0.5,
    'lr_scheduler': 'CosineAnnealingWarmRestarts',
    'lr_scheduler_kw': {'T_0': 5, 'T_mult': 2},
}


Seq2Seq_HP = {
    'target_size': 1,
    'rnn_type': 'LSTM',
    'dropout': 0.2,
    'hidden_size': 12,
    'teacher_forcing_rate': 0.5,
    'trans_hidden_size': None,
    'trans_continuous_var': None,
    'trans_category_var': None,
    'trans_bidirectional': True,
    'trans_rnn_type': 'LSTM',
    'loss_fn': 'MSELoss',
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'lr_scheduler': 'CosineAnnealingWarmRestarts',
    'lr_scheduler_kw': {'T_0': 5, 'T_mult': 2},
}
