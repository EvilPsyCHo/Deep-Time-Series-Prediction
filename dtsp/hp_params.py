# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/19 14:28
"""


class Params(dict):

    def __getattr__(self, item):
        return self.get(item)


BASIC_SEQ2SEQ_HP = Params({
    'target_size': 1,
    'hidden_size': 12,
    'rnn_type': 'LSTM',
    'optimizer': 'Adam',
    'lr': 0.001,
    'train_dataloader': None,
    'loss_fn': 'MSEloss',
    'teacher_forcing_rate': 0.1,
})
