# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/7 14:14
"""


def walk_forward_split(series_idxes, encode_lens, decode_lens, n_test):
    train_index = series_idxes[: -n_test]
    valid_index = series_idxes[-(decode_lens + n_test - 1 + encode_lens):]
    return train_index, valid_index
