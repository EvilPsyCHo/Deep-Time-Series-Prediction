# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/7 14:14
"""


def walk_forward_split(series_idxes, encode_lens, decode_lens, test_size):
    n = len(series_idxes) - encode_lens - decode_lens + 1
    if test_size < 1:
        test_size = int(test_size * n)
    train_index = series_idxes[: -test_size]
    valid_index = series_idxes[-(decode_lens + test_size - 1 + encode_lens):]
    return train_index, valid_index
