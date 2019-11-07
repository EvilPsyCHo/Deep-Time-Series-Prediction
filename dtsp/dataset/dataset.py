# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/7 14:27
"""
from keras.utils import Sequence
import numpy as np
from .utils import walk_forward_split


class SimpleSeq2SeqDataSet(Sequence):

    def __init__(self, seqs, encode_lens, decode_lens):
        """Simple Seq2Seq DataSet for Keras.

        Args:
            seqs (numpy.Array): 1D or 2D target series, dim 1 index means the time step
            encode_lens (int): num of encoder step
            decode_lens (int): num of decoder step
        """
        self.ndim = seqs.ndim
        self.target_dim = 1 if self.ndim == 1 else self.seqs.shape[1]
        self.seqs = seqs
        self.encode_lens = encode_lens
        self.decode_lens = decode_lens

    def __len__(self):
        return len(self.seqs) - self.encode_lens - self.decode_lens + 1

    def __getitem__(self, item):
        enc_inputs = self.seqs[item: item + self.encode_lens]
        dec_inputs = self.seqs[item + self.encode_lens - 1: item + self.encode_lens + self.decode_lens - 1]
        dec_outputs = self.seqs[item + self.encode_lens: item + self.encode_lens + self.decode_lens]
        enc_inputs, dec_inputs, dec_outputs = (enc_inputs.reshape(1, -1, self.target_dim),
                                               dec_inputs.reshape(1, -1, self.target_dim),
                                               dec_outputs.reshape(1, -1, self.target_dim))
        return (
            {'enc_input': enc_inputs, 'dec_input': dec_inputs},
            dec_outputs)


def create_simple_seq2seq_dataset(series, encode_lens, decode_lens, n_test):
    train_idxes, valid_idxes = walk_forward_split(np.arange(len(series)), encode_lens, decode_lens, n_test)
    train_set = SimpleSeq2SeqDataSet(series[train_idxes], encode_lens, decode_lens)
    valid_set = SimpleSeq2SeqDataSet(series[valid_idxes], encode_lens, decode_lens)
    return train_set, valid_set
