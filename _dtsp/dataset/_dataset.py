# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/7 14:27
"""
import abc
from keras.utils import Sequence
import numpy as np
from .utils import walk_forward_split


class BaseSeq2SeqDataSet(abc.ABC, Sequence):

    def __init__(self, seqs, encode_lens, decode_lens, batch_size, shuffle=True):
        """Simple Seq2Seq DataSet for Keras.

        Args:
            seqs (numpy.Array): 1D or 2D target series
            encode_lens (int): num of encoder step
            decode_lens (int): num of decoder step
        """
        if seqs.ndim == 1:
            self.target_dim = 1
            self.seqs = seqs.reshape(-1, 1).astype("float32")
        else:
            self.target_dim = seqs.shape[1]
            self.seqs = seqs.astype("float32")
        self.encode_lens = encode_lens
        self.decode_lens = decode_lens
        self.batch_size = batch_size
        self.idxes = np.arange(len(self.seqs) - self.encode_lens - self.decode_lens + 1)
        self.shuffle = shuffle
        self.shuffle_idxes()

    def __len__(self):
        return int(np.ceil((len(self.seqs) - self.encode_lens - self.decode_lens + 1) / self.batch_size))

    def shuffle_idxes(self):
        if self.shuffle:
            np.random.shuffle(self.idxes)

    def on_epoch_end(self):
        self.shuffle_idxes()


class SimpleSeq2SeqDataSet(BaseSeq2SeqDataSet):

    def _get(self, item):
        enc_input = self.seqs[item: item + self.encode_lens]
        dec_input = self.seqs[item + self.encode_lens - 1: item + self.encode_lens + self.decode_lens - 1]
        dec_output = self.seqs[item + self.encode_lens: item + self.encode_lens + self.decode_lens]
        return enc_input, dec_input, dec_output

    def __getitem__(self, item):
        idxes = self.idxes[item * self.batch_size: (item + 1) * self.batch_size]
        enc_input, dec_input, dec_output = zip(*[self._get(i) for i in idxes])
        return (
            {'enc_input': np.stack(enc_input), 'dec_input': np.stack(dec_input)},
            np.stack(dec_output))

    def get(self, item):
        enc_input, dec_input, dec_output = self._get(item)
        enc_inputs, dec_inputs, dec_outputs = (enc_input.reshape(1, -1, self.target_dim),
                                               dec_input.reshape(1, -1, self.target_dim),
                                               dec_output.reshape(1, -1, self.target_dim))
        return ({'enc_input': enc_inputs, 'dec_input': dec_inputs},
                dec_outputs)


class SimpleWaveNetDataSet(BaseSeq2SeqDataSet):

    def _get(self, item):
        inputs = self.seqs[item: item + self.encode_lens + self.decode_lens - 1]
        outputs = self.seqs[item + self.encode_lens: item + self.encode_lens + self.decode_lens]
        return inputs, outputs

    def __getitem__(self, item):
        idxes = self.idxes[item * self.batch_size: (item + 1) * self.batch_size]
        inputs, outputs = zip(*[self._get(i) for i in idxes])
        return np.stack(inputs), np.stack(outputs)

    def get(self, item):
        inputs = self.seqs[item: item + self.encode_lens]
        outputs = self.seqs[item + self.encode_lens: item + self.encode_lens + self.decode_lens]
        return inputs.reshape(1, -1, self.target_dim), outputs.reshape(1, -1, self.target_dim)


def create_simple_seq2seq_dataset(series, encode_lens, decode_lens, n_test, batch_size):
    train_idxes, valid_idxes = walk_forward_split(np.arange(len(series)), encode_lens, decode_lens, n_test)
    train_set = SimpleSeq2SeqDataSet(series[train_idxes], encode_lens, decode_lens, batch_size)
    valid_set = SimpleSeq2SeqDataSet(series[valid_idxes], encode_lens, decode_lens, batch_size)
    return train_set, valid_set


def create_simple_wavenet_dataset(series, encode_lens, decode_lens, n_test, batch_size):
    train_idxes, valid_idxes = walk_forward_split(np.arange(len(series)), encode_lens, decode_lens, n_test)
    train_set = SimpleWaveNetDataSet(series[train_idxes], encode_lens, decode_lens, batch_size)
    valid_set = SimpleWaveNetDataSet(series[valid_idxes], encode_lens, decode_lens, batch_size)
    return train_set, valid_set
