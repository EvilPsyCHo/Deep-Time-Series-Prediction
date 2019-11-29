# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/18 16:30
"""
from torch.utils.data import Dataset
import numpy as np


class _BaseSimpleDataSet(Dataset):

    def __init__(self, seqs, enc_lens, dec_lens):
        if seqs.ndim == 1:
            self.target_dim = 1
            self.seqs = seqs.reshape(-1, 1).astype("float32")
        else:
            self.target_dim = seqs.shape[1]
            self.seqs = seqs.astype("float32")
        self.enc_lens = enc_lens
        self.dec_lens = dec_lens
        self.idxes = np.arange(len(self))

    def __len__(self):
        return len(self.seqs) - self.enc_lens - self.dec_lens + 1


class SimpleSeq2SeqDataSet(_BaseSimpleDataSet):

    def __getitem__(self, item):
        enc_seqs = self.seqs[item: item + self.enc_lens]
        dec_input_seqs = self.seqs[item + self.enc_lens - 1: item + self.enc_lens + self.dec_lens - 1]
        dec_output_seqs = self.seqs[item + self.enc_lens: item + self.enc_lens + self.dec_lens]
        return {'enc_inputs': enc_seqs, 'dec_inputs': dec_input_seqs, 'dec_outputs': dec_output_seqs}


class SimpleWaveNetDataSet(_BaseSimpleDataSet):

    def __getitem__(self, item):
        enc_seqs = self.seqs[item: item + self.enc_lens].transpose(1, 0)
        dec_output_seqs = self.seqs[item + self.enc_lens: item + self.enc_lens + self.dec_lens].transpose(1, 0)
        return {'enc_inputs': enc_seqs, 'dec_outputs': dec_output_seqs}


class Seq2SeqDataSet(Dataset):

    def __init__(self, seqs, enc_lens, dec_lens, continuous_var=None, categorical_var=None):
        self.seqs = seqs
        if seqs.ndim == 1:
            self.target_dim = 1
            self.seqs = seqs.reshape(-1, 1).astype("float32")
        else:
            self.target_dim = seqs.shape[1]
            self.seqs = seqs.astype("float32")

        self.enc_lens = enc_lens
        self.dec_lens = dec_lens
        self.cont = continuous_var

        if self.cont is not None:
            if self.cont.ndim == 1:
                self.cont = self.cont.reshape(-1, 1).astype("float32")
            else:
                self.cont = self.cont.astype("float32")

        self.cate = categorical_var

        if self.cate is not None:
            if self.cate.ndim == 1:
                self.cate = self.cate.reshape(-1, 1).astype("int64")
            else:
                self.cate = self.cate.astype("int64")

        self.idxes = np.arange(len(self))

    def __len__(self):
        return len(self.seqs) - self.enc_lens - self.dec_lens + 1

    def __getitem__(self, item):
        enc_seqs = self.seqs[item: item + self.enc_lens]
        dec_input_seqs = self.seqs[item + self.enc_lens - 1: item + self.enc_lens + self.dec_lens - 1]
        dec_output_seqs = self.seqs[item + self.enc_lens: item + self.enc_lens + self.dec_lens]
        result = {'enc_inputs': enc_seqs, 'dec_inputs': dec_input_seqs, 'dec_outputs': dec_output_seqs}

        if self.cont is not None:
            result['continuous_x'] = self.cont[item: item + self.enc_lens + self.dec_lens]
        if self.cate is not None:
            result['category_x'] = self.cate[item: item + self.enc_lens + self.dec_lens]
        return result
