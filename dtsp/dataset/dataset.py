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
        dec_input_seqs = self.seqs[item + self.enc_lens - 1: item + self.enc_lens + self.dec_lens - 1].transpose(1, 0)
        dec_output_seqs = self.seqs[item + self.enc_lens: item + self.enc_lens + self.dec_lens].transpose(1, 0)
        return {'enc_inputs': enc_seqs, 'dec_inputs': dec_input_seqs, 'dec_outputs': dec_output_seqs}
