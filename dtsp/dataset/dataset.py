# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/18 16:30
"""
from torch.utils.data import DataLoader, Dataset
import numpy as np


class SimpleDataSet(Dataset):

    def __init__(self, seqs, enc_lens, dec_lens):
        if seqs.ndim == 1:
            self.target_dim = 1
            self.seqs = seqs.reshape(-1, 1).astype("float32")
        else:
            self.target_dim = seqs.shape[1]
            self.seqs = seqs.astype("float32")
        self.enc_lens = enc_lens
        self.dec_lens = dec_lens
        self.idxes = np.arange(len(self.seqs) - self.enc_lens - self.dec_lens + 1)

    def __len__(self):
        return len(self.seqs) - self.enc_lens - self.dec_lens + 1

    def __getitem__(self, item):
        enc_seqs = self.seqs[item: item + self.enc_lens]
        dec_input_seqs = self.seqs[item + self.enc_lens - 1: item + self.enc_lens + self.dec_lens - 1]
        dec_output_seqs = self.seqs[item + self.enc_lens: item + self.enc_lens + self.dec_lens]
        return enc_seqs, dec_input_seqs, dec_output_seqs
