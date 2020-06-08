# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/6/1 14:24
"""
import numpy as np
import torch
import torch.utils.data as torch_data
from deepseries.log import get_logger
import json

logger = get_logger(__name__)
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Value:

    def __init__(self, data, name, enc=True, dec=True, mapping=None):
        assert isinstance(data, np.ndarray)
        assert data.ndim in [2, 3]
        self.is_property = False
        if data.ndim == 2:
            self.is_property = True
            data = np.expand_dims(data, axis=2)
        self.data = data
        self.name = name
        self.enc = enc
        self.dec = dec
        self.is_cat = True if str(self.data.dtype)[:3] == "int" else False
        self.mapping = mapping

    def sub(self, time_idx=None):
        if time_idx is None:
            return self
        if self.is_property:
            return self
        else:
            return Value(self.data[:, :, time_idx], self.name, self.enc, self.dec, self.mapping)

    def read_batch(self, series_idx, time_idx):
        """

        Args:
            series_idx: 1D array
            time_idx: 2D array

        Returns:

        """
        if self.mapping is not None:
            series_idx = np.array([self.mapping.get(i) for i in series_idx])

        if self.is_property:
            if len(series_idx) == 1:
                batch = self.data[[series_idx.item()]].repeat(time_idx.shape[1], axis=2).repeat(time_idx.shape[0], axis=0)
            else:
                batch = self.data[series_idx].repeat(time_idx.shape[1], axis=2)
            return batch

        if len(series_idx) == 1:
            batch = self.data[series_idx.item()][:, time_idx].transpose(1, 0, 2)
        else:
            batch = self.data[series_idx, :, :][:, :, time_idx.squeeze()]
        return batch

class Seq2SeqDataSet(torch_data.Dataset):

    def __init__(self, series, enc_len, dec_len, features=None, weights=None, seq_last=True,
                 device=DEFAULT_DEVICE, mode='train'):
        self.series = series
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.weights = weights
        self.seq_last = seq_last
        self.device = device
        self.mode = mode

        self.num_series = series.data.shape[0]
        self.num_starts = series.data.shape[2] - enc_len - dec_len + 1
        self.features = features
        self.enc_num = self.feature_filter(lambda x: x.enc and not x.is_cat, features)
        self.enc_cat = self.feature_filter(lambda x: x.enc and x.is_cat, features)
        self.dec_num = self.feature_filter(lambda x: x.dec and not x.is_cat, features)
        self.dec_cat = self.feature_filter(lambda x: x.dec and x.is_cat, features)

    @staticmethod
    def feature_filter(func, features):
        if features is None:
            return None
        ret = list(filter(func, features))
        if len(ret) == 0: return None
        return ret

    def __len__(self):

        if self.num_series == 1:
            return self.num_starts
        else:
            return self.num_series

    def read_batch(self, features, series_idx, time_idx):
        if features is None:
            return None
        batch = np.concatenate([f.read_batch(series_idx, time_idx) for f in features], axis=1)
        if not self.seq_last:
            batch = batch.transpose([0, 2, 1])
        return torch.as_tensor(batch, dtype=torch.long if features[0].is_cat else torch.float, device=self.device)

    def __getitem__(self, items):
        """

        Args:
            items: (series idxes: 1D array, time idxes: 2D array)
        Returns:

        """
        enc_idx = np.stack([np.arange(i, i+self.enc_len) for i in items[1]], axis=0)
        dec_idx = np.stack([np.arange(i+self.enc_len, i+self.enc_len+self.dec_len) for i in items[1]], axis=0)
        series_idx = items[0]

        feed_x = {
            "enc_x": self.read_batch([self.series], series_idx, enc_idx),
            "dec_x": self.read_batch([self.series], series_idx, dec_idx - 1),
            "enc_num": self.read_batch(self.enc_num, series_idx, enc_idx),
            "dec_num": self.read_batch(self.dec_num, series_idx, dec_idx),
            "enc_cat": self.read_batch(self.enc_cat, series_idx, enc_idx),
            "dec_cat": self.read_batch(self.dec_cat, series_idx, dec_idx),
            "dec_len": dec_idx.shape[1],
        }
        feed_y = self.read_batch([self.series], series_idx, dec_idx)
        weight = self.read_batch([self.weights], series_idx, dec_idx) if self.weights is not None else None
        return feed_x, feed_y, weight


class Seq2SeqSampler(torch_data.Sampler):

    def __init__(self, data_source, batch_size, sampling_rate=1., random_seed=42):
        self.data_source = data_source
        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.random_seed = np.random.RandomState(random_seed)

    def __iter__(self):
        if self.data_source.num_series == 1:
            starts = np.arange(self.data_source.num_starts)
            self.random_seed.shuffle(starts)
            for i in range(len(self)):
                yield np.array([0]), starts[i*self.batch_size: (i+1)*self.batch_size]
        else:
            idxes = np.arange(self.data_source.num_series)
            starts = np.arange(self.data_source.num_starts)
            self.random_seed.shuffle(idxes)
            for i in range(len(self)):
                start = self.random_seed.choice(starts)
                yield idxes[i * self.batch_size: (i + 1) * self.batch_size], np.array([start])

    def __len__(self):
        n = np.floor(len(self.data_source) * self.sampling_rate)
        if n % self.batch_size == 0:
            return int(n // self.batch_size)
        else:
            return int(n // self.batch_size) + 1


def seq2seq_collate_fn(batch):
    (x, y, weight) = batch[0]
    return x, y, weight


def forward_split(time_idx, enc_len, valid_size):
    if valid_size < 1:
        valid_size = int(np.floor(len(time_idx) * valid_size))
    valid_idx = time_idx[-(valid_size + enc_len):]
    train_idx = time_idx[:-valid_size]
    return train_idx, valid_idx


def create_seq2seq_data_loader(series, enc_len, dec_len, batch_size, time_idx=None, weights=None, sampling_rate=1.,
                               features=None, seq_last=False, device=DEFAULT_DEVICE, mode='train', seed=42,
                               num_workers=0, pin_memory=False):
    series = Value(series, 'series').sub(time_idx)
    weights = None if weights is None else Value(weights, 'weights').sub(time_idx)
    features = None if features is None else [f.sub(time_idx) for f in features]
    data_set = Seq2SeqDataSet(series, enc_len, dec_len, features, weights, seq_last, device, mode)
    sampler = Seq2SeqSampler(data_set, batch_size, sampling_rate, seed)
    data_loader = torch_data.DataLoader(data_set, collate_fn=seq2seq_collate_fn, sampler=sampler,
                                        num_workers=num_workers, pin_memory=pin_memory)
    # logger.info(f"---------- {mode} dataset information ----------")
    # logger.info(json2str(data_loader.dataset.info))
    # proportion = batch_size * num_iterations / len(data_set)
    # logger.info(f"data loader sampling proportion of each epoch: {proportion*100:.1f}%")
    return data_loader


def json2str(data):
    return json.dumps(data, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False)
