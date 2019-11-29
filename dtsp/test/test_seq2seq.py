# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/29 10:37
"""
from dtsp.dataset import example_data, Seq2SeqDataSet, walk_forward_split
from dtsp.models import Seq2Seq
from torch.utils.data import Subset, DataLoader
from pathlib import Path
import shutil


def test_seq2seq():
    hp = {
        'path': Path('.').resolve() / 'logs',
        'target_size': 20,
        'rnn_type': 'LSTM',
        'dropout': 0.1,
        'hidden_size': 36,
        'teacher_forcing_rate': 0.5,
        'use_attn': True,
        'trans_hidden_size': 4,
        'trans_continuous_var': None,
        'trans_category_var': [(13, 2)],
        'trans_bidirectional': True,
        'trans_rnn_type': 'LSTM',
        'loss_fn': 'MSELoss',
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'lr_scheduler': 'CosineAnnealingWarmRestarts',
        'lr_scheduler_kw': {'T_0': 5, 'T_mult': 10},
    }

    n_test = 12
    n_val = 12
    enc_lens = 36
    dec_lens = 12
    batch_size = 8
    epochs = 50
    data = example_data()
    series = data['series']
    categorical = data['categorical_var']

    mu = series[:-(n_test+n_val)].mean(axis=0)
    std = series[:-(n_test + n_val)].std(axis=0)
    series = (series - mu) / std

    dataset = Seq2SeqDataSet(series, enc_lens, dec_lens, categorical_var=categorical)
    idxes = list(range(len(dataset)))
    train_idxes, _idxes = walk_forward_split(idxes, enc_lens, dec_lens, test_size=n_test + n_val)
    valid_idxes, test_idxes = walk_forward_split(_idxes, enc_lens, dec_lens, test_size=n_test)

    trn_set = Subset(dataset, train_idxes)
    val_set = Subset(dataset, valid_idxes)
    test_set = Subset(dataset, test_idxes)
    trn_ld = DataLoader(trn_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_ld = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
    test_ld = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = Seq2Seq(hp)
    model.fit(epochs, trn_ld, val_ld, early_stopping=10, save_every_n_epochs=None, save_best_model=True)
    model.reload(model.best_model_path())
    print(' - ' * 20)
    print(f'train loss: {model.evaluate_cycle(trn_ld):.3f}, valid loss: {model.evaluate_cycle(val_ld):.3f}, test loss :{model.evaluate_cycle(test_ld):.3f}')
    shutil.rmtree(hp['path'])


def test_seq2seq_without_trans():
    hp = {
        'path': Path('.').resolve() / 'logs',
        'target_size': 20,
        'rnn_type': 'LSTM',
        'use_attn': False,
        'dropout': 0.2,
        'hidden_size': 36,
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
        'lr_scheduler_kw': {'T_0': 5, 'T_mult': 10},
    }

    n_test = 12
    n_val = 12
    enc_lens = 36
    dec_lens = 12
    batch_size = 8
    epochs = 50
    data = example_data()
    series = data['series']

    mu = series[:-(n_test+n_val)].mean(axis=0)
    std = series[:-(n_test + n_val)].std(axis=0)
    series = (series - mu) / std

    dataset = Seq2SeqDataSet(series, enc_lens, dec_lens)
    idxes = list(range(len(dataset)))
    train_idxes, _idxes = walk_forward_split(idxes, enc_lens, dec_lens, test_size=n_test + n_val)
    valid_idxes, test_idxes = walk_forward_split(_idxes, enc_lens, dec_lens, test_size=n_test)

    trn_set = Subset(dataset, train_idxes)
    val_set = Subset(dataset, valid_idxes)
    test_set = Subset(dataset, test_idxes)
    trn_ld = DataLoader(trn_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_ld = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
    test_ld = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = Seq2Seq(hp)
    model.fit(epochs, trn_ld, val_ld, early_stopping=10, save_every_n_epochs=None, save_best_model=True)
    model.reload(model.best_model_path())
    print(model)
    print(' - ' * 20)
    print(f'train loss: {model.evaluate_cycle(trn_ld):.3f}, valid loss: {model.evaluate_cycle(val_ld):.3f}, test loss :{model.evaluate_cycle(test_ld):.3f}')
    shutil.rmtree(hp['path'])
