# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/20 15:33
"""
import os
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder
from deepseries.model import Wave2Wave
from deepseries.train import Learner
from deepseries.data import Value, create_seq2seq_data_loader, forward_split
import deepseries.functional as F
from deepseries.nn.loss import RMSE
from torch.optim import Adam
from deepseries.optim import ReduceCosineAnnealingLR


DIR = "./data"
N_ROWS = None
DROP_BEFORE = 100
BATCH_SIZE = 32
LAGS = [365, 182, 90, 28]
MAX_LAGS = max(LAGS)


SEQ_LAST = False
ENC_LEN = 56
DEC_LEN = 28

VALID_LEN = 28
TEST_LEN = 28

TRAIN_LAST_DAY = 1913

CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "int16", 'snap_TX': 'int16', 'snap_WI': 'int16' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }


def load_data():
    label_encoders = {}

    prices = pd.read_csv(os.path.join(DIR, "sell_prices.csv"), dtype=PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            if col not in label_encoders:
                label_encoders[col] = LabelEncoder().fit(prices[col].astype(str).fillna("None"))
            prices[col] = label_encoders[col].transform(prices[col].astype(str).fillna("None")).astype("int16")

    cal = pd.read_csv(os.path.join(DIR, "calendar.csv"), dtype=CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            if col not in label_encoders:
                label_encoders[col] = LabelEncoder().fit(cal[col].astype(str).fillna("None"))
            cal[col] = label_encoders[col].transform(cal[col].astype(str).fillna("None")).astype("int16")

    numcols = [f"d_{day}" for day in range(1, TRAIN_LAST_DAY + 1)]
    catcols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']
    dtype = {numcol: "float32" for numcol in numcols}
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv(os.path.join(DIR, "sales_train_validation.csv"),
                     usecols=catcols + numcols, dtype=dtype, nrows=N_ROWS)

    for col in catcols:
        if col != "id":
            if col not in label_encoders:
                label_encoders[col] = LabelEncoder().fit(dt[col].astype(str).fillna("None"))
            dt[col] = label_encoders[col].transform(dt[col].astype(str).fillna("None")).astype("int16")

    for day in range(TRAIN_LAST_DAY + 1, TRAIN_LAST_DAY + 28 + 1):
        dt[f"d_{day}"] = np.nan

    product = dt[catcols].copy()
    print(f"product shape {product.shape}")

    dt = pd.melt(dt,
                 id_vars=catcols,
                 value_vars=[col for col in dt.columns if col.startswith("d_")],
                 var_name="d",
                 value_name="sales")

    dt = dt.merge(cal[['d', 'wm_yr_wk']], on="d", copy=False)
    dt = dt.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)
    dt['d'] = dt['d'].str.replace('d_', '').astype("int32")
    price = dt.pivot(index="id", columns="d", values="sell_price")
    xy = dt.pivot(index="id", columns="d", values="sales")
    del dt
    gc.collect()
    print(f"sale_xy shape {xy.shape}")
    print(f"price shape {price.shape}")

    cal_use_col = ['date', 'wday', 'month', 'year', 'event_name_1',
                   'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX',
                   'snap_WI']
    cal = cal[cal_use_col]
    cal['year'] = cal['year'] - cal['year'].min()
    cal['quarter'] = cal.date.dt.quarter
    cal = cal.drop("date", axis=1).T
    cal = cal[xy.columns]
    print(f"calendar shape {cal.shape}")

    submission = pd.read_csv(os.path.join(DIR, "sample_submission.csv"))
    print(f"submisson shape {submission.shape}")
    return xy, price, cal, product, submission

df_series, df_price, df_calendar, df_product, df_sub = load_data()

series = df_series.values
price = df_price.values

series_is_nan = np.isnan(series)
series_is_zero = series == 0

starts, ends = F.get_valid_start_end(series_is_nan)
series_lags = F.make_lags(series, LAGS, use_smooth=True)
series_lags_corr = F.batch_autocorr(series, LAGS, starts, ends, threshold=1.05)
series_lags_corr = F.normalize(series_lags_corr, axis=0)[0]
series_lags_corr = Value(series_lags_corr, name='series_lags_corr')

series, series_mean, series_std = F.normalize(series[:, np.newaxis, DROP_BEFORE:], axis=2)
series_lags = F.normalize(series_lags[:, :, DROP_BEFORE:])[0]
series_lags = Value(series_lags, 'xy_lags')

time_idxes = np.arange(series.shape[2])
trn_idx, val_idx = forward_split(time_idxes, ENC_LEN, VALID_LEN+TEST_LEN)
val_idx, test_idx = forward_split(val_idx, ENC_LEN, TEST_LEN)
trn_dl = create_seq2seq_data_loader(series, enc_len=ENC_LEN, dec_len=DEC_LEN, time_idx=trn_idx,
                                    batch_size=BATCH_SIZE,
                                    features=[series_lags, series_lags_corr],
                                    seq_last=True, device='cuda', mode='train', num_workers=0, pin_memory=False)

val_dl = create_seq2seq_data_loader(series, enc_len=ENC_LEN, dec_len=DEC_LEN, time_idx=val_idx,
                                    batch_size=BATCH_SIZE,
                                    features=[series_lags, series_lags_corr],
                                    seq_last=True, device='cuda', mode='valid')

model = Wave2Wave(1, enc_num_size=8, dec_num_size=8, residual_channels=32, skip_channels=32,
                  num_blocks=2, num_layers=6, debug=False)
opt = Adam(model.parameters(), 0.001)
loss_fn = RMSE()
lr_scheduler = ReduceCosineAnnealingLR(opt, 64, eta_min=1e-4, gamma=0.998)
model.cuda()
learner = Learner(model, opt, './m5_rnn', lr_scheduler=lr_scheduler, verbose=10)
learner.fit(10, trn_dl, val_dl, patient=64, start_save=-1, early_stopping=True)
