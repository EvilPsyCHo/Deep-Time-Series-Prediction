# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/13 15:17
"""
from bentoml.artifact import PytorchModelArtifact
import os
import gc
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import scipy as sp
from torch.optim import Adam
from deepseries.model.rnn2rnn import RNN2RNN
from deepseries.train import Learner
from deepseries.dataset import TimeSeries, Property, Seq2SeqDataLoader
from deepseries.nn.loss import RMSELoss, MSELoss
from deepseries.optim import ReduceCosineAnnealingLR
from deepseries.functional import get_valid_start_end
warnings.filterwarnings("ignore")

DIR = "./data"
N_ROWS = 100
BATCH_SIZE = 32
LAGS = [365, 182, 90, 28]
MAX_LAGS = max(LAGS)
DROP_BEFORE = 1000

SEQ_LAST = False
ENC_LEN = 56
DEC_LEN = 28

VALID_LEN = 28
TEST_LEN = 28

TRAIN_LAST_DAY = 1913
USE_SERIES_LEN = TRAIN_LAST_DAY - DROP_BEFORE + 1 + 28

CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "int16", 'snap_TX': 'int16', 'snap_WI': 'int16' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }

# FIRST_DAY = datetime(2011, 1, 29)
# FORECAST_DAY = datetime(2016,4, 25)


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

    numcols = [f"d_{day}" for day in range(DROP_BEFORE, TRAIN_LAST_DAY + 1)]
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
# series
series = df_series.values
price = df_price.values

# series state
series_nan = np.isnan(series).astype("int8")
series_zero = (series == 0).astype("int8")
start, end = get_valid_start_end(np.bitwise_or(series_nan, series_zero))

# series statistics

series_valid_masked = np.ma.masked_array(series, mask=series_nan.astype(bool))

series_mean = series_valid_masked.mean(axis=1).data
series_std = series_valid_masked.std(axis=1).data
series_skew = sp.stats.mstats.skew(series_valid_masked, axis=1).data
series_kurt = np.clip(sp.stats.mstats.kurtosis(series_valid_masked, axis=1).data, None, 10)

# series normalization

series = np.nan_to_num(
    (series - np.expand_dims(series_mean, 1)) / (np.expand_dims(series_std, 1) + 1e-7), False).astype("float32")


# lag
from deepseries.functional import make_lags, batch_autocorr, get_valid_start_end
series_lags = []
for l in LAGS:
    series_lags.append(make_lags(series, l, use_smooth=True if l > 100 else False))


series_lags = np.concatenate(series_lags, 1).transpose([0, 2, 1])
series_lags = np.nan_to_num(series_lags)
corr = batch_autocorr(series, LAGS, start, end, 1.05)


# series statistic features

series_mean_mean = series_mean.mean()
series_mean_std = series_mean.std()
series_std_mean = series_std.mean()
series_std_std = series_std.std()
series_skew_mean = series_skew.mean()
series_skew_std = series_skew.std()
series_kurt_mean = series_kurt.mean()
series_kurt_std = series_kurt.std()

xy_series_mean = (series_mean - series_mean_mean) / series_mean_std
xy_series_std = (series_std - series_std_mean) / series_std_std
xy_series_skew = (series_skew - series_skew_mean) / series_skew_std
xy_series_kurt = (series_kurt - series_kurt_mean) / series_kurt_std


xy_series_stats = np.stack([xy_series_mean, xy_series_std, xy_series_skew, xy_series_kurt], axis=1)
x_series_label = np.stack([series_nan, series_zero], axis=1)
series = np.expand_dims(series, axis=1)


# series categorical features

xy_series_cat = df_product.set_index("id").values.astype("int64")

# calendar feature

def periodic_feature(x, T):
    psin = np.sin(x * np.pi * 2 / T)
    pcos = np.cos(x * np.pi * 2 / T)
    return np.stack([psin, pcos], axis=0)

xy_calendar_embed = np.concatenate([
    periodic_feature(df_calendar.T['wday'].values, 7),
    periodic_feature(df_calendar.T['month'].values, 12),
    periodic_feature(df_calendar.T['quarter'].values, 4),
    df_calendar.T[['snap_CA', 'snap_TX', 'snap_WI']].values.T
])
xy_calendar_embed = np.expand_dims(xy_calendar_embed, 0)

xy_event_type = np.expand_dims(
    df_calendar.T[['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']].T.values, 0)


class ForwardSpliter:

    def split(self, time_idx, enc_len, dec_len, valid_size):
        if valid_size < 1:
            valid_size = int(np.floor(len(time_idx) * valid_size))
        valid_idx = time_idx[-(valid_size + enc_len):]
        train_idx = time_idx[:-valid_size]
        return train_idx, valid_idx

series = series[:, :, MAX_LAGS:]
xy_event_type = xy_event_type[:, :, MAX_LAGS:]
xy_calendar_embed = xy_calendar_embed[:, :, MAX_LAGS:]
x_series_label = x_series_label[:, :, MAX_LAGS:]
series_lags = series_lags[:, MAX_LAGS:]


spliter = ForwardSpliter()

train_idx, valid_idx = spliter.split(np.arange(series.shape[2]), ENC_LEN, DEC_LEN, VALID_LEN + TEST_LEN)
valid_idx, test_idx = spliter.split(valid_idx, ENC_LEN, DEC_LEN, TEST_LEN)

train_series = TimeSeries(series[:, :, train_idx].transpose(0, 2, 1))
valid_series = TimeSeries(series[:, :, valid_idx].transpose(0, 2, 1))
test_series = TimeSeries(series[:, :, test_idx].transpose(0, 2, 1))

train_xy_series_stats = Property(xy_series_stats)
valid_xy_series_stats = Property(xy_series_stats)
test_xy_series_stats = Property(xy_series_stats)

train_x_series_label = TimeSeries(x_series_label[: ,: ,train_idx].transpose(0, 2, 1))
valid_x_series_label = TimeSeries(x_series_label[:, :, valid_idx].transpose(0, 2, 1))
test_x_series_label = TimeSeries(x_series_label[:, :, test_idx].transpose(0, 2, 1))

train_xy_cat = Property(xy_series_cat)
valid_xy_cat = Property(xy_series_cat)
test_xy_cat = Property(xy_series_cat)


train_series_lags = TimeSeries(series_lags[:, train_idx])
valid_series_lags = TimeSeries(series_lags[:, valid_idx])
test_series_lags = TimeSeries(series_lags[:, test_idx])

train_series_corr = Property(corr)
valid_series_corr = Property(corr)
test_series_corr = Property(corr)

train_xy_event = TimeSeries(xy_event_type[:, :, train_idx].transpose(0, 2, 1), idx_map=dict(zip(np.arange(N_ROWS), [0] * N_ROWS)))
valid_xy_event = TimeSeries(xy_event_type[:, :, valid_idx].transpose(0, 2, 1), idx_map=dict(zip(np.arange(N_ROWS), [0] * N_ROWS)))
test_xy_event = TimeSeries(xy_event_type[:, :, test_idx].transpose(0, 2, 1), idx_map=dict(zip(np.arange(N_ROWS), [0] * N_ROWS)))

train_xy_calendar_embed = TimeSeries(xy_calendar_embed[:, :, train_idx].transpose(0, 2, 1), idx_map=dict(zip(np.arange(N_ROWS), [0] * N_ROWS)))
valid_xy_calendar_embed = TimeSeries(xy_calendar_embed[:, :, valid_idx].transpose(0, 2, 1), idx_map=dict(zip(np.arange(N_ROWS), [0] * N_ROWS)))
test_xy_calendar_embed = TimeSeries(xy_calendar_embed[:, :, test_idx].transpose(0, 2, 1), idx_map=dict(zip(np.arange(N_ROWS), [0] * N_ROWS)))


train_dl = Seq2SeqDataLoader(train_series,
                             batch_size=BATCH_SIZE,
                             enc_lens=ENC_LEN,
                             dec_lens=DEC_LEN,
                             use_cuda=True,
                             mode='train',
                             time_free_space=10,
                             enc_num_feats=[train_series_lags, train_series_corr, train_xy_series_stats, train_x_series_label, train_xy_calendar_embed],
                             enc_cat_feats=[train_xy_cat, train_xy_event],
                             dec_num_feats=[train_series_lags, train_series_corr, train_xy_series_stats, train_xy_calendar_embed],
                             dec_cat_feats=[train_xy_cat, train_xy_event],
                             seq_last=False,
                            )

valid_dl = Seq2SeqDataLoader(valid_series,
                             batch_size=BATCH_SIZE,
                             enc_lens=ENC_LEN,
                             dec_lens=DEC_LEN,
                             use_cuda=True,
                             mode='valid',
                             time_free_space=0,
                             enc_num_feats=[valid_series_lags, valid_series_corr, valid_xy_series_stats, valid_x_series_label, valid_xy_calendar_embed],
                             enc_cat_feats=[valid_xy_cat, valid_xy_event],
                             dec_num_feats=[valid_series_lags, valid_series_corr, valid_xy_series_stats, valid_xy_calendar_embed],
                             dec_cat_feats=[valid_xy_cat, valid_xy_event],
                             seq_last=False
                              )

test_dl = Seq2SeqDataLoader(test_series,
                            batch_size=BATCH_SIZE,
                            enc_lens=ENC_LEN,
                            dec_lens=DEC_LEN,
                            use_cuda=True,
                            mode='test',
                            time_free_space=0,
                            enc_num_feats=[test_series_lags, test_series_corr, test_xy_series_stats, test_x_series_label, test_xy_calendar_embed],
                            enc_cat_feats=[test_xy_cat, test_xy_event],
                            dec_num_feats=[test_series_lags, test_series_corr, test_xy_series_stats, test_xy_calendar_embed],
                            dec_cat_feats=[test_xy_cat, test_xy_event],
                            seq_last=False,
                            )


model = RNN2RNN(series_size=1, hidden_size=256, compress_size=64,
                enc_num_size=23,                enc_cat_size=[(3049, 16), (7, 1), (10, 1), (3, 1), (3, 1), (32, 4), (5, 1), (5, 1), (3, 1)],
                dec_num_size=21, attn_heads=4, attn_size=32, residual=False,
                dec_cat_size=[(3049, 16), (7, 1), (10, 1), (3, 1), (3, 1), (32, 4), (5, 1), (5, 1), (3, 1)],
                dropout=0.1, num_layers=1, rnn_type="GRU")
opt = Adam(model.parameters(), 0.001)
loss_fn = MSELoss()
lr_scheduler = ReduceCosineAnnealingLR(opt, 64, eta_min=1e-4, gamma=0.998)
model.cuda()
learner = Learner(model, opt, './m5_rnn', lr_scheduler=lr_scheduler, verbose=5000)
learner.fit(1000, train_dl, valid_dl, patient=64, start_save=-1, early_stopping=True)
learner.load(174)
learner.model.eval()

def predict_submission(model, test_dl):
    model.eval()
    test_dl.test()
    preds = []
    for batch in test_dl:
        batch.pop('dec_x')
        preds.append(model.predict(**batch)[0].cpu().detach().numpy())
    preds = np.concatenate(preds, axis=0).squeeze()
    return preds

preds = predict_submission(learner.model, test_dl)
yhat = preds * np.expand_dims(series_std, 1) + np.expand_dims(series_mean, 1)
yhat.mean(1)[:20]
top1 = pd.read_csv("./data/submission_top1.csv").set_index("id").loc[df_series.index]
top1.mean(1)[:20].values

yhat.mean()
top1.mean().mean()
def plot(idx):
    plt.figure(figsize=(16, 5))
    plt.plot(yhat[idx], label='wave')
    plt.plot(top1.iloc[idx].values, label='lgb')
    plt.legend()

plot(32)
