# encoding: utf-8
# Author: 周知瑞
# Mail: evilpsycho42@gmail.com
import typing
import numpy as np
from scipy.ndimage import convolve1d, convolve


def normalize(x, axis=-1, fillna=True, mask=None):
    if mask is None:
        mask = np.isnan(x)
    else:
        mask = np.bitwise_or(np.isnan(x), mask)
    x = np.ma.MaskedArray(x, mask=mask)
    mu = np.ma.mean(x, axis=axis, keepdims=True)
    std = np.ma.std(x, axis=axis, keepdims=True)
    x_norm = (x - mu) / (std + 1e-6)
    if fillna:
        x_norm = np.nan_to_num(x_norm)
    return x_norm.data, mu.data, std.data


def lag(x, n, smooth=False):
    """

    Args:
        x (ndarray): batch (series x features x seq )
        n (int): n lag
        smooth (bool):

    Returns:
        x_lag
    """
    # TODO: smooth
    if not smooth or (n < 3):
        res = np.zeros_like(x)
        res[:, :, n:] = x[:, :, :-n]
        res[:, :, :n] = np.nan
    else:
        left = lag(x, n - 1, smooth=False)
        mid = lag(x, n, smooth=False)
        right = lag(x, n + 1, smooth=False)
        res = left * 0.25 + mid * 0.5 + right * 0.25
    return res


# TODO: custom conv function, https://stackoverflow.com/questions/47441952/3d-convolution-in-python
# def smooth(x, window=3, rate=0.5, mode="center"):
#     """
#
#     Args:
#         x (ndarray): batch(series, features, seq)
#         window (int):
#         rate (float):
#         mode (str): center or causal
#
#     Returns:
#
#     """
#     assert isinstance(x, np.ndarray)
#     assert x.ndim == 3, "input x ndarray: batch(series, features, seq) "
#     assert window // 2 != 0
#     weight = [rate ** (window // 2 - i) if i <= window // 2 else rate ** (i - window // 2) for i in
#               range(window)]
#     weight = np.array(weight) / np.sum(weight)
#     print(weight)
#     if mode == "center":
#         pad_x = np.pad(x, [(0, 0), (0, 0), (int((window - 1) / 2), int((window - 1) / 2))], 'edge')
#     elif mode == "causal":
#         pad_x = np.pad(x, [(0, 0), (0, 0), (int(window - 1), 0)], 'edge')
#     else:
#         raise ValueError("Only support center or causal mode.")
#     return convolve1d(pad_x, weight, axis=2, mode="nearest")


# TODO numba jit speed


def single_autocorr(series, lag):
    """
    Autocorrelation for single data series
    :param series: traffic series
    :param lag: lag, days
    :return:
    """
    s1 = series[lag:]
    s2 = series[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0


def batch_autocorr(data, lag, starts, ends, threshold, backoffset=0):
    """
    Calculate autocorrelation for batch (many time series at once)
    :param data: Time series, shape [n_pages, n_days]
    :param lag: Autocorrelation lag
    :param starts: Start index for each series
    :param ends: End index for each series
    :param threshold: Minimum support (ratio of time series length to lag) to calculate meaningful autocorrelation.
    :param backoffset: Offset from the series end, days.
    :return: autocorrelation, shape [n_series]. If series is too short (support less than threshold),
    autocorrelation value is NaN
    """
    n_series = data.shape[0]
    n_days = data.shape[1]
    max_end = n_days - backoffset
    corr = np.empty(n_series, dtype=np.float64)
    support = np.empty(n_series, dtype=np.float64)
    for i in range(n_series):
        series = data[i]
        end = min(ends[i], max_end)
        real_len = end - starts[i]
        support[i] = real_len/lag
        if support[i] > threshold:
            series = series[starts[i]:end]
            c_365 = single_autocorr(series, lag)
            c_364 = single_autocorr(series, lag-1)
            c_366 = single_autocorr(series, lag+1)
            # Average value between exact lag and two nearest neighborhs for smoothness
            corr[i] = 0.5 * c_365 + 0.25 * c_364 + 0.25 * c_366
        else:
            corr[i] = np.NaN
    return corr #, support


def get_valid_start_end(mask):
    """
    Args:
        mask (ndarray of bool): invalid mask
    Returns:
    """
    ns = mask.shape[0]
    nt = mask.shape[1]
    start_idx = np.full(ns, -1, dtype=np.int32)
    end_idx = np.full(ns, -1, dtype=np.int32)

    for s in range(ns):
        # scan from start to the end
        for t in range(nt):
            if not mask[s][t]:
                start_idx[s] = t
                break
        # reverse scan, from end to start
        for t in range(nt - 1, -1, -1):
            if not mask[s][t]:
                end_idx[s] = t + 1
                break
    return start_idx, end_idx


# def get_trend(x, max_T, use_smooth=True, smooth_windows=5, smooth_ration=0.5):
#     if use_smooth:
#         x = smooth(x, smooth_windows, smooth_ration)
#     lag = make_lags(x, max_T, use_smooth).squeeze()
#     return np.where(lag == 0, 0, x / lag)


def forward_split(time_idx, enc_len, valid_size):
    if valid_size < 1:
        valid_size = int(np.floor(len(time_idx) * valid_size))
    valid_idx = time_idx[-(valid_size + enc_len):]
    train_idx = time_idx[:-valid_size]
    return train_idx, valid_idx
