# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/26 14:10
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import deepseries.functional as F
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')


class SeriesAnalysisModel:

    """https://blog.csdn.net/claroja/article/details/70841382"""

    def __init__(self, series, mask=None, mask_zero=True, mask_nan=True):
        self.series = series
        if mask is None:
            self.mask = F.mask_zero_nan(series, mask_zero, mask_nan)
        else:
            self.mask = np.bitwise_or(mask, F.mask_zero_nan(series, mask_zero, mask_nan))
        self.mask_zero = mask_zero
        self.starts, self.ends = F.get_valid_start_end(self.series, self.mask)
        self.valid_lens = self.ends - self.starts
        self.autocorr = None
        self.trend = None
        self.max_T = None

    def get_autocorr(self, n_lags, threshold=1.5,  backoffset=0, use_smooth=False):
        self.autocorr = F.batch_autocorr(self.series, n_lags, self.starts, self.ends, threshold, backoffset, use_smooth)
        return self

    def plot_autocorr(self, idx=None, figsize=(8, 5)):
        corr = self.autocorr
        if self.series.shape[0] == 1:
            idx = 0
        if idx is None:
            f = plt.figure(figsize=figsize)
            gs = f.add_gridspec(4, 1)
            ax1 = plt.subplot(gs[:2])
            ax2 = plt.subplot(gs[2])
            ax3 = plt.subplot(gs[3])

            im = ax1.imshow(corr, aspect="auto", vmin=-1, vmax=1, cmap='coolwarm')
            plt.colorbar(im, extend='both', shrink=0.6, ax=ax1)
            ax1.set_title("series autocorr")

            valid_corr = np.ma.array(corr, mask=np.isnan(corr))
            ax2.plot(np.abs(valid_corr).mean(0))
            ax2.set_title("mean absolute autocorr over time")

            ax3.hist(np.abs(corr).argmax(1), bins=max(1, corr.shape[1] // 5))
            ax3.set_title("time distribution of max absolute autocorr")

            plt.tight_layout()
        else:
            f, ax = plt.subplots(figsize=figsize)
            not_nan_idx = np.where(~np.isnan(corr[idx]))[0]
            ax.plot(not_nan_idx, corr[idx][not_nan_idx])
            ax.set_title("series autocorr")

    def plot_valid(self, figsize=(8, 5)):
        f, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(self.mask, aspect="auto", vmin=0, vmax=1, cmap='gray', alpha=0.8)
        f.colorbar(im, ax=ax, shrink=0.6)
        ax.set_title("series valid value map (zero means valid)")
        plt.tight_layout()
        return im

    def get_trend(self, max_T, use_smooth=True, smooth_windows=5, smooth_ratio=0.5):
        self.trend = F.get_trend(self.series, max_T, use_smooth, smooth_windows, smooth_ratio)
        self.max_T = max_T
        return self

    def plot_trend(self, idx=None, figsize=(16, 5), drop_before=None):
        if self.series.shape[0] == 1:
            idx = 0
        if drop_before is None:
            drop_before = self.max_T
        if idx is not None:
            f, ax = plt.subplots(figsize=figsize)
            not_nan_idx = np.where(~np.isnan(self.trend[idx]))[0]
            ax.plot(not_nan_idx, self.trend[idx][not_nan_idx], alpha=0.8, c='red', label='trend')
            ax1 = ax.twinx()
            ax1.plot(self.series[idx], c='blue', alpha=0.8, label='series')
            f.legend(loc='upper right')
            ax1.set_title(f"series trend")
            plt.tight_layout()
        else:
            f, ax = plt.subplots(ncols=2, figsize=figsize)
            im = ax[0].imshow(np.where(np.isnan(self.trend), 1, self.trend)[:, drop_before:], aspect='auto', cmap='coolwarm', vmin=0.5, vmax=1.5)
            ax[0].set_title(f"series trend drop before {drop_before}")
            plt.colorbar(im, extend='both', shrink=0.6, ax=ax[0])

            pd.DataFrame(self.trend[:, drop_before:]).median(axis=0).plot(ax=ax[1])
            ax[1].set_title(f'trend median over time drop before {drop_before}')
            plt.tight_layout()
