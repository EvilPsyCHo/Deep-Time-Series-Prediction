# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/20 10:51
"""
import torch.nn as nn
import torch
from deepseries.nn.cnn import WaveNet
from deepseries.nn.comm import Embeddings, Concat
from deepseries.nn.loss import RMSE
from deepseries.log import get_logger

logger = get_logger(__name__)


class Wave2Wave(nn.Module):

    def __init__(self, target_size, enc_cat_size=None, enc_num_size=None, dec_cat_size=None, dec_num_size=None,
                 residual_channels=32, share_embeds=False, skip_channels=32, num_blocks=3, num_layers=8,
                 dropout=.0, hidden_channels=128, loss_fn=RMSE(), debug=False, nonlinearity="Tanh"):
        super(Wave2Wave, self).__init__()
        self.debug = debug
        self.enc_embeds = Embeddings(enc_cat_size, seq_last=True)
        if share_embeds:
            self.dec_embeds = self.enc_embeds
        else:
            self.dec_embeds = Embeddings(dec_cat_size, seq_last=True)
        self.concat = Concat(dim=1)
        self.dropout = nn.Dropout(dropout)
        enc_input_channels = (self.enc_embeds.output_size +
                              target_size +
                              (enc_num_size if isinstance(enc_num_size, int) else 0))
        dec_input_channels = (self.dec_embeds.output_size +
                              target_size +
                              (dec_num_size if isinstance(dec_num_size, int) else 0))
        self.encoder = WaveNet(enc_input_channels, residual_channels, skip_channels, num_blocks, num_layers)
        self.decoder = WaveNet(dec_input_channels, residual_channels, skip_channels, num_blocks, num_layers)
        self.conv_output1 = nn.Conv1d(skip_channels, hidden_channels, kernel_size=1)
        self.conv_output2 = nn.Conv1d(hidden_channels, target_size, kernel_size=1)
        self.nonlinearity = getattr(nn, nonlinearity)()
        self.loss_fn = loss_fn

    def encode(self, x, num=None, cat=None):
        x = self.concat(x, num, self.enc_embeds(cat))
        x = self.dropout(x)
        _, state = self.encoder.encode(x)
        return state

    def decode(self, x, state, num=None, cat=None):
        x = self.concat(x, num, self.enc_embeds(cat))
        x = self.dropout(x)
        skips, state = self.decoder.decode(x, state)
        output = self.nonlinearity(self.conv_output1(skips))
        output = self.conv_output2(output)
        return output, state

    def batch_loss(self, x, y, w=None):
        state = self.encode(x['enc_x'], x['enc_num'], x['enc_cat'])
        preds = []
        for step in range(x['dec_len']):
            pred, state = self.decode(x['dec_x'][:, :, [step]],
                                      state,
                                      x['dec_num'][:, :, [step]] if x['dec_num'] is not None else None,
                                      x['dec_cat'][:, :, [step]] if x['dec_cat'] is not None else None)
            preds.append(pred)
        preds = torch.cat(preds, dim=2)
        if self.debug:
            message = f"batch loss predict mean: {preds.mean():.3f}, target mean: {y.mean():.3f}"
            logger.info(message)
        loss = self.loss_fn(preds, y, w)
        del state
        return loss

    @torch.no_grad()
    def predict(self, enc_x, dec_len, enc_num=None, enc_cat=None, dec_num=None, dec_cat=None):
        state = self.encode(enc_x, enc_num, enc_cat)
        preds = []
        y = enc_x[:, :, [-1]]
        for step in range(dec_len):
            y, state = self.decode(y, state,
                                   dec_num[:, :, [step]] if dec_num is not None else None,
                                   dec_cat[:, :, [step]] if dec_cat is not None else None)
            preds.append(y)
        del state
        return torch.cat(preds, dim=2)
