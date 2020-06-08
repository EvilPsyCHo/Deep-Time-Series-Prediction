# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/6 10:40
"""
import torch.nn as nn
import torch
from deepseries.nn import *
from deepseries.log import get_logger


logger = get_logger(__name__)


class RNNEncoder(nn.Module):

    """
    Args:
        x:
        num:
        cat:
    Returns:
        compress, outputs, hidden
    """

    def __init__(self, series_size, hidden_size, compress_size, nonlinearity='SELU',
                 dropout=0.1, num_size=None, cat_size=None, rnn_type='GRU', num_layers=1, initializer='xavier'):
        super().__init__()
        rnn_input_size = series_size + (num_size if num_size else 0)
        self.embeds = Embeddings(cat_size, seq_last=False)
        rnn_input_size += self.embeds.output_size
        self.concat = Concat(2)
        self.rnn = getattr(nn, rnn_type)(rnn_input_size, hidden_size, num_layers=num_layers,
                                         dropout=dropout, batch_first=True)
        self.compress = Dense(hidden_size, compress_size, dropout=dropout, nonlinearity=nonlinearity)
        init_rnn(self.rnn, initializer)

    def forward(self, x, num=None, cat=None):
        """
        Args:
            x: B x S x N
            num:
            cat:

        Returns:
            rnn_compress: B x S x C
            rnn_states: ...

        """
        input = self.concat(x, num, self.embeds(cat))
        outputs, hidden = self.rnn(input)
        compress = self.compress(outputs)
        return compress, outputs, hidden


class RNNDecoder(nn.Module):

    def __init__(self, series_size, hidden_size, compress_size, nonlinearity='SELU', residual=True,
                 attn_heads=None, attn_size=None, dropout=0.1, num_size=None, cat_size=None, rnn_type='GRU', n_layers=1):
        """
            https://stackoverflow.com/questions/44238154/what-is-the-difference-between-luong-attention-and-bahdanau-attention
            https://distill.pub/2016/augmented-rnns/
            https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#global-vs-local-attention
            https://towardsdatascience.com/sequence-2-sequence-model-with-attention-mechanism-9e9ca2a613a
        Args:
            series_dim:
            hidden_dim:
            compress_dim:
            activation:
            residual:
            attn_heads:
            attn_size:
            dropout:
            num_feat:
            cat_feat:
            rnn_type:
            n_layers:
            initializer:
        """
        super().__init__()
        self.embeds = Embeddings(cat_size, seq_last=False)
        self.concat = Concat(dim=2)
        rnn_input_size = series_size + self.embeds.output_size + (num_size if num_size else 0)
        self.rnn = getattr(nn, rnn_type)(rnn_input_size, hidden_size, num_layers=n_layers,
                                         dropout=dropout, batch_first=True)
        self.residual = residual
        concat_size = hidden_size
        if attn_heads is not None:
            self.attn = Attention(attn_heads, attn_size, hidden_size, compress_size, compress_size, dropout=dropout)
            concat_size += attn_size
        if residual:
            concat_size += rnn_input_size
        self.output = Dense(concat_size, series_size, dropout=dropout, nonlinearity=nonlinearity)
        init_rnn(self.rnn, 'xavier')

    def forward(self, prev_y, enc_compress, hidden, num=None, cat=None):
        input = self.concat(prev_y, num, self.embeds(cat))
        output, hidden = self.rnn(input, hidden)
        if hasattr(self, 'attn'):
            attn_output, p_attn = self.attn(output, enc_compress, enc_compress)
        else:
            attn_output, p_attn = None, None
        concat = self.concat(
            output,
            attn_output,
            input if self.residual else None
        )
        next_y = self.output(concat)
        return next_y, output, hidden, p_attn


class RNN2RNN(nn.Module):

    def __init__(self, series_size, hidden_size, compress_size, nonlinearity='SELU', residual=True,
                 attn_heads=None, attn_size=None, dropout=0.1, enc_num_size=None, enc_cat_size=None,
                 dec_num_size=None, dec_cat_size=None, rnn_type='GRU', num_layers=1, beta1=1e-6, beta2=1e-6,
                 loss_fn=RMSE(), debug=False):
        super().__init__()
        self.encoder = RNNEncoder(series_size, hidden_size, compress_size, nonlinearity, dropout, enc_num_size,
                                  enc_cat_size, rnn_type, num_layers)
        self.decoder = RNNDecoder(series_size, hidden_size, compress_size, nonlinearity, residual, attn_heads,
                                  attn_size, dropout, dec_num_size, dec_cat_size, rnn_type, num_layers)
        self.loss_fn = loss_fn
        self.loss_stable = RNNStabilityLoss(beta1)
        self.loss_activate = RNNActivationLoss(beta2)
        self.debug = debug

    def batch_loss(self, x, y, w=None):
        enc_len = x['enc_x'].shape[1]
        dec_len = x['dec_len']

        compress, enc_output, enc_hidden = self.encoder(x=x['enc_x'], num=x['enc_num'], cat=x['enc_cat'])
        yhat, dec_output, dec_hidden, p_attn = self.decoder(prev_y=x['dec_x'], num=x['dec_num'], cat=x['dec_cat'],
                                                            enc_compress=compress, hidden=enc_hidden)
        if self.debug:
            logger.info(f"yhat mean: {yhat.mean(): .2f}, target mean: {y.mean():.2f}")
        bl = (self.loss_fn(yhat, y, w) +
              self.loss_stable(enc_output) / enc_len +
              self.loss_stable(dec_output) / dec_len +
              self.loss_activate(enc_output) / enc_len +
              self.loss_activate(dec_output) / dec_len)
        return bl

    def forward(self, enc_x, dec_len, enc_num=None, enc_cat=None, dec_num=None, dec_cat=None):
        compress, enc_output, enc_hidden = self.encoder(enc_x, enc_num, enc_cat)
        hidden = enc_hidden
        prev_y = enc_x[:, [-1]]
        ys = []
        ps = []

        for i in range(dec_len):
            prev_y, _, hidden, p_attn = self.decoder(prev_y, compress, hidden,
                                                     num=None if dec_num is None else dec_num[:, [i]],
                                                     cat=None if dec_cat is None else dec_cat[:, [i]])
            ys.append(prev_y)
            ps.append(p_attn)
        return torch.cat(ys, 1), torch.cat(ps, 2) if hasattr(self.decoder, 'attn') else None

    @torch.no_grad()
    def predict(self, *args, **kw):
        return self(*args, **kw)
