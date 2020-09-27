# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/9/27 14:52
"""
import torch
import torch.nn as nn
from deepseries.models.seq2seq.utils import *
from deepseries.models.seq2seq.encoder import RNNEncoder
from deepseries.models.seq2seq.decoder import AttnRNNDecoder
import random


class AttnSeq2Seq(nn.Module):

    def __init__(self, encoder_inputs, decoder_inputs, target_size, decode_length,
                 rnn_type, hidden_size, num_layers, bidirectional, dropout,
                 attn_heads, attn_size, loss_fn=nn.MSELoss(), share_embeddings=None, teacher_forcing_rate=0.5):
        super().__init__()
        self.encoder_inputs = Inputs(encoder_inputs)
        self.decoder_inputs = Inputs(decoder_inputs)
        self.target_size = target_size
        self.decode_length = decode_length
        self.loss_fn = loss_fn
        self.teacher_forcing_rate = teacher_forcing_rate

        if share_embeddings is not None:
            for (enc_name, dec_name) in share_embeddings:
                self.decoder_inputs.categorical_inputs.embeddings[dec_name] =\
                    self.encoder_inputs.categorical_inputs.embeddings[enc_name]

        self.encoder = RNNEncoder(self.encoder_inputs.output_size, rnn_type, hidden_size,
                                  bidirectional, num_layers, dropout)
        num_directional = 2 if bidirectional else 1
        self.decoder = AttnRNNDecoder(self.decoder_inputs.output_size + self.target_size, self.target_size, rnn_type,
                                      hidden_size * num_directional, num_layers, dropout, attn_heads, attn_size)

    def forward(self, feed_dict_step, encoder_outputs, hidden, last_target):
        if self.decoder_inputs.output_size != 0:
            dec_inputs = self.decoder_inputs(feed_dict_step)
            dec_inputs = torch.cat([dec_inputs, last_target], dim=2)
        else:
            dec_inputs = last_target
        pred, hidden, attn_weights = self.decoder(dec_inputs, hidden, encoder_outputs)
        return pred, hidden, attn_weights

    @torch.no_grad()
    def predict(self, feed_dict, decode_length=None, return_attn=False):
        if decode_length is None:
            decode_length = self.decode_length
        enc_inputs = self.encoder_inputs(feed_dict)
        enc_outputs, hidden = self.encoder(enc_inputs)
        preds = []
        attns = []
        pred = feed_dict["enc_target"][:, [-1], :]
        for step in range(decode_length):
            feed_dict_step = {k: v[:, [step]] for k, v in feed_dict.items()}
            pred, hidden, attn_weights = self(feed_dict_step, enc_outputs, hidden, pred)
            preds.append(pred)
            attns.append(attn_weights)
        preds = torch.cat(preds, dim=1)
        attns = torch.cat(attns, dim=2).transpose(2, 1)  # batch, seq, head, attn_size
        if return_attn:
            return preds, attns
        else:
            return preds

    def batch_loss(self, feed_dict, target):
        enc_inputs = self.encoder_inputs(feed_dict)
        enc_outputs, hidden = self.encoder(enc_inputs)
        if random.random() < self.teacher_forcing_rate:
            if self.decoder_inputs.output_size != 0:
                dec_inputs = self.decoder_inputs(feed_dict)
                dec_inputs = torch.cat([dec_inputs, feed_dict['dec_target']], dim=2)
            else:
                dec_inputs = feed_dict['dec_target']
            preds, hidden, attn_weights = self.decoder(dec_inputs, hidden, enc_outputs)
        else:
            preds = []
            pred = feed_dict["enc_target"][:, [-1], :]
            for step in range(self.decode_length):
                feed_dict_step = {k: v[:, [step]] for k, v in feed_dict.items()}
                pred, hidden, attn_weights = self(feed_dict_step, enc_outputs, hidden, pred)
                preds.append(pred)
            preds = torch.cat(preds, dim=1)
        loss = self.loss_fn(preds, target)
        return loss


if __name__ == "__main__":
    batch_size = 4
    dec_len = 7
    enc_len = 14
    target_size = 8
    hidden_size = 32
    num_layers = 1
    bidirectional = False
    dropout = 0.5
    share_embeddings = [("enc_week", "dec_week")]

    encode_inputs = {"numerical": [("enc_target", 8)], "categorical": [("enc_week", 8, 2)]}
    decode_inputs = {"categorical": [("dec_week", 8, 2)]}
    model = AttnSeq2Seq(encode_inputs, decode_inputs, target_size, dec_len, "LSTM", hidden_size, num_layers,
                        bidirectional, dropout, 2, 12, share_embeddings=share_embeddings, teacher_forcing_rate=0.5)
    feed_dict = {"enc_target": torch.rand(batch_size, enc_len, target_size),
                 "dec_target": torch.rand(batch_size, dec_len, target_size),
                 "enc_week": torch.randint(0, 4, (batch_size, enc_len)),
                 "dec_week": torch.randint(0, 4, (batch_size, dec_len))
                 }
    target = torch.rand(batch_size, dec_len, target_size)
    model.batch_loss(feed_dict, target)

    preds, attns = model.predict(feed_dict, 7, True)
