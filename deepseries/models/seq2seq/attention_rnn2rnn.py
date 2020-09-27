# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/9/27 14:52
"""
import torch
import torch.nn as nn
from .utils import *
from .encoder import RNNEncoder
from .decoder import AttnRNNDecoder


class AttnSeq2Seq(nn.Module):

    def __init__(self, encoder_inputs, decoder_inputs, target_size, decode_length,
                 rnn_type, hidden_size, num_layers, bidirectional, dropout,
                 attn_heads, attn_size, loss_fn=nn.MSELoss(), share_embeddings=None):
        super().__init__()
        self.encoder_inputs = Inputs(encoder_inputs)
        self.decoder_inputs = Inputs(decoder_inputs)
        self.target_size = target_size
        self.decode_length = decode_length
        self.loss_fn = loss_fn

        if share_embeddings is not None:
            pass

        self.encoder = RNNEncoder(self.encoder_inputs.output_size, rnn_type, hidden_size,
                                  bidirectional, num_layers, dropout)
        num_directional = 2 if bidirectional else 1
        self.decoder = AttnRNNDecoder(self.decoder_inputs.output_size + self.target_size, self.target_size, rnn_type,
                                      hidden_size * num_directional, num_layers, dropout, attn_heads, attn_size)

    def

    def batch_loss(self, feed_dict, target, last_target):
        enc_inputs = self.encoder_inputs(feed_dict)
        enc_outputs, enc_hidden = self.encoder(enc_inputs)
        dec_inputs = self.decoder_inputs(feed_dict)
        dec_inputs = torch.cat([dec_inputs, last_target], dim=2)
        pred, hidden, attn_weights = self.decoder(dec_inputs, enc_hidden, enc_outputs)
        loss = self.loss_fn(pred, target)
        return loss
