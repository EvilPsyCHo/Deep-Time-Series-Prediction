# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/6 10:57
"""
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import numpy as np


class SimpleSeq2Seq(Model):

    def __init__(self, target_dim, hidden_size):
        super(SimpleSeq2Seq, self).__init__()
        self.target_dim = target_dim
        self.hidden_size = hidden_size
        self.encoder = layers.LSTM(self.hidden_size, return_state=True)
        self.decoder = layers.LSTM(self.hidden_size, return_state=True, return_sequences=True)
        self.out = layers.Dense(target_dim, activation='relu')

    def call(self, enc_inputs, dec_inputs):
        # inputs shape: batch_size X time_step X inputs_dim
        encoder_outputs, h, c = self.encoder(enc_inputs)
        decoder_outputs, _, _ = self.decoder(dec_inputs, initial_state=(h, c))
        outputs = self.out(decoder_outputs)
        return outputs

    def decode(self, enc_inputs, n_step):
        """

        Args:
            enc_inputs (ArrayLike): shape = (Batch, TimeStep, Dim)
            n_step (Int):

        Returns:

        """
        decode_seqs = []
        batch, enc_lens, dim = enc_inputs.shape
        encoder_outputs, h, c = self.encoder(enc_inputs)
        decoder_inputs = enc_inputs[:, -1, :].reshape(batch, 1, dim)
        for i in range(n_step):
            decoder_outputs, h, c = self.decoder(decoder_inputs, initial_state=(h, c))
            outputs = self.out(decoder_outputs)
            decode_seqs.append(outputs)
            decoder_inputs = outputs
        return decode_seqs


x = np.random.rand(10, 12, 1)
model = SimpleSeq2Seq(1, 10)
model.compile(optimizer='Adam', loss='mse',
              metrics=['mae'])

model.fit((x[:, :6, :], x[:, 1:7, :]), x[:, 2:8, :], batch_size=12, epochs=10)