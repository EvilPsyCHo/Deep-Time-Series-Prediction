# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/12 16:14
"""
import os
from keras.layers import Input, Dense, LSTM, Dropout, Activation
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from dtsp.models.attention import Attention
import numpy as np
import matplotlib.pyplot as plt


class Seq2Seq:

    def __init__(self, target_dim, hidden_size, save_dir, optimizer='adam',
                 activation='tanh', dropout=.0, verbose=0, loss_fn='mse', metrics=['mae']):
        self.save_dir = save_dir
        self.verbose = verbose
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.history = None
        self.opt = optimizer

        enc_input = Input((None, target_dim), name='enc_input')
        enc_rnn = LSTM(hidden_size, return_state=True, name='encoder_rnn', return_sequences=True)
        enc_outputs, enc_h, enc_c = enc_rnn(enc_input)
        enc_states = [enc_h, enc_c]

        dec_input = Input((None, target_dim), name='dec_input')
        dec_rnn = LSTM(hidden_size, return_state=True, return_sequences=True, name='decoder_rnn')

        dec_outputs, dec_h, dec_c = dec_rnn(dec_input, initial_state=enc_states)
        attn = Attention()
        attn_outputs, prob = attn(dec_outputs, enc_outputs)

        dec_dense = Sequential([
            Activation(activation),
            Dropout(dropout, name='dense_dropout_1'),
            Dense(hidden_size * 2, activation=activation, name='dense_1'),
            Dropout(dropout, name='dense_dropout_2'),
            Dense(target_dim, name='dense_2')])
        dense_outputs = dec_dense(attn_outputs)

        model = Model([enc_input, dec_input], dense_outputs)
        model.compile(loss=self.loss_fn, optimizer=self.opt, metrics=self.metrics)
        self.model = model

        self.enc_model = Model(enc_input, enc_states)

        dec_state_inputs = [Input(shape=(hidden_size,)), Input(shape=(hidden_size,))]
        dec_outputs, dec_h, dec_c = dec_rnn(dec_input, initial_state=dec_state_inputs)
        dec_states = [dec_h, dec_c]
        dense_outputs = dec_dense(dec_outputs)
        self.dec_model = Model([dec_input] + dec_state_inputs, [dense_outputs] + dec_states)

    def fit(self, *args, **kwargs):
        save_path = os.path.join(self.save_dir, "model_{epoch:03d}-{val_loss:.4f}.hdf5")
        checkpoint = ModelCheckpoint(save_path, save_weights_only=True, verbose=self.verbose)
        early = EarlyStopping(patience=10, verbose=self.verbose)
        callbacks = [checkpoint, early]
        kwargs['callbacks'] = callbacks
        self.history = self.model.fit(*args, **kwargs)

    def fit_generator(self,
                      generator,
                      validation_data,
                      steps_per_epoch=None,
                      epochs=1,
                      verbose=1,
                      validation_steps=None,
                      class_weight=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      shuffle=True,
                      initial_epoch=0):
        save_path = os.path.join(self.save_dir, "model_{epoch:03d}-{val_loss:.4f}.hdf5")
        checkpoint = ModelCheckpoint(save_path, save_weights_only=True, verbose=self.verbose)
        early = EarlyStopping(patience=10, verbose=self.verbose)
        callbacks = [checkpoint, early]
        history = self.model.fit_generator(generator,
                                           steps_per_epoch=steps_per_epoch,
                                           epochs=epochs,
                                           verbose=verbose,
                                           callbacks=callbacks,
                                           validation_data=validation_data,
                                           validation_steps=validation_steps,
                                           class_weight=class_weight,
                                           max_queue_size=max_queue_size,
                                           workers=workers,
                                           use_multiprocessing=use_multiprocessing,
                                           shuffle=shuffle,
                                           initial_epoch=initial_epoch)
        self.history = history

    def predict(self, x, predict_lens):
        dec_states = self.enc_model.predict(x)
        dec_inputs = np.expand_dims(x[:, -1, ], axis=1)
        preds = []
        for step in range(predict_lens):
            dec_outputs, h, c = self.dec_model.predict([dec_inputs] + dec_states)
            dec_inputs = dec_outputs
            dec_states = [h, c]
            preds.append(dec_inputs)
        preds = np.concatenate(preds, axis=1)
        return preds

    def load_weights(self, file_path):
        if os.path.exists(file_path):
            self.model.load_weights(file_path)

    def save_weights(self, file_path):
        self.model.load_weights(file_path)

    def plot_loss(self, figsize=(12, 6)):
        f = plt.figure(figsize=figsize)
        h = self.history.history
        plt.plot(h['loss'], label='train_loss')
        plt.plot(h['val_loss'], label='val_loss')
        plt.legend()
