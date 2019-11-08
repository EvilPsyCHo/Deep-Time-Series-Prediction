# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/8 15:44
"""
import os
import matplotlib.pyplot as plt
from keras.layers import *
from keras.initializers import *
from keras.regularizers import *
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K
from keras.callbacks import *
import numpy as np


def dilation_block_v2(filters, kernel_size, dilation):
    def f(x):
        half = filters // 2

        x = Conv1D(half, 1, padding='same', activation='relu')(x)

        x_f = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation)(x)
        x_g = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation)(x)

        skip = Multiply()([Activation('tanh')(x_f), Activation('sigmoid')(x_g)])
        skip = Conv1D(half, 1, padding='same', activation='relu')(skip)
        # add residual connection
        x = Add()([x, skip])

        return x, skip
    return f


def dilation_block_v1(filters, kernel_size, dilation):
    def f(x):
        residual = x

        x = Conv1D(filters=filters, kernel_size=kernel_size,
                   dilation_rate=dilation,
                   activation='linear', padding='causal', use_bias=False)(x)

        x = Activation('selu')(x)

        skip = Conv1D(1, 1, activation='linear', use_bias=False,
                      kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(x)

        x = Conv1D(1, 1, activation='linear', use_bias=False,
                   kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05))(x)

        x = Add()([x, residual])

        return x, skip

    return f


class SimpleWaveNet(object):

    def __init__(self, target_dim, filters, n_layers, save_dir, dropout=.0, wavenet_mode='v1',
                 loss_fn='mse', optimizer='default', verbose=0, metrics=['mae']):
        self.save_dir = save_dir
        self.verbose = verbose
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.wavenet_mode = wavenet_mode
        self.history = None
        self.dropout = dropout
        if optimizer == 'default':
            self.opt = Adam(lr=0.001, beta_2=0.99, decay=0.001)
        elif callable(optimizer) or isinstance(optimizer, str):
            self.opt = optimizer
        if self.wavenet_mode == 'v1':
            cnn_block = dilation_block_v1
        else:
            cnn_block = dilation_block_v2

        skips = []

        x = Input(shape=(None, target_dim))
        for l in range(n_layers):
            x, skip_i = cnn_block(filters, 2, 2)(x)
            skips.append(skip_i)

        out = Activation('relu')(Add()(skips))
        out = Conv1D(filters, 1, padding='same')(out)
        out = Activation('relu')(out)
        out = Dropout(self.dropout)(out)
        out = Conv1D(target_dim, 1, padding='same')(out)
        model = Model(x, out)
        model.compile(loss=self.loss_fn, optimizer=self.opt, metrics=self.metrics)
        self.model = model

    def predict(self, x, predict_lens):
        # x = (batch, seq_lens, target_dim)
        preds = []
        for step in range(predict_lens):
            preds.append(self.model.predict(x))
            # pred = (batch, 1, target_dim)
            x = K.concatenate([x[:, 1:, :], preds[-1]], axis=1)
        return preds

    def fit_generator(self,
                      generator,
                      validation_data,
                      steps_per_epoch=None,
                      epochs=1,
                      verbose=1,
                      validation_steps=None,
                      validation_freq=1,
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
                                           validation_freq=validation_freq,
                                           class_weight=class_weight,
                                           max_queue_size=max_queue_size,
                                           workers=workers,
                                           use_multiprocessing=use_multiprocessing,
                                           shuffle=shuffle,
                                           initial_epoch=initial_epoch)
        self.history = history

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
        return f
