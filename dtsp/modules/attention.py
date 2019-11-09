# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/7 16:49
"""
import keras.backend as K
from keras.layers import Layer, Dense


class GeneralAttention(Layer):

    def __init__(self):
        # self.hidden_size = hidden_size
        super(GeneralAttention, self).__init__()

    def build(self, input_shape):
        hidden_size = input_shape[0][0]
        self.W = Dense(hidden_size)

        super(GeneralAttention, self).build(input_shape)
