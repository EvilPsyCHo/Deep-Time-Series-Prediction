# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/12 15:22
"""
import keras.backend as K
from keras.layers import Layer, Dense, Input, RNN

# [1] writing your own Keras Layers (https://keras.io/layers/writing-your-own-keras-layers/)


class Attention(Layer):

    def __init__(self):
        self.W = None
        super(Attention, self).__init__()

    def build(self, input_shape):
        """

        :param input_shape: [decoder_hidden_states, encoder_hidden_states] = [(B, 1, H), ([B, S, H)]
        """
        self.W = self.add_weight('attn_W', shape=(input_shape[0][-1], input_shape[0][-1]), initializer='glorot_uniform')
        super(Attention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        decoder_hidden_states, encoder_hidden_states = inputs[0], inputs[1]
        q = decoder_hidden_states  # (B, N, H)
        k = encoder_hidden_states  # (B, S, H)
        v = encoder_hidden_states  # (B, S, H)
        print('-' * 40)
        print(q.shape, k.shape)
        q = K.dot(q, self.W)  # (B, N, H)
        k = K.permute_dimensions(k, (0, 2, 1))  # (B, H, S)
        energy = K.dot(q, k)  # (B, N, S)
        prob = K.softmax(energy, axis=2)  # (B, N, S)
        attn = K.dot(prob, v)  # (B, N, H)
        return attn, prob

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        q, k = input_shape
        return [q, (k[0], q[1], k[1])]



