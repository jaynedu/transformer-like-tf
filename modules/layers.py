# -*- coding: utf-8 -*-
# @Date: 2020/7/22 14:08
# @Author: Du Jing
# @FileName: layers
# ---- Description ----
#


import numpy as np
import tensorflow as tf

from modules.base import gelu, layer_normalization

__all__ = [
    'position_encoding',
    'feed_forward',
    'residual_connection'
]


def position_encoding(inputs, maxlen, masking=True, scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (batch_size, seq_len, nfeature)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.
    returns
    3d tensor that has the same shape as inputs.
    '''

    nfeature = inputs.get_shape().as_list()[-1]  # static
    batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch_size, 1])  # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - (i % 2)) / nfeature) for i in range(nfeature)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, nfeature)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)


def feed_forward(inputs, units: list, dropout, training: bool):
    inner = tf.layers.dense(inputs, units[0])
    activated_inner = gelu(inner)
    outputs = tf.layers.dense(activated_inner, units[1])
    if training is not None and training is True:
        outputs = tf.layers.dropout(outputs, dropout, training=training)
    return outputs


def residual_connection(x, fx, training):
    outputs = x + fx
    outputs = tf.layers.batch_normalization(outputs, training=training)
    # outputs = layer_normalization(outputs)
    return outputs
