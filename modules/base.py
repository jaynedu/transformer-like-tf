# -*- coding: utf-8 -*-
# @Date: 2020/7/23 12:37
# @Author: Du Jing
# @FileName: base
# ---- Description ----
#

import tensorflow as tf


def layer_normalization(inputs, epsilon=1e-8, scope="norm_layer"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs


def gelu(input_tensor):
    '''Gaussian Error Linear Unit.'''
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf
