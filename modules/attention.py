# -*- coding: utf-8 -*-
# @Date: 2020/7/23 12:41
# @Author: Du Jing
# @FileName: attention
# ---- Description ----
#

import tensorflow as tf


def scaled_dot_product_attention(Q, K, V, dropout, training, scope="scaled_dot_product_attention"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]
        outputs = tf.matmul(Q, K, transpose_b=True)  # dot product
        outputs /= (d_k ** 0.5)  # scale

        outputs = tf.nn.softmax(outputs)  # softmax

        # draw
        score = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention score", tf.expand_dims(score[:1], -1))

        if training is not None and training is True:
            outputs = tf.layers.dropout(outputs, rate=dropout, training=training)

        outputs = tf.matmul(outputs, V)  # weighted sum (batch_size, seqlen, d_v)
        return outputs


def scaled_dot_product_attention_handmade(Q, K, V, dropout, training, scope="scaled_dot_product_attention"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]
        outputs = tf.matmul(Q, K, transpose_b=True)  # dot product
        outputs /= (d_k ** 0.5)  # scale

        z = outputs - tf.reduce_max(outputs, axis=2, keepdims=True)
        z = tf.exp(z)
        outputs = z / tf.reduce_sum(z, axis=2, keepdims=True)

        # draw
        score = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention score", tf.expand_dims(score[:1], -1))

        if training is not None and training is True:
            outputs = tf.layers.dropout(outputs, rate=dropout, training=training)

        outputs = tf.matmul(outputs, V)  # weighted sum (batch_size, seqlen, d_v)
        return outputs


def linear_attention_tayler(Q, K, V, dropout, training, scope="tayler_dot_product_attention"):
    eps = 1e-7
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        n_k, d_k = Q.get_shape().as_list()[1:]
        # _Q = tf.tanh(Q)
        # _K = tf.tanh(K)
        _Q = tf.nn.l2_normalize(Q)
        _K = tf.nn.l2_normalize(K)

        outputs = (V + tf.matmul(_Q, tf.matmul(_K, V, transpose_a=True)) / d_k ** 0.5) / \
                  (tf.reduce_sum(tf.cast(n_k, tf.float32) + tf.matmul(_Q, _K, transpose_b=True) / d_k**0.5, axis=2, keepdims=True) + eps)

        if training is not None and training is True:
            outputs = tf.layers.dropout(outputs, rate=dropout, training=training)

        return outputs


def linear_attention_kernel(Q, K, V, dropout, training, scope="linear_attention_kernel"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        _Q = tf.nn.elu(Q) + 1
        _K = tf.nn.elu(K) + 1
        outputs = tf.matmul(_Q, tf.matmul(_K, V, transpose_a=True))/tf.reduce_sum(tf.matmul(_Q, _K, transpose_b=True), axis=2, keepdims=True)

        if training is not None and training is True:
            outputs = tf.layers.dropout(outputs, rate=dropout, training=training)

        return outputs


def linear_attention_multiple_softmax(Q, K, V, dropout, training, scope="linear_attention_multiple_softmax"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        _Q = tf.exp(Q)
        _K = tf.exp(K)
        Q_ = _Q / tf.reduce_sum(_Q, axis=2, keepdims=True)
        K_ = _K / tf.reduce_sum(_K, axis=1, keepdims=True)
        outputs = tf.matmul(Q_, tf.matmul(K_, V, transpose_a=True))
        if training is not None and training is True:
            outputs = tf.layers.dropout(outputs, rate=dropout, training=training)

        return outputs


def multi_head_attention(keys, queries, values, head_num, head_size, dropout, training, scope="multi_head_attention",
                         type="softmax"):
    '''
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    '''
    assert type in ["softmax", "tayler", "kernel", "multi_softmax"], print("check the attention unit type!")
    hidden_size = head_num * head_size  # d_model = hidden_size
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        Q = tf.layers.dense(queries, hidden_size, use_bias=True)
        K = tf.layers.dense(keys, hidden_size, use_bias=True)
        V = tf.layers.dense(values, hidden_size, use_bias=True)

        # Split and concat
        _Q = tf.concat(tf.split(Q, head_num, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        _K = tf.concat(tf.split(K, head_num, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        _V = tf.concat(tf.split(V, head_num, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        if type == "softmax":
            score = scaled_dot_product_attention_handmade(_Q, _K, _V, dropout=dropout, training=training)
        if type == "tayler":
            score = linear_attention_tayler(_Q, _K, _V, dropout=dropout, training=training)
        if type == "kernel":
            score = linear_attention_kernel(_Q, _K, _V, dropout, training)
        if type == "multi_softmax":
            score = linear_attention_multiple_softmax(_Q, _K, _V, dropout, training)
        outputs = tf.concat(tf.split(score, head_num, axis=0), axis=2)

        return outputs

