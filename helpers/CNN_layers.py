# this file construct a few layers for CNN's in tensorflow

import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, dtype=tf.float32)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, dtype=tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class ConvLayer:
    """Convolution layer for Graphs, followed by RELU"""

    def __init__(self, input, F, window=(5, 5)):
        """
        :param input: input from previous layer, size [None, ?, ?, ?]
        :param F: number of filters
        """
        self.F = F
        F_previous = int(input.get_shape()[3])
        # weights for the convolution
        self.W = weight_variable(list(window) + [F_previous, self.F])
        # bias
        self.b = bias_variable([self.F])

        # compute the output after convolution + relu
        self.output = tf.nn.relu(conv2d(input, self.W) + self.b)

    def max_pool(self):
        """Apply max-pooling to this layer"""
        self.output = max_pool_2x2(self.output)


class DenseLayer:
    """ Dense layer"""
    def __init__(self, input, u):
        """
        :param input:
        :param u: number of units
        """
        if len(input.get_shape())>2:
            # from conv to dense
            _, W, H, F = input.get_shape()  # _, width, height, filters
            W, H, F = int(W), int(H), int(F)

            self.W = weight_variable([W * H * F, u])
            self.b = bias_variable([u])

            y = tf.reshape(input, [-1, W * H * F])

            self.output = tf.matmul(y, self.W) + self.b

        elif len(input.get_shape())== 2:
            # from dense to dense
            _, M = input.get_shape()
            M = int(M)

            self.W = weight_variable([M, u])
            self.b = bias_variable([u])

            y = tf.reshape(input, [-1, M])

            self.output = tf.matmul(y, self.W) + self.b

        self.relu()  # always perform a relu

    def relu(self):
        self.output = tf.nn.relu(self.output)

    def sofmax(self):
        self.output = tf.nn.softmax(self.output)

    def drop_out(self):
        """ Don't forget to pass keep_prob in the feed_dict when running """
        self.keep_prob = tf.placeholder(tf.float32)
        self.output = tf.nn.dropout(self.output, self.keep_prob)
