# import required packages
from Utils.batch_norm import batchNorm
from Utils.convFixedPadding import conv2D_fiexed_padding
import tensorflow as tf


def yoloConvBlock(inputs, filters, training, data_format):
    """
    Creates block for additional layer on the top of Yolo
    :param inputs: input image batch
    :param filters: no of filters
    :param training: true/false
    :param data_format: channels_last'first
    :return: conv output
    """
    inputs = conv2D_fiexed_padding(inputs,
                                   filters=filters,
                                   kernel_size=1,
                                   data_format=data_format)
    inputs = batchNorm(inputs,
                       training=training,
                       data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=0.1)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=2 * filters,
                                   kernel_size=3,
                                   data_format=data_format)
    inputs = batchNorm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=0.1)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=filters,
                                   kernel_size=1,
                                   data_format=data_format)
    inputs = batchNorm(inputs,
                       training=training,
                       data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=0.1)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=2 * filters,
                                   kernel_size=3,
                                   data_format=data_format)
    inputs = batchNorm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=0.1)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=filters,
                                   kernel_size=1,
                                   data_format=data_format)
    inputs = batchNorm(inputs,
                       training=training,
                       data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=0.1)

    route = inputs

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=2 * filters,
                                   kernel_size=3,
                                   data_format=data_format)
    inputs = batchNorm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=0.1)

    return route, inputs
    pass
