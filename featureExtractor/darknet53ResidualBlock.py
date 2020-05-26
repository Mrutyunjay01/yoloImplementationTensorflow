from Utils.convFixedPadding import conv2D_fiexed_padding
from Utils.batch_norm import batchNorm
import tensorflow as tf

_LEAKY_RELU = 0.1


def darknet53_residualBlock(inputs, filters, training, data_format, strides):
    """
    creates a residual block for resnet
    ref : https://miro.medium.com/max/792/1*7u6XWGYl7lLgc0EcKG1NMw.png
    :param inputs: input tensor
    :param filters: no of filters
    :param training: True/False
    :param data_format: channels_first/channels_last
    :param strides: strides
    :return: residual block constructor
    """
    shortcuts = inputs

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=filters,
                                   kernel_size=1,
                                   strides=strides,
                                   data_format=data_format)
    inputs = batchNorm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=2*filters,
                                   kernel_size=3,
                                   strides=strides,
                                   data_format=data_format)
    inputs = batchNorm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs += shortcuts

    return inputs
    pass
