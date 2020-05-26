from Utils.convFixedPadding import conv2D_fiexed_padding
from Utils.batch_norm import batchNorm
import tensorflow as tf
_LEAKY_RELU = 0.1

def darknet53(inputs, training, data_format):
    """
    creates Darknet 53 model for feature extraction
    :param inputs:
    :param training:
    :param data_format:
    :return:
    """
    inputs = conv2D_fiexed_padding(inputs,
                                   filters=32,
                                   kernel_size=3,
                                   data_format=data_format)
    inputs = batchNorm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=64,
                                   kernel_size=3,
                                   strides=2,
                                   data_format=data_format)
    inputs = batchNorm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    # 1st residual block
