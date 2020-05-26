from Utils.convFixedPadding import conv2D_fiexed_padding
from Utils.batch_norm import batchNorm
from featureExtractor.darknet53ResidualBlock import darknet53_residualBlock
import tensorflow as tf

_LEAKY_RELU = 0.1


def darknet53(inputs, training, data_format):
    """
    creates Darknet 53 model for feature extraction
    :param inputs:
    :param training:
    :param data_format:
    :return: C3, C4, C5 from stage 3, 4, 5 respectively as working similarly to feature extractor
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
    inputs = darknet53_residualBlock(inputs, training=training,
                                     filters=32, strides=1, data_format=data_format)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=128,
                                   kernel_size=3,
                                   strides=2,
                                   data_format=data_format)
    inputs = batchNorm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    # 2nd residual block
    for _ in range(2):
        inputs = darknet53_residualBlock(inputs,
                                         training=training,
                                         filters=64,
                                         strides=1,
                                         data_format=data_format)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=256,
                                   kernel_size=3,
                                   strides=2,
                                   data_format=data_format)
    inputs = batchNorm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    # 3rd res block
    for _ in range(8):
        inputs = darknet53_residualBlock(inputs,
                                         filters=128,
                                         training=training,
                                         data_format=data_format,
                                         strides=1)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=512,
                                   kernel_size=3,
                                   strides=2,
                                   data_format=data_format)
    inputs = batchNorm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    C3 = inputs
    # 4th res block
    for _ in range(8):
        inputs = darknet53_residualBlock(inputs,
                                         filters=256,
                                         strides=1,
                                         training=training,
                                         data_format=data_format)

    inputs = conv2D_fiexed_padding(inputs,
                                   filters=1024,
                                   kernel_size=3,
                                   strides=2,
                                   data_format=data_format)
    inputs = batchNorm(inputs, training, data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    C4 = inputs
    # 5th res block
    for _ in range(4):
        inputs = darknet53_residualBlock(inputs,
                                         filters=512,
                                         training=training,
                                         data_format=data_format,
                                         strides=1)
    C5 = inputs

    return C3, C4, C5
    # as we will not be considering the pooling and fc and softmax layer
    # because we are interested in only feature extractor
