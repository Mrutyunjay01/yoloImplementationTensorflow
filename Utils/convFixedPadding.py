from Utils.fixed_padding import fixedPadding
import tensorflow as tf


def conv2D_fiexed_padding(inputs, filters, kernel_size, data_format, strides=1):
    """ strided convolution with explicit padding """
    if strides > 1:
        inputs = fixedPadding(inputs, kernel_size, data_format)

    return tf.keras.layers.Conv2D(
        inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same' if strides == 1 else 'valid',
        use_bias=False,
        data_format=data_format
    )
    pass
