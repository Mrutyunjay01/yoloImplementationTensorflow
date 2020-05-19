import tensorflow as tf


def fixedPadding(inputs, kernel_size, data_format):
    """
    Pads the input along the spatial dimensions independently of input size
    :param inputs: Tensor input
    :param kernel_size: kernel to be used in the Conv2D or MaxPool2D
    :param data_format: channels_last or channels_first
    :return: A tensor with the same format as input
    """

    pad_total = kernel_size - 1
    pad_beginnig = pad_total // 2
    pad_end = pad_total - pad_beginnig

    if data_format == 'channels_first':
        pad_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                     [pad_beginnig, pad_end],
                                     [pad_beginnig, pad_end]])

    else:
        pad_inputs = tf.pad(inputs, [[0, 0],
                                     [pad_beginnig, pad_end],
                                     [pad_beginnig, pad_end],
                                     [0, 0]])

    return pad_inputs
