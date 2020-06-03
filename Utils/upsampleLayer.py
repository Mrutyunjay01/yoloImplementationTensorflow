import tensorflow as tf


# as we will be concatenating previous layers in darket to next layers by
# skiping a few layers, we need to upsample the prev layers
# here we perform neearest neighbour interpolation upsampling

def upsample(inputs, out_shape, data_format):
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        height = out_shape[3]
        width = out_shape[2]
    else:
        height = out_shape[2]
        width = out_shape[1]

    inputs = tf.image.resize(inputs, (height, width), method='nearest')

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    return inputs
