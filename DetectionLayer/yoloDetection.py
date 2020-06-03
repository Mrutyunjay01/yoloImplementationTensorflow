import tensorflow as tf


def yoloDetection(inputs, n_classes, anchors, imgSize, data_format):
    """
    Detects boxes with respect to anchors.
    4 coordinates, 1 objectiveness score, n_classes probabilities from feature map
    :param inputs:
    :param n_classes:
    :param anchors:
    :param imgSize:
    :param data_format:
    :return:
    """
    n_anchors = len(anchors)

    # inputs from feature extractor output
    inputs = tf.keras.layers.Conv2D(inputs,
                                    filters=n_anchors * (5 + n_classes),
                                    kernel_size=1,
                                    strides=1,
                                    use_bias=True,
                                    data_format=data_format)

    shape = inputs.get_shape().as_list()
    gridShape = shape[2:4] if data_format == 'channels_first' else shape[1:3]

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])  # took channels attribute to the last dimension

    inputs = tf.reshape(inputs, [-1, n_anchors * gridShape[0] * gridShape[1], 5 + n_classes])
    # in the prev step, I inverted the dimensions, took image heiht widht wrt anchor boxes,
    # and extracted probability scores and cordinates score

    strides = (imgSize[0] // gridShape[0],
               imgSize[1] // gridShape[1])

    box_centers, box_shapes, confiScore, classes = tf.split(inputs, [2, 2, 1, n_classes], axis=-1)

    x = tf.range(gridShape[0], dtype=tf.float32)
    y = tf.range(gridShape[1], dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(x, y)  # creates a grid of x * y i.e same as image size

    # flattes the tensors into vectors
    x_offset = tf.reshape(x_offset, (-1, 1))
    y_offset = tf.reshape(y_offset, (-1, 1))
    # stack the vectors column wise
    xy_off = tf.concat([x_offset, y_offset], axis=-1)

    # as we will be predicting values for number of anchor boxes
    # tile the result n_anchor times
    xy_off = tf.tile(xy_off, [1, n_anchors])
    # unroll the params to concat with box_centers
    xy_off = tf.reshape(xy_off, [1, -1, 2])

    box_centers = tf.nn.sigmoid(box_centers)
    box_centers = (box_centers + xy_off) * strides

    anchors = tf.tile(anchors, [gridShape[0] * gridShape[1], 1])
    box_shapes = tf.exp(box_shapes) * anchors

    confiScore = tf.nn.sigmoid(confiScore)

    classes = tf.nn.sigmoid(classes)

    inputs = tf.concat([box_centers, box_shapes, confiScore, classes], axis=-1)
    # 2 for box_centers, box_shapes each, 1 for confiscore and no of classes, hence 5 * n_classes

    return inputs

