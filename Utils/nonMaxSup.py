import tensorflow as tf


# The model is going to produce a lot of boxes, so we need a way to discard the boxes with low confidence scores.
# Also, to avoid having multiple boxes for one object, we will discard the boxes with high overlap as well using
# non-max suppresion for each class.

def buildBoxes(inputs):
    """
    computes top left and bottom right points of the boxes
    :param inputs: output from the detection layer
    :return: boxes for the detected object
    """
    # retrieve coordinates from the output of detection layer
    center_x, center_y, width, height, confiScore, classes = tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    topLeft_x = center_x - width / 2
    topLeft_y = center_y - height / 2

    bottomRight_x = center_x + width / 2
    bottomRight_y = center_y + height / 2

    boxes = tf.concat([topLeft_x, topLeft_y,
                       bottomRight_x, bottomRight_y,
                       confiScore,
                       classes], axis=-1)

    return boxes


def nms(inputs, n_classes, max_ouput_size, iou_threshold, confidence_threshold):
    """
    compute nms separately for each class
    :param inputs: boxes with corner coordinates
    :param n_classes: number of classes
    :param max_ouput_size: max number of boxes to choose
    :param iou_threshold: iou threshold
    :param confidence_threshold: conf threshold
    :return: boxes dictionary wrt classes
    """
    batch = tf.unstack(inputs)
    boxes_dicts = []

    for boxes in batch:

        # select the boxes with score more than confidence score
        boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)
        # select the index of the class with max probability
        classes = tf.argmax(boxes[:, 5:], axis=-1)
        # why 5: ?
        # because after 5th row onwards it comes classes probabilities
        classes = tf.expand_dims(classes, axis=-1)

        boxes = tf.concat([boxes[:,:5], classes], axis=-1)

        boxes_dict = {}

        for cls in range(n_classes):
            mask = tf.equal(boxes[:, 5], cls)
            maskShape = mask.get_shape()

            if maskShape.ndims != 0:
                class_boxes = tf.boolean_mask(boxes,mask)
                boxes_coords, boxes_confiScore, _ = tf.split(class_boxes,
                                                             [4, 1, -1],
                                                             axis=-1)
                boxes_confiScore = tf.reshape(boxes_confiScore, [-1])
                indices = tf.image.non_max_suppression(boxes_coords,
                                                       boxes_confiScore,
                                                       max_ouput_size=max_ouput_size,
                                                       iou_threshold=iou_threshold)
                class_boxes = tf.gather(class_boxes, indices=indices)
                boxes_dict[cls] = class_boxes[:, :5]

        boxes_dicts.append(boxes_dict)

    return boxes_dicts
