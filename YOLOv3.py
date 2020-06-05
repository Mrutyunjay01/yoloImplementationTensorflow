import tensorflow as tf
from featureExtractor.darknet53 import darknet53
from featureExtractor.yoloConvBlock import yoloConvBlock
from DetectionLayer.yoloDetection import yoloDetection
from Utils.convFixedPadding import conv2D_fiexed_padding
from Utils.batch_norm import batchNorm
from Utils.upsampleLayer import upsample
from Utils.nonMaxSup import buildBoxes, nms

_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]


class YOLOV3:
    """
    Yolo v3 main model class.
    """

    def __init__(self, n_classes, model_size, max_output_size, iou_threshold, confidence_threshold, data_format=None):
        """
        Creates the model for YOLO framework.
        :param n_classes: Numvber of classes
        :param model_size: ipnut size of the model
        :param max_output_size: max number of boxes to be selected for each class
        :param iou_threshold: threshold for iou
        :param confidence_threshold: threshold for confidence score
        :param data_format: channels_last or channels_first
        """
        if not data_format:
            if tf.test.is_built_with_cuda():
                data_format = 'channels_first'

            else:
                data_format = 'channels_last'

        self.n_classes = n_classes
        self.model_size = model_size
        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.data_format = data_format

    def __call__(self, inputs, training):
        """
        Add operations to detect boxes for a batch of input images.
        :param inputs: A tensor representing a batch of input images
        :param training: A boolean value whether to train or inference mode
        :return: A list containing class to boxes dictionaries
        """
        with tf.compat.v1.variable_scope('yolo_v3_model'):
            if self.data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            # normalize the pixel values
            inputs = inputs / 255.0

            # extract layers from darknet extractor
            C3, C4, C5 = darknet53(inputs, training=training,
                                   data_format=self.data_format)

            # extract output from yolo layer (on top of the darknet)
            route, out = yoloConvBlock(C5,
                                       filters=512,
                                       training=training,
                                       data_format=self.data_format)
            detectionS1 = yoloDetection(out,
                                        n_classes=self.n_classes,
                                        anchors=_ANCHORS[6:9],
                                        imgSize=self.model_size,
                                        data_format=self.data_format)
            out = conv2D_fiexed_padding(route,
                                        filters=256,
                                        kernel_size=1,
                                        data_format=self.data_format)
            out = batchNorm(out, training=training, data_format=self.data_format)
            out = tf.nn.leaky_relu(out, alpha=0.1)

            # concat C4 with out via upsampling
            upsample_size = C4.get_shape().as_list()
            out = upsample(out, out_shape=upsample_size, data_format=self.data_format)

            axis = 1 if self.data_format == 'channels_first' else 3
            out = tf.concat([out, C4], axis=axis)

            route, out = yoloConvBlock(out,
                                       filters=256,
                                       training=training,
                                       data_format=self.data_format)
            detectionS2 = yoloDetection(out,
                                        n_classes=self.n_classes,
                                        anchors=_ANCHORS[3:6],
                                        imgSize=self.model_size,
                                        data_format=self.data_format)
            out = conv2D_fiexed_padding(route,
                                        filters=128,
                                        kernel_size=1,
                                        data_format=self.data_format)
            out = batchNorm(out, training=training, data_format=self.data_format)
            out = tf.nn.leaky_relu(out, alpha=0.1)

            # concat C4 with out via upsampling
            upsample_size = C3.get_shape().as_list()
            out = upsample(out, out_shape=upsample_size, data_format=self.data_format)

            axis = 1 if self.data_format == 'channels_first' else 3
            out = tf.concat([out, C3], axis=axis)

            route, out = yoloConvBlock(out,
                                       filters=128,
                                       training=training,
                                       data_format=self.data_format)
            detectionS3 = yoloDetection(out,
                                        n_classes=self.n_classes,
                                        anchors=_ANCHORS[0:3],
                                        imgSize=self.model_size,
                                        data_format=self.data_format)
            out = tf.concat([detectionS1, detectionS2, detectionS3], axis=1)

            out = buildBoxes(out)

            boxes_dictionary = nms(out,
                                   n_classes=self.n_classes,
                                   max_ouput_size=self.max_output_size,
                                   iou_threshold=self.iou_threshold,
                                   confidence_threshold=self.confidence_threshold)

            return boxes_dictionary
