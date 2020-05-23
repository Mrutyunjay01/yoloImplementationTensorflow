from Utils.convFixedPadding import conv2D_fiexed_padding
import tensorflow


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
    
    pass
