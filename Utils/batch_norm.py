import tensorflow as tf

## define hyperparameters for Batch Normalisation
_EPSILON = 1e-05
_DECAY = 0.9


def batchNorm(inputs, training, data_format):
    """
    Performs bn using standard set of parameters
    :param inputs: input images
    :param training: train true/false
    :param data_format: channels_last or channels_first
    :return: bn applied
    """
    return tf.keras.layers.BatchNormalization(inputs=inputs,
                                              axis=1 if data_format == 'channels_first' else 3,
                                              momentum=_DECAY,
                                              epsilon=_EPSILON,
                                              scale=True,
                                              trainable=training)
