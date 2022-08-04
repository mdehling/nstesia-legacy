"""
Image Pre-/Post-Processing layers.
"""

__all__ = [
    'ImagePreProcessing',
    'ImagePostProcessing',
]

import tensorflow as tf

from .. import image as nst_image


class ImagePreProcessing(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return nst_image.preprocess(inputs)


class ImagePostProcessing(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return nst_image.postprocess(inputs)
