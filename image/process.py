"""
NSTesia image pre-/post-processing.
"""

__all__ = [
    'preprocess',
    'postprocess',
]

import tensorflow as tf


imagenet_means = [123.680, 116.779, 103.939]


def preprocess(x):
    return x - tf.constant(imagenet_means, shape=(1,1,1,3))


def postprocess(x):
    x = x + tf.constant(imagenet_means, shape=(1,1,1,3))
    return tf.clip_by_value(x, 0.0, 255.0)
