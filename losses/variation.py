"""
Variation Loss.
"""

__all__ = [
    'Variation',
]

import tensorflow as tf


class Variation(tf.keras.losses.Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, content_image, stylized_image):
        variation_losses = tf.image.total_variation(stylized_image)
        return tf.reduce_mean(variation_losses)
