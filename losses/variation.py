"""
Anisotropic Total Variation Loss.
"""

__all__ = [
    'Variation',
]

import tensorflow as tf


class Variation(tf.keras.losses.Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, content_image, stylized_image):
        if isinstance(stylized_image, tuple):
            stylized_image = stylized_image[0]

        variation_losses = tf.image.total_variation(stylized_image)
        return tf.reduce_mean(variation_losses)
