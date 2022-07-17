"""
Reflection padding.
"""

__all__ = [
    'ReflectionPadding2D',
]

import tensorflow as tf


class ReflectionPadding2D(tf.keras.layers.Layer):
    """
    Reflection padding layer.

    For use with 4-D (batch_size, height, width, channels) format only!
    """
    def __init__(self, padding=(1,1), **kwargs):
        super().__init__(**kwargs)

        self.padding = padding

        hpadding, wpadding = self.padding
        self.padding_tensor = tf.constant(
            [[0,0], [hpadding, hpadding], [wpadding, wpadding], [0,0]]
        )

    def call(self, inputs, **kwargs):
        return tf.pad(inputs, self.padding_tensor, mode='reflect')

    def get_config(self):
        config = super().get_config()
        config.update({
            'padding': self.padding
        })
        return config
