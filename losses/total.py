"""
Total Loss.

A simple weighted sum of individual losses.
"""

__all__ = [
    'Total',
]

import tensorflow as tf


class Total(tf.keras.losses.Loss):

    def __init__(self, losses, weights, **kwargs):
        super().__init__(**kwargs)

        self.losses = losses
        self.weights = weights

    def call(self, y_true, y_pred):
        weighted_losses = [
            weight * loss_fn(y_true, y_pred)
            for loss_fn, weight in zip(self.losses, self.weights)
        ]
        return tf.add_n(weighted_losses)

    def get_config(self):
        config = super().get_config()
        config.update({
            'losses': self.losses,
            'weights': self.weights
        })
        return config
