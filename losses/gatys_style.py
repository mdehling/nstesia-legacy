"""
Gatys Style Loss.
"""

__all__ = [
    'GatysStyle',
]

import tensorflow as tf
from .. import models as nst_models


class GatysStyle(tf.keras.losses.Loss):

    def __init__(self, style_image=None, model='vgg16', **kwargs):
        super().__init__(**kwargs)

        self.style_image = style_image

        if isinstance(model, str):
            self.style_model = nst_models.GatysStyle(model=model)
        else:
            self.style_model = model

        if self.style_image is not None:
            self.style_targets = self.style_model(self.style_image)
        else:
            self.style_targets = None

    def call(self, style_image, stylized_image):
        style_targets = self.style_targets or self.style_model(style_image)
        style_features = self.style_model(stylized_image)

        style_layer_losses = [
            tf.reduce_mean(tf.square(target-feature))
            for target, feature in zip(style_targets, style_features)
        ]

        return tf.add_n(style_layer_losses) / len(style_layer_losses)
