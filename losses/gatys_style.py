"""
Gatys Style Loss.
"""

__all__ = [
    'GatysStyle',
]

import tensorflow as tf
import nstesia.models as nst_models


class GatysStyle(tf.keras.losses.Loss):

    def __init__(self, style_image, model='vgg16', **kwargs):
        super().__init__(**kwargs)

        self.style_image = style_image

        if isinstance(model, str):
            self.style_model = nst_models.GatysStyle(model=model)
        else:
            self.style_model = model

        self.style_targets = self.style_model(self.style_image)

    def call(self, content_image, stylized_image):
        style_targets = self.style_targets
        style_features = self.style_model(stylized_image)

        style_layer_losses = [
            tf.reduce_mean(tf.square(target-feature))
            for target, feature in zip(style_targets, style_features)
        ]

        return tf.add_n(style_layer_losses) / len(style_layer_losses)
