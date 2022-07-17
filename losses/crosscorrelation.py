"""
Berger/Memisevic CrossCorrelation Loss.

References:
    [1] Berger, Memisevic - Incorporating Long-Range Consistency in CNN-Based
        Texture Generation, 2017.
"""

__all__ = [
    'CrossCorrelation',
]

import tensorflow as tf
#import nstesia.models as nst_models
from .. import models as nst_models


class CrossCorrelation(tf.keras.losses.Loss):

    def __init__(self, style_image, transformations, model='vgg16', **kwargs):
        super().__init__(**kwargs)

        self.style_image = style_image
        self.transformations = transformations

        if isinstance(model, str):
            self.style_model = nst_models.GatysStyle(
                model=model, transformations=transformations
            )
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
