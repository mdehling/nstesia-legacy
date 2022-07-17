"""
Gatys Content Loss.
"""

__all__ = [
    'GatysContent',
]

import tensorflow as tf
import nstesia.models as nst_models


class GatysContent(tf.keras.losses.Loss):

    def __init__(self, model='vgg16', **kwargs):
        super().__init__(**kwargs)
        
        if isinstance(model, str):
            self.content_model = nst_models.GatysContent(model=model)
        else:
            self.content_model = model

    def call(self, content_image, stylized_image):
        content_targets = self.content_model(content_image)
        content_features = self.content_model(stylized_image)

        content_layer_losses = [
            tf.reduce_mean(tf.square(target-feature))
            for target, feature in zip(content_targets, content_features)
        ]

        return tf.add_n(content_layer_losses) / len(content_layer_losses)
