"""
Dumoulin Content/Style Losses.
"""

__all__ = [
    'DumoulinStyle',
    'DumoulinContent',
]

import tensorflow as tf

from .. import models as nst_models
from .. import losses as nst_losses


class DumoulinStyle(tf.keras.losses.Loss):

    def __init__(self, style_images, model='vgg16', **kwargs):
        super().__init__(**kwargs)

        self.style_images = style_images

        if isinstance(model, str):
            self.style_model = nst_models.GatysStyle(model=model)
        else:
            self.style_model = model

        # The style_model returns a tuple of tensors of shapes (1,Cl,Cl) for
        # each style image, where Cl is the number of features in layer l.
        # This concatenates the tensors of matching layers for different style
        # images.  The result is a tuple of tensors of shape (N,Cl,Cl), where
        # N is the number of style images.
        self.style_targets = tuple(
            tf.concat(features, 0)
            for features in zip(*tuple(
                self.style_model(image) for image in self.style_images
            ))
        )

    def call(self, y_true, y_pred):
        stylized_image = y_pred[0]
        style_no = y_pred[1]

        style_targets = tuple(
            tf.gather(target, style_no) for target in self.style_targets
        )
        style_features = self.style_model(stylized_image)

        style_layer_losses = [
            tf.reduce_mean(tf.square(target-feature))
            for target, feature in zip(style_targets, style_features)
        ]

        return tf.add_n(style_layer_losses) / len(style_layer_losses)


class DumoulinContent(tf.keras.losses.Loss):

    def __init__(self, model='vgg16', **kwargs):
        super().__init__(**kwargs)

        self.gatys_content_loss = nst_losses.GatysContent(model=model)

    def call(self, y_true, y_pred):
        content_image = y_true[0]
        stylized_image = y_pred[0]

        return self.gatys_content_loss(content_image, stylized_image)
