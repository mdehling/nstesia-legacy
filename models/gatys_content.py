"""
Gatys Content Model.
"""

__all__ = [
    'GatysContent',
]

import tensorflow as tf


class GatysContent(tf.keras.models.Model):

    def __init__(self, model='vgg16'):
        """
        Initialize a GatysContent model instance.
        """
        if model == 'vgg16':
            base_model = tf.keras.applications.vgg16.VGG16(
                include_top=False, weights='imagenet'
            )
            content_layers = [
                'block3_conv3'
            ]
            preprocess = tf.keras.applications.vgg16.preprocess_input
        else:
            raise ValueError("Only model 'vgg16' supported for now.")

        base_model.trainable = False

        inputs = base_model.input
        outputs = tuple(
            base_model.get_layer(name).output for name in content_layers
        )

        super().__init__(inputs=inputs, outputs=outputs)

        self.base_model = base_model
        self.content_layers = content_layers
        self.preprocess = preprocess

    def call(self, inputs, **kwargs):
        return super().call(self.preprocess(inputs), **kwargs)
