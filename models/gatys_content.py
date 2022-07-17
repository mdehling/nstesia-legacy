"""
Gatys Content Model.
"""

__all__ = [
    'GatysContent',
]

import tensorflow as tf


def init_vgg16():
    base_model = tf.keras.applications.vgg16.VGG16(
        include_top=False, weights='imagenet'
    )
    # Layers as used by Johnson et al [2].
    content_layers = [
        'block3_conv3',
    ]
    preprocess = tf.keras.applications.vgg16.preprocess_input

    return base_model, content_layers, preprocess


def init_vgg19():
    base_model = tf.keras.applications.vgg19.VGG19(
        include_top=False, weights='imagenet'
    )
    # Layers as used by Gatys et al [1].
    content_layers = [
        "block5_conv2",
    ]
    preprocess = tf.keras.applications.vgg19.preprocess_input

    return base_model, content_layers, preprocess


init_model = {
    'vgg16':    init_vgg16,
    'vgg19':    init_vgg19,
}


class GatysContent(tf.keras.models.Model):

    # TODO: Take content_layers and preprocess as parameters and allow model
    # to be a keras model instead of a string.
    def __init__(self, model='vgg16'):
        """
        Initialize a GatysContent model instance.
        """
        try:
            base_model, content_layers, preprocess = init_model[model]()
        except KeyError:
            raise ValueError(f"Model '{model}' not supported.")

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
