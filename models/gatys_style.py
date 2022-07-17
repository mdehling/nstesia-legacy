"""
Gatys Style Model.
"""

__all__ = [
    'GatysStyle',
]

import tensorflow as tf


def avg_gram_tensor(Fl):
    """
    Averaged Gram tensor.
    """
    Fl_shape = tf.shape(Fl)

    height = Fl_shape[1]
    width = Fl_shape[2]
    channels = Fl_shape[3]

    return tf.linalg.einsum('bijr,bijs->brs', Fl, Fl) \
        / tf.cast(height*width*channels, tf.float32)


def init_vgg16():
    base_model = tf.keras.applications.vgg16.VGG16(
        include_top=False, weights='imagenet'
    )
    # Layers as used by Johnson et al [3].
    style_layers = [
        'block1_conv2',
        'block2_conv2',
        'block3_conv3',
        'block4_conv3',
    ]
    preprocess = tf.keras.applications.vgg16.preprocess_input

    return base_model, style_layers, preprocess


def init_vgg19():
    base_model = tf.keras.applications.vgg19.VGG19(
        include_top=False, weights='imagenet'
    )
    # Layers as used by Gatys et al [1,2].
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    preprocess = tf.keras.applications.vgg19.preprocess_input

    return base_model, style_layers, preprocess


init_model = {
    'vgg16':    init_vgg16,
    'vgg19':    init_vgg19,
}


class GatysStyle(tf.keras.models.Model):

    # TODO: Take style_layers and preprocess as parameters and allow model to
    # be a keras model instead of a string.
    def __init__(self, model='vgg16'):
        """
        Initialize a GatysStyle model instance.
        """
        try:
            base_model, style_layers, preprocess = init_model[model]()
        except KeyError:
            raise ValueError(f"Model '{model}' not supported.")

        base_model.trainable = False

        inputs = base_model.input
        outputs = tuple(
            base_model.get_layer(name).output for name in style_layers
        )

        super().__init__(inputs=inputs, outputs=outputs)

        self.base_model = base_model
        self.style_layers = style_layers
        self.preprocess = preprocess

    def call(self, inputs, **kwargs):
        F = super().call(self.preprocess(inputs), **kwargs)

        # Maybe one day AutoGraph will learn to handle list comprehensions...
        G_list = []
        for Fl in F:
            G_list.append( avg_gram_tensor(Fl) )

        return tuple(G_list)
