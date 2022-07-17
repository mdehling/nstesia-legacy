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


class GatysStyle(tf.keras.models.Model):

    def __init__(self, model='vgg16'):
        """
        Initialize a GatysStyle model instance.
        """
        if model == 'vgg16':
            base_model = tf.keras.applications.vgg16.VGG16(
                include_top=False, weights='imagenet'
            )
            style_layers = [
                'block1_conv2',
                'block2_conv2',
                'block3_conv3',
                'block4_conv3'
            ]
            preprocess = tf.keras.applications.vgg16.preprocess_input
        else:
            raise ValueError("Only model 'vgg16' supported for now.")

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
