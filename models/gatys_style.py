"""
Gatys Style Model.

References:
    [1] Gatys, Ecker, Bethge - Texture Synthesis Using Convolutional Neural
        Networks, 2015.
    [2] Gatys, Ecker, Bethge - A Neural Algorithm of Artistic Style, 2015.
    [3] Johnson, Alahi, Fei-Fei - Perceptual Losses for Real-Time Style
        Transfer and Super-Resolution, 2016.
    [4] Berger, Memisevic - Incorporating Long-Range Consistency in CNN-Based
        Texture Generation, 2017.
"""

__all__ = [
    'GatysStyle',
]

import tensorflow as tf


def identity(Fl):
    """
    The identity transformation.
    """
    return Fl


def avg_gram_tensor(Fl, T1=identity, T2=identity):
    """
    Averaged Gram tensor.

    The Gram tensor $G^l$ here is defined as the inner product
    $$ G^l_{r,s} = < T1(F^l_{r,:}) | T2(F^l_{s,:}) > $$
    In Gatys' original work [1,2] the transformations T1, T2 are the identity.
    In later work by Berger, Memisevic [4] they are translation or reflection
    operators and used to calculate the crosscorrelation loss.

    Args:
        Fl:     The feature vector (output) of a layer l.
        T1, T2: Transformations to apply to the left/right.
    """
    T1_Fl = T1(Fl)
    T2_Fl = T2(Fl)

    shape = tf.shape(T1_Fl)
    height = shape[1]
    width = shape[2]
    channels = shape[3]

    return tf.linalg.einsum('bijr,bijs->brs', T1_Fl, T2_Fl) \
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
    # Gatys et al [1] initially use all layers of the vgg19 network up to and
    # include block4_pool.
    style_layers = [
        "block1_conv1",
        "block1_conv2",
        "block1_pool",
        "block2_conv1",
        "block2_conv2",
        "block2_pool",
        "block3_conv1",
        "block3_conv2",
        "block3_conv3",
        "block3_conv4",
        "block3_pool",
        "block4_conv1",
        "block4_conv2",
        "block4_conv3",
        "block4_conv4",
        "block4_pool",
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
    # TODO: Allow transformations to be specified per layer.  In [4] they are
    # applied only to the deeper layers because computing too many low-layer
    # Gram tensors is computationally expensive.
    def __init__(self, model='vgg16', transformations=None):
        """
        Initialize a GatysStyle model instance.

        Args:
            model:  A string indicating which CNN to use for feature
                    extraction, e.g., 'vgg19'.
            transformations:
                    A list of pairs of transformations to apply to the
                    left/right of the inner product computing the Gram tensor.
                    None means no transformations and is equivalent to
                    specifying [(identity,identity)].
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
        self.transformations = transformations or [(identity,identity)]

    def call(self, inputs, **kwargs):
        F = super().call(self.preprocess(inputs), **kwargs)

        # Maybe one day AutoGraph will learn to handle list comprehensions.
        G_list = []
        for T1, T2 in self.transformations:
            for Fl in F:
                G_list.append( avg_gram_tensor(Fl, T1=T1, T2=T2) )

        return tuple(G_list)
