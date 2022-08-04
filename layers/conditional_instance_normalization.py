"""
Conditional Instance Normalization Layer.
"""

__all__ = [
    'ConditionalInstanceNormalization',
]

import tensorflow as tf


class ConditionalInstanceNormalization(tf.keras.layers.Layer):
    """
    Conditional Instance Normalization layer.

    Args:
        axis:   The axis to normalize over.  Not yet implemented.
        epsilon:
                Small float value added to the variance to avoid dividing by
                zero.

    Input shape:
        A 2-tuple (x,c) of the feature tensor x to normalize and the
        condition c to use.

    Output shape:
        The normalized feature tensor.

    References:
        [1] Dumoulin, Shlens, Kudlur - A Learned Representation for Artistic
            Style, 2017.
    """

    def __init__(self, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)

        self.epsilon = epsilon

    def build(self, input_shape):

        if not isinstance(input_shape, tuple):
            raise ValueError("Expected tuple input")

        if input_shape[0].rank != 4 or input_shape[1].rank != 2:
            raise ValueError("Expected tuple of tensors of ranks (4,2).")

        C = input_shape[0][-1]
        N = input_shape[1][1]

        self.gamma = self.add_weight(name='gamma', shape=(1,1,C,N),
            initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=(1,1,C,N),
            initializer='zeros', trainable=True)

    def call(self, inputs, training=None):
        x = inputs[0]                                           # (B,H,W,C)
        c = inputs[1]                                           # (B,N)

        mu, var = tf.nn.moments(x, axes=[1,2], keepdims=True)   # (B,1,1,C)

        x_norm = (x - mu) / tf.sqrt(var + self.epsilon)         # (B,H,W,C)

        c_gamma = tf.tensordot(c, self.gamma, [1,3])            # (B,1,1,C)
        c_beta = tf.tensordot(c, self.beta, [1,3])              # (B,1,1,C)

        return c_gamma * x_norm + c_beta                        # (B,H,W,C)

    def get_config(self):
        config = super().get_config()
        config.update({
            'epsilon':  self.epsilon,
        })
        return config
