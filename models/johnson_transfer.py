"""
Johnson Style Transfer Model.
"""

__all__ = [
    'JohnsonStyleTransfer',
]

import tensorflow as tf
import tensorflow.keras.layers as k_layers
import tensorflow_addons as tfa

from .. import layers as nst_layers
from .. import losses as nst_losses


normalization_layer = {
    'none':     k_layers.Layer,
    'batch':    k_layers.BatchNormalization,
    'instance': tfa.layers.InstanceNormalization,
}

ERRMSG_UNKNOWN_NORMALIZE_PARAMETER = (
    "Parameter 'normalize' must be one of 'none', 'batch', or 'instance'."
)


imagenet_means = [123.680, 116.779, 103.939]

def preprocess(x):
    return x - tf.constant(imagenet_means, shape=(1,1,1,3))

def postprocess(x):
    x = x + tf.constant(imagenet_means, shape=(1,1,1,3))
    return tf.clip_by_value(x, 0.0, 255.0)


class ConvBlock(k_layers.Layer):
    """
    Convolutional block.
    """

    def __init__(self, filters=3, kernel_size=(3,3), strides=(2,2),
                 transpose=False, normalize='batch', activation='relu',
                 **kwargs):
        """
        Initialize a convolutional block.

        Args:
            transpose:
                    Whether to use a Conv2DTranspose or a Conv2D layer.
            normalize:
                    The type of normalization layer to use: 'none', 'batch',
                    or 'instance'.
            activation:
                    The type of activation to use, e.g., 'relu' or 'tanh'.
        """
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.transpose = transpose
        self.normalize = normalize
        self.activation = activation

        if transpose is True:
            self.conv = k_layers.Conv2DTranspose(
                filters, kernel_size,
                strides=strides, padding='same', name="conv_t"
            )
        else:
            self.conv = k_layers.Conv2D(
                filters, kernel_size,
                strides=strides, padding='same', name="conv"
            )

        try:
            self.norm = normalization_layer[normalize](name="norm")
        except KeyError:
            raise ValueError(ERRMSG_UNKNOWN_NORMALIZE_PARAMETER)

        self.act = k_layers.Activation(activation, name="act")

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.norm(x, training=training)
        return self.act(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters':      self.filters,
            'kernel_size':  self.kernel_size,
            'strides':      self.strides,
            'transpose':    self.transpose,
            'normalize':    self.normalize,
            'activation':   self.activation,
        })
        return config


class ResBlock(k_layers.Layer):
    """
    Residual block as defined the supplementary material to Johnson et al.
    """

    def __init__(self, filters=128, normalize='batch', **kwargs):
        """
        Initialize a residual block.
        """
        super().__init__(**kwargs)

        self.filters = filters
        self.normalize = normalize

        self.conv1 = k_layers.Conv2D(filters, (3,3), name='conv1')
        self.relu1 = k_layers.Activation('relu', name='relu')
        self.conv2 = k_layers.Conv2D(filters, (3,3), name='conv2')

        try:
            self.norm1 = normalization_layer[normalize](name='norm1')
            self.norm2 = normalization_layer[normalize](name='norm2')
        except KeyError:
            raise ValueError(ERRMSG_UNKNOWN_NORMALIZE_PARAMETER)

        # The conv layers above use 'same' padding, so we need to crop in the
        # residual connection to produce the same size.
        self.crop = k_layers.Cropping2D(2, name='cropping')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.norm1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x, training=training)

        return x + self.crop(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'normalize': self.normalize
        })
        return config


class JohnsonStyleTransfer(tf.keras.models.Sequential):

    def __init__(self, normalize='batch', filters=(32, 64, 128),
                 tanh_factor=150.0, **kwargs):
        """
        Create an instance of a Johnson Style Transfer model.
        """
        super().__init__(**kwargs)

        f1, f2, f3 = filters

        # Take arbitrary-size RGB images as input.
        self.add( k_layers.InputLayer(input_shape=(None,None,3)) )
    
        self.add( k_layers.Lambda(preprocess, name='preprocess') )

        self.add(
            nst_layers.ReflectionPadding2D(padding=(40,40), name='rpad')
        )
    
        self.add(
            ConvBlock(filters=f1, kernel_size=(9,9), strides=(1,1),
                      normalize=normalize, name='conv_block_1')
        )
        self.add(
            ConvBlock(filters=f2, normalize=normalize, name='conv_block_2')
        )
        self.add(
            ConvBlock(filters=f3, normalize=normalize, name='conv_block_3')
        )
    
        for i in range(5):
            self.add(
                ResBlock(filters=f3, normalize=normalize,
                         name=f'res_block_{i+1}')
            )
    
        self.add(
            ConvBlock(filters=f2, transpose=True,
                      normalize=normalize, name='conv_block_4')
        )
        self.add(
            ConvBlock(filters=f1, transpose=True,
                      normalize=normalize, name='conv_block_5')
        )
        self.add(
            ConvBlock(filters=3, kernel_size=(9,9), strides=(1,1),
                      normalize='none', name='conv_block_6',
                      activation='tanh')
        )
    
        self.add(
            k_layers.Rescaling(scale=tanh_factor, name='rescale'),
        )
    
        self.add( k_layers.Lambda(postprocess, 'postprocess') )

    @classmethod
    def from_checkpoint(cls, file):
        custom_objects = {
            'JohnsonStyleTransfer': tf.keras.models.Sequential,
            'ReflectionPadding2D':  nst_layers.ReflectionPadding2D,
            'ConvBlock':            ConvBlock,
            'ResBlock':             ResBlock,
            'Total':                nst_losses.Total,
        }

        transfer_model = tf.keras.models.load_model(
            file, custom_objects=custom_objects
        )

        return transfer_model
