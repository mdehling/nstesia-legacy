"""
Dumoulin Style Transfer Model.
"""

__all__ = [
    'DumoulinStyleTransfer',
]

import tensorflow as tf
import tensorflow.keras.layers as k_layers

from .. import layers as nst_layers
from .. import losses as nst_losses


class ConvBlock(k_layers.Layer):
    """
    Convolutional Block layer.

    Args:
        transpose:
                Whether to use a Conv2DTranspose or a Conv2D layer.
        activation:
                The type of activation to use, e.g., 'relu' or 'tanh'.
    """

    def __init__(self, filters=3, kernel_size=(3,3), strides=(2,2),
                 transpose=False, activation='relu', **kwargs):

        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.transpose = transpose
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

        self.norm = nst_layers.ConditionalInstanceNormalization(name="norm")
        self.act = k_layers.Activation(activation, name="act")

    def call(self, inputs):
        x = self.conv(inputs[0])
        x = self.norm((x,inputs[1]))
        return self.act(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters':      self.filters,
            'kernel_size':  self.kernel_size,
            'strides':      self.strides,
            'transpose':    self.transpose,
            'activation':   self.activation,
        })
        return config


class ResBlock(k_layers.Layer):
    """
    Residual block as defined the supplementary material to Johnson et al.
    """

    def __init__(self, filters=128, **kwargs):
        """
        Initialize a residual block.
        """
        super().__init__(**kwargs)

        self.filters = filters

        self.conv1 = k_layers.Conv2D(filters, (3,3), name='conv1')
        self.norm1 = nst_layers.ConditionalInstanceNormalization(name='norm1')
        self.relu1 = k_layers.Activation('relu', name='relu')
        self.conv2 = k_layers.Conv2D(filters, (3,3), name='conv2')
        self.norm2 = nst_layers.ConditionalInstanceNormalization(name='norm2')

        # The conv layers above use 'same' padding, so we need to crop in the
        # residual connection to produce the same size.
        self.crop = k_layers.Cropping2D(2, name='cropping')

    def call(self, inputs, training=False):
        x = self.conv1(inputs[0])
        x = self.norm1((x,inputs[1]))
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2((x,inputs[1]))

        return x + self.crop(inputs[0])

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
        })
        return config


class DumoulinStyleTransfer(tf.keras.models.Model):
    """
    Domoulin Style Transfer model.
    """

    def __init__(self, n_styles=None, filters=(32, 64, 128), tanh_factor=150.0, **kwargs):

        f1, f2, f3 = filters

        x = k_layers.Input(shape=(None,None,3))
        c = k_layers.Input(shape=(n_styles,))

        y = nst_layers.ImagePreProcessing(name='preprocess')(x)
        y = nst_layers.ReflectionPadding2D(padding=(40,40), name='rpad')(y)
    
        y = ConvBlock(filters=f1, kernel_size=(9,9), strides=(1,1),
                           name='conv_block_1')((y,c))
        y = ConvBlock(filters=f2, name='conv_block_2')((y,c))
        y = ConvBlock(filters=f3, name='conv_block_3')((y,c))

        for i in range(5):
            y = ResBlock(filters=f3, name=f'res_block_{i+1}')((y,c))
    
        y = ConvBlock(filters=f2, transpose=True, name='conv_block_4')((y,c))
        y = ConvBlock(filters=f1, transpose=True, name='conv_block_5')((y,c))
        y = ConvBlock(filters=3, kernel_size=(9,9), strides=(1,1),
                      name='conv_block_6', activation='tanh')((y,c))
    
        y = k_layers.Rescaling(scale=tanh_factor, name='rescale')(y)
        y = nst_layers.ImagePostProcessing(name='postprocess')(y)

        super().__init__(inputs=(x,c), outputs=y, **kwargs)

    @classmethod
    def from_checkpoint(cls, file):
        custom_objects = {
            'DumoulinStyleTransfer':    tf.keras.models.Model,
            'ImagePreProcessing':       nst_layers.ImagePreProcessing,
            'ImagePostProcessing':      nst_layers.ImagePostProcessing,
            'ReflectionPadding2D':      nst_layers.ReflectionPadding2D,
            'ConvBlock':                ConvBlock,
            'ResBlock':                 ResBlock,
            'Total':                    nst_losses.Total,
        }

        transfer_model = tf.keras.models.load_model(
            file, custom_objects=custom_objects
        )

        return transfer_model
