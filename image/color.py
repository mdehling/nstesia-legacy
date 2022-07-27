"""
NSTesia image color operations.
"""

__all__ = [
    'rgb_to_y_cc',
    'y_cc_to_rgb',
]

import tensorflow as tf


A_RGB2YCbCr = tf.constant([[ 0.2126,  0.7152,  0.0722],
                           [-0.1146, -0.3854,  0.5000],
                           [ 0.5000, -0.4542, -0.0458]], dtype=tf.float32)

A_YCbCr2RGB = tf.constant([[1.0,  0.0000,  1.5748],
                           [1.0, -0.1873, -0.4681],
                           [1.0,  1.8556,  0.0000]], dtype=tf.float32)


def rgb_to_y_cc(image_rgb):
    """
    Convert RGB image tensor to separate luminance Y and chrominance CbCr
    tensors.

    Args:
        image_rgb:  A 4-D image tensor of shape (batch_size,height,width,3).

    Returns:
        A pair of tensors of shapes (batch_size,height,width,1) and
        (batch_size,height,width,2), the first being the luminance, the second
        the chrominance tensor.
    """
    # Apply the linear color transformation to each pixels RGB values.
    image_ycc = tf.einsum('dc,byxc->byxd', A_RGB2YCbCr, image_rgb)

    return image_ycc[:,:,:,0:1], image_ycc[:,:,:,1:3]


def y_cc_to_rgb(image_y, image_cc=None):
    """
    Combine a 1-channel luminance and (optionally) a 2-channel chrominance
    tensor into a 3-channel RGB image tensor.

    Args:
        image_y:    The luminance tensor of shape (batch_size,height,width,1).
        image_cc:   The chrominance tensor of shape
                    (batch_size,height,width,2) or None.  If None, the
                    chrominance tensor is assumed to be all zero.

    Returns:
        An RGB image tensor of shape (batch_size,height,width,3) representing
        the combination of the luminance and chrominance tensors.
    """
    if image_cc is None:
        zeros = tf.zeros(tf.shape(image_y), dtype=tf.float32)
        image_cc = tf.concat([zeros, zeros], axis=-1)

    image_ycc = tf.concat([image_y, image_cc], axis=-1)

    return tf.einsum('dc,byxc->byxd', A_YCbCr2RGB, image_ycc)
