"""
NSTesia image utility functions.
"""

__all__ = [
    'load_images',
    'save_images',
    'show_images',
]

import tensorflow as tf
import matplotlib.pyplot as plt


def load_images(files, target_size=(256,256)):
    """
    Load images from files.

    Args:
        files:  A filename or list of filenames to load.
        target_size:
                A tuple (height,width) to resize each image to or None to keep
                the original size.  In the latter case, the height and width of
                all loaded images must match.

    Returns:
        A 4-D tensor of shape (N, height, width, channels) of the N images.
    """
    if not isinstance(files, list):
        files = [ files ]

    image_tensors = []
    for file in files:
        image = tf.keras.utils.load_img(file, target_size=target_size,
                                        interpolation='bicubic')
        image_tensors.append(
            tf.keras.utils.img_to_array(image, dtype="float32")
        )

    return tf.stack(image_tensors, axis=0)


def save_images(images, path):
    """
    Save images to files.

    Args:
        images: A 4-D tensor of shape (N, height, width, channels) or a list of
                such with matching (height, width, channels).
        path:   Where to save the file.  When saving a single image this can be
                simply a file name.  In general the string is interpreted as a
                format string and the variable i (range 0..N-1) can be used in
                format fields.
    """
    if isinstance(images, list):
        images = tf.concat(images, axis=0)

    N, _, _, _ = tf.shape(images).numpy()

    for i in range(N):
        tf.keras.utils.save_img(path.format(i=i), images[i])


def show_images(images, title=None, n_cols=4, width=20):
    """
    Show images on a grid.

    Args:
        images: A 4-D tensor of shape (N, height, width, channels) or a list of
                such with matching (height, width, channels).
        title:  Either None or a format string to be used as the title for each
                image.
        n_cols: The number of columns in the grid.
        width:  The total width of the grid.
    """
    if isinstance(images, list):
        images = tf.concat(images, axis=0)

    N, image_height, image_width, channels = tf.shape(images).numpy()

    n_rows = (N-1) // n_cols + 1
    height = n_rows * (width/n_cols) * (image_height/image_width)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(width,height))
    [ ax.set_axis_off() for ax in axs.ravel() ]

    for idx, ax in zip(range(N),axs.ravel()):
        ax.imshow(tf.keras.utils.array_to_img(images[idx]))
        if title is not None:
            ax.set_title(title.format(i=idx))
