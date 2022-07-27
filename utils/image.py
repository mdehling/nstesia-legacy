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


def load_images(files, min_dim=None, target_size=None):
    """
    Load images from files.

    Args:
        files:  A filename or list of filenames to load.
        min_dim:
                Minimum desired dimension (height or width).  The image is
                resized so that its shorter side fits this dimension while
                keeping the aspect ratio.  Takes precedence over target_size.
        target_size:
                A tuple (height,width) to resize each image to or None to keep
                the original size.  In the latter case, the height and width
                of all loaded images must match.

    Returns:
        A 4-D tensor of shape (N, height, width, channels) of the N images.
    """
    if not isinstance(files, list):
        files = [ files ]

    image_tensors = []
    for file in files:
        image = tf.keras.utils.load_img(file)
        image_tensor = tf.keras.utils.img_to_array(image, dtype="float32")

        if min_dim is not None:
            image_height, image_width, _ = tf.shape(image_tensor).numpy()
            short_dim = min(image_height, image_width)
            target_size = ( int( image_height/short_dim * min_dim ),
                            int( image_width /short_dim * min_dim ) )

        if target_size is not None:
            image_tensor = tf.image.resize(image_tensor, target_size)

        image_tensors.append(image_tensor)

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


def show_images(image_tensors, titles=None, n_cols=4, width=20):
    """
    Show images on a grid.

    Args:
        image_tensors:
                A 4-D tensor of shape (N, height, width, channels) or a list
                of such.  If a list, can contain None entries to leave a
                space in the grid blank.
        titles: Either None or a list of titles - one for each image.
        n_cols: The number of columns in the grid.
        width:  The total width of the grid.
    """
    if not isinstance(image_tensors, list):
        image_tensors = [ image_tensors ]

    # list of 3-D tensors of shape (height,width,channels)
    images = [
        image   for image_tensor in image_tensors
                for image in ( tf.unstack(image_tensor, axis=0)
                               if image_tensor is not None else [ None ] )
    ]

    N = len(images)

    image_heights = [ tf.shape(image).numpy()[0] for image in images
                                                 if image is not None ]
    image_widths  = [ tf.shape(image).numpy()[1] for image in images
                                                 if image is not None ]

    max_image_height = max(image_heights)
    max_image_width = max(image_widths)

    n_rows = (N-1) // n_cols + 1
    height = n_rows * (width/n_cols) * (max_image_height/max_image_width)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(width,height))
    [ ax.set_axis_off() for ax in axs.ravel() ]

    for idx, ax in zip(range(N),axs.ravel()):
        if images[idx] is None:
            continue

        ax.imshow(tf.keras.utils.array_to_img(images[idx]))
        if titles is not None:
            ax.set_title(titles[idx])
