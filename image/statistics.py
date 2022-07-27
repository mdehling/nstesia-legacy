"""
NSTesia image statistics.
"""

__all__ = [
    'match_statistics',
]

import tensorflow as tf


def image_statistics(X):
    """Compute mean and covariance matrix."""
    batch_size, height, width, channels = tf.shape(X)
    N = height * width

    mu = tf.einsum('bijc->bc', X) / tf.cast(N, tf.float32)

    mu_ = tf.reshape(mu, (batch_size,1,1,channels))
    Sigma = tf.einsum('bijc,bijd->bcd', X-mu_, X-mu_) / tf.cast(N, tf.float32)

    return mu, Sigma


def eigen_decomposition(M):
    """
    Given a real symmetric matrix M, return its orthogonal eigenvalue
    decomposition (U, Lambda).
    """
    L, U = tf.linalg.eigh(M)
    Lambda = tf.linalg.diag(L)

    return U, Lambda


def match_statistics(source_image, target_image):
    """
    Modify source image to match the mean and covariance of the target image.

    Args:
        source_image:
                The source image, i.e., the one to modify.
        target_image:
                The target image, i.e., the one whose mean and covariance
                should be matched.

    Returns:
        A modified version of the source image matching the image statistics
        of the target image.
    """
    batch_size, _, _, channels = tf.shape(source_image)
    
    mu_t, Sigma_t = image_statistics(target_image)
    U_t, Lambda_t = eigen_decomposition(Sigma_t)

    mu_s, Sigma_s = image_statistics(source_image)
    U_s, Lambda_s = eigen_decomposition(Sigma_s)

    sqrt_Sigma_t = U_t @ tf.sqrt(Lambda_t) @ tf.transpose(U_t, perm=[0,2,1])
    sqrti_Sigma_s = U_s @ tf.linalg.inv(tf.sqrt(Lambda_s)) @ tf.transpose(U_s, perm=[0,2,1])

    A = sqrt_Sigma_t @ sqrti_Sigma_s
    b = mu_t - tf.linalg.matvec(A, mu_s)

    # Use broadcasting to apply the affine transformation to each pixel
    A = tf.reshape(A, (batch_size,1,1,channels,channels))
    b = tf.reshape(b, (batch_size,1,1,channels))

    matched_source_image = tf.linalg.matvec(A, source_image) + b

    return tf.clip_by_value(matched_source_image, 0.0, 255.0)
