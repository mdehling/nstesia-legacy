"""
Iterative Image Generator.
"""

__all__ = [
    'ImageGenerator',
]

import tensorflow as tf


class ImageGenerator:

    def __init__(self, image, optimizer, loss):
        """
        Initialize an ImageGenerator instance.

        Args:
            image:  The initial image.
            optimizer:
                    The optimizer to use.
            loss:   The loss function.
        """
        self.image = tf.Variable(image)
        self.optimizer = optimizer
        self.loss = loss

        self.images = []
        self.losses = []

    @tf.function
    def compute_gradients(self):
        with tf.GradientTape() as tape:
            loss = self.loss(self.image, self.image)

        return loss, tape.gradient(loss, self.image)

    def __call__(self, steps=1, iterations_per_step=100):
        """
        Compute steps * iterations_per_step iterations of loss minimization
        using the provided optimizer.
        """
        for n in range(steps):
            for m in range(iterations_per_step):
                loss, grad = self.compute_gradients()

                self.optimizer.apply_gradients([(grad, self.image)])
                self.image.assign(tf.clip_by_value(self.image, 0.0, 255.0))

                self.losses.append(loss)

            # save the result at the end of each step
            self.images.append( tf.identity(self.image) )
