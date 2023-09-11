import numpy as np


class Activation_ReLU:
    """A class representing a rectified linear activation function."""

    def forward(self, inputs):
        """A method for calculating the output values from a forward pass."""
        self.output = np.maximum(0, inputs)
