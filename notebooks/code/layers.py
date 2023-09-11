import numpy as np


class Layer_Dense:
    """A class representing a dense or fully-connected layer in a neural network."""

    def __init__(self, n_inputs: int, n_neurons: int):
        # Initialize weights and biases:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray | list):
        """A method for calculating the output values from a forward pass."""
        # Calculate output values from inputs, weights, and biases:
        self.output = np.dot(inputs, self.weights) + self.biases
