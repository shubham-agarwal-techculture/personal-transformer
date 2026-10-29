"""
Feed-Forward Networks for the Transformer implementation.

This module implements the position-wise feed-forward network used in both
encoder and decoder layers of the Transformer.

The FFN consists of two linear transformations with a non-linear activation
in between. It's applied to each position separately and identically.
"""

import numpy as np
from .utils import gelu, relu, initialize_weights, initialize_bias


class PositionWiseFFN:
    """
    Position-wise Feed-Forward Network.

    This is a two-layer feed-forward network applied to each position
    independently and identically. It consists of:

    1. A linear transformation that expands the dimension (d_model -> d_ff)
    2. A non-linear activation function (ReLU or GELU)
    3. A linear transformation that projects back to d_model (d_ff -> d_model)

    The formula is:
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    Or with GELU:
        FFN(x) = GELU(xW_1 + b_1)W_2 + b_2

    Where:
    - d_model: Input and output dimension
    - d_ff: Hidden layer dimension (typically 4x d_model)
    - W_1, b_1: Weights and bias for first layer
    - W_2, b_2: Weights and bias for second layer

    The expansion to d_ff allows the model to capture more complex patterns
    before projecting back to the original dimension.

    Attributes:
        d_model: Input and output dimension
        d_ff: Hidden layer dimension
        W1: First layer weights (d_model, d_ff)
        b1: First layer bias (d_ff,)
        W2: Second layer weights (d_ff, d_model)
        b2: Second layer bias (d_model,)
        activation: Activation function ('relu' or 'gelu')
    """

    def __init__(self, d_model, d_ff, activation='gelu'):
        """
        Initialize the position-wise feed-forward network.

        Args:
            d_model: Input and output dimension
            d_ff: Hidden layer dimension (typically 4x d_model)
            activation: Activation function to use ('relu' or 'gelu')
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation

        # Initialize weights
        # First layer: d_model -> d_ff (expansion)
        self.W1 = initialize_weights((d_model, d_ff), method='xavier')
        self.b1 = initialize_bias(d_ff)

        # Second layer: d_ff -> d_model (projection back)
        self.W2 = initialize_weights((d_ff, d_model), method='xavier')
        self.b2 = initialize_bias(d_model)

    def forward(self, x):
        """
        Apply the feed-forward network to the input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)

        Example:
            >>> ffn = PositionWiseFFN(d_model=512, d_ff=2048)
            >>> x = np.random.randn(2, 10, 512)
            >>> output = ffn.forward(x)
            >>> output.shape
            (2, 10, 512)
        """
        # First linear transformation: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        # x @ W1 + b1
        hidden = np.dot(x, self.W1) + self.b1

        # Apply activation function
        if self.activation == 'relu':
            hidden = relu(hidden)
        elif self.activation == 'gelu':
            hidden = gelu(hidden)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Second linear transformation: (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        # hidden @ W2 + b2
        output = np.dot(hidden, self.W2) + self.b2

        return output
