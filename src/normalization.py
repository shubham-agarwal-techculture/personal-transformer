"""
Normalization and Dropout for the Transformer implementation.

This module implements:
- Layer Normalization: Normalizes activations across features
- Dropout: Randomly drops units during training for regularization

These are essential components for training deep networks effectively.
"""

import numpy as np


class LayerNormalization:
    """
    Layer Normalization.

    Layer normalization normalizes the activations of a layer for each
    sample independently. Unlike batch normalization, it doesn't depend
    on the batch statistics, making it suitable for sequence models.

    The formula is:
        LayerNorm(x) = gamma * (x - mu) / sqrt(sigma^2 + epsilon) + beta

    Where:
    - mu: Mean of the features for each sample
    - sigma^2: Variance of the features for each sample
    - gamma: Learnable scale parameter
    - beta: Learnable shift parameter
    - epsilon: Small constant for numerical stability

    Layer normalization helps stabilize the hidden state dynamics in deep
    networks and is applied after each sub-layer (attention and FFN) in
    the Transformer.

    Attributes:
        d_model: Dimension of the features
        gamma: Learnable scale parameter (d_model,)
        beta: Learnable shift parameter (d_model,)
        epsilon: Small constant for numerical stability
    """

    def __init__(self, d_model, epsilon=1e-6):
        """
        Initialize layer normalization.

        Args:
            d_model: Dimension of the features
            epsilon: Small constant for numerical stability (default: 1e-6)
        """
        self.d_model = d_model
        self.epsilon = epsilon

        # Initialize learnable parameters
        # gamma: scale parameter, initialized to 1
        self.gamma = np.ones(d_model)

        # beta: shift parameter, initialized to 0
        self.beta = np.zeros(d_model)

    def forward(self, x):
        """
        Apply layer normalization to the input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Normalized tensor of shape (batch_size, seq_len, d_model)

        Example:
            >>> ln = LayerNormalization(d_model=512)
            >>> x = np.random.randn(2, 10, 512)
            >>> output = ln.forward(x)
            >>> output.shape
            (2, 10, 512)
        """
        # Compute mean along the feature dimension (last axis)
        # Shape: (batch_size, seq_len)
        mean = np.mean(x, axis=-1, keepdims=True)

        # Compute variance along the feature dimension
        # Shape: (batch_size, seq_len)
        variance = np.var(x, axis=-1, keepdims=True)

        # Compute standard deviation
        # Shape: (batch_size, seq_len)
        std = np.sqrt(variance + self.epsilon)

        # Normalize: (x - mean) / std
        # Shape: (batch_size, seq_len, d_model)
        normalized = (x - mean) / std

        # Scale and shift: gamma * normalized + beta
        # Shape: (batch_size, seq_len, d_model)
        output = self.gamma * normalized + self.beta

        return output


class Dropout:
    """
    Dropout regularization.

    Dropout randomly sets a fraction of input units to zero during training.
    This prevents the model from relying too heavily on any single feature
    and helps prevent overfitting.

    During training, each unit is dropped with probability p (kept with probability 1-p).
    During inference, all units are kept but scaled by (1-p) to maintain the expected value.

    In this educational implementation, we implement a simple version that
    can be used during forward passes. For a complete training implementation,
    you would need to handle the training/inference mode switch.

    The formula during training:
        output = mask * x / (1 - p)

    Where mask is a binary matrix with values 0 or 1.

    Attributes:
        dropout_rate: Probability of dropping a unit (default: 0.1)
        training: Whether in training mode (True) or inference mode (False)
    """

    def __init__(self, dropout_rate=0.1, training=True):
        """
        Initialize dropout.

        Args:
            dropout_rate: Probability of dropping a unit (default: 0.1)
            training: Whether in training mode (default: True)
        """
        self.dropout_rate = dropout_rate
        self.training = training

    def forward(self, x):
        """
        Apply dropout to the input.

        Args:
            x: Input tensor of any shape

        Returns:
            Output tensor with dropout applied (same shape as input)

        Example:
            >>> dropout = Dropout(dropout_rate=0.1, training=True)
            >>> x = np.random.randn(2, 10, 512)
            >>> output = dropout.forward(x)
            >>> output.shape
            (2, 10, 512)
        """
        if not self.training:
            # During inference, no dropout is applied
            return x

        # Generate dropout mask
        # Each element is kept with probability (1 - dropout_rate)
        # We scale by 1/(1-p) to maintain the expected value
        keep_prob = 1.0 - self.dropout_rate
        mask = (np.random.random(x.shape) < keep_prob).astype(float)

        # Apply mask and scale
        # Scaling ensures that the expected value of the output
        # equals the expected value of the input
        output = (mask * x) / keep_prob

        return output

    def set_training_mode(self, training):
        """
        Set the training mode.

        Args:
            training: True for training mode, False for inference mode
        """
        self.training = training
