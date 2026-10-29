"""
Core utility functions for the Transformer implementation.

This module provides fundamental mathematical operations used throughout
the transformer architecture, with a focus on educational clarity.
"""

import numpy as np


def softmax(x, axis=-1):
    """
    Compute the softmax activation function with numerical stability.

    Softmax converts a vector of real numbers into a probability distribution:
    softmax(x_i) = exp(x_i) / sum(exp(x_j))

    To avoid numerical overflow/underflow, we subtract the max value before
    exponentiation: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Args:
        x: Input array of any shape
        axis: Axis along which to compute softmax (default: -1, last axis)

    Returns:
        Array of same shape as x with softmax applied along specified axis

    Example:
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> softmax(x)
        array([0.09003057, 0.24472847, 0.66524096])
    """
    # Subtract max for numerical stability (prevents overflow)
    # This doesn't change the result because exp(x - max) / sum(exp(x - max))
    # = exp(x) / exp(max) / sum(exp(x) / exp(max))
    # = exp(x) / sum(exp(x))
    x_shifted = x - np.max(x, axis=axis, keepdims=True)

    # Compute exponentials
    exp_x = np.exp(x_shifted)

    # Normalize by sum along the specified axis
    # keepdims=True ensures we can broadcast correctly
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x):
    """
    Rectified Linear Unit activation function.

    ReLU(x) = max(0, x)

    This is a simple non-linear activation that sets all negative values to zero.
    It's computationally efficient and helps mitigate the vanishing gradient problem.

    Args:
        x: Input array of any shape

    Returns:
        Array of same shape with ReLU applied element-wise

    Example:
        >>> x = np.array([-1.0, 0.0, 1.0])
        >>> relu(x)
        array([0., 0., 1.])
    """
    return np.maximum(0, x)


def gelu(x):
    """
    Gaussian Error Linear Unit activation function.

    GELU(x) = x * Φ(x) where Φ(x) is the standard normal cumulative distribution function.

    This is the activation function used in modern transformers (BERT, GPT, etc.).
    It's smoother than ReLU and often performs better in deep networks.

    The exact implementation uses the approximation:
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    Args:
        x: Input array of any shape

    Returns:
        Array of same shape with GELU applied element-wise

    Reference:
        "Gaussian Error Linear Units (GELUs)" (Hendrycks & Gimpel, 2016)
    """
    # Constants for the GELU approximation
    sqrt_2_over_pi = np.sqrt(2 / np.pi)
    coeff = 0.044715

    # Compute the tanh approximation
    # This approximates the cumulative distribution function of the standard normal
    tanh_arg = sqrt_2_over_pi * (x + coeff * x ** 3)
    cdf = 0.5 * (1 + np.tanh(tanh_arg))

    # GELU = x * CDF(x)
    return x * cdf


def get_positional_encoding(seq_len, d_model):
    """
    Generate sinusoidal positional encodings.

    Positional encodings allow the model to understand the order of tokens
    in the sequence. They use sine and cosine functions of different frequencies:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Where:
    - pos is the position in the sequence
    - i is the dimension index
    - d_model is the embedding dimension

    This choice allows the model to easily learn to attend by relative positions,
    since for any fixed offset k, PE(pos+k) can be represented as a linear function
    of PE(pos).

    Args:
        seq_len: Length of the sequence (number of positions)
        d_model: Dimension of the model (embedding size)

    Returns:
        Array of shape (seq_len, d_model) containing positional encodings

    Example:
        >>> pe = get_positional_encoding(seq_len=10, d_model=512)
        >>> pe.shape
        (10, 512)
    """
    # Create an array of positions: [0, 1, 2, ..., seq_len-1]
    positions = np.arange(seq_len)[:, np.newaxis]  # Shape: (seq_len, 1)

    # Create an array of dimensions: [0, 1, 2, ..., d_model-1]
    dims = np.arange(d_model)[np.newaxis, :]  # Shape: (1, d_model)

    # Compute the divisor: 10000^(2i/d_model)
    # We only use even dimensions (2i), so we divide by 2
    # Shape: (1, d_model)
    div_term = np.power(10000, 2 * dims / d_model)

    # Compute the argument for sin/cos: pos / divisor
    # Shape: (seq_len, d_model)
    angles = positions / div_term

    # Apply sin to even dimensions, cos to odd dimensions
    # We use this by computing sin(angles) and cos(angles) and interleaving them
    pe = np.zeros((seq_len, d_model))

    # Even indices (0, 2, 4, ...) get sin
    pe[:, 0::2] = np.sin(angles[:, 0::2])

    # Odd indices (1, 3, 5, ...) get cos
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe


def initialize_weights(shape, method='xavier'):
    """
    Initialize weights using various methods.

    Proper weight initialization is crucial for training deep networks.
    Poor initialization can lead to vanishing or exploding gradients.

    Args:
        shape: Tuple specifying the shape of the weight matrix
        method: Initialization method ('xavier', 'he', or 'normal')

    Returns:
        Initialized weight array of specified shape

    Methods:
        - 'xavier': Xavier/Glorot initialization (good for tanh/sigmoid)
          Scale: sqrt(2 / (fan_in + fan_out))
        - 'he': He initialization (good for ReLU and variants)
          Scale: sqrt(2 / fan_in)
        - 'normal': Standard normal distribution
    """
    if method == 'xavier':
        # Xavier/Glorot initialization
        # Designed to keep the variance of activations constant across layers
        fan_in, fan_out = shape[0], shape[1]
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(*shape) * scale

    elif method == 'he':
        # He initialization
        # Designed for ReLU activations, accounts for the fact that
        # half the ReLU outputs are zero
        fan_in = shape[0]
        scale = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape) * scale

    elif method == 'normal':
        # Standard normal initialization
        return np.random.randn(*shape) * 0.01

    else:
        raise ValueError(f"Unknown initialization method: {method}")


def initialize_bias(shape):
    """
    Initialize bias terms.

    Biases are typically initialized to zero or small random values.
    We'll use zeros for simplicity.

    Args:
        shape: Tuple specifying the shape of the bias vector

    Returns:
        Zero-initialized bias array of specified shape
    """
    return np.zeros(shape)
