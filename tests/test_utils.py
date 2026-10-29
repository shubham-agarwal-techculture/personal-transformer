"""
Unit tests for utility functions.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import src module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import softmax, relu, gelu, get_positional_encoding, initialize_weights, initialize_bias


def test_softmax():
    """Test softmax function."""
    print("Testing softmax...")

    # Test 1D array
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)
    assert np.allclose(result.sum(), 1.0), "Softmax should sum to 1"
    assert result.shape == x.shape, "Output shape should match input"
    print("  [PASS] 1D array test passed")

    # Test 2D array
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = softmax(x, axis=-1)
    assert np.allclose(result.sum(axis=-1), np.array([1.0, 1.0])), "Each row should sum to 1"
    print("  [PASS] 2D array test passed")

    # Test numerical stability
    x = np.array([1000.0, 1001.0, 1002.0])
    result = softmax(x)
    assert not np.any(np.isnan(result)), "Should handle large values without NaN"
    assert not np.any(np.isinf(result)), "Should handle large values without Inf"
    print("  [PASS] Numerical stability test passed")


def test_relu():
    """Test ReLU activation."""
    print("Testing relu...")

    x = np.array([-1.0, 0.0, 1.0])
    result = relu(x)
    expected = np.array([0.0, 0.0, 1.0])
    assert np.allclose(result, expected), "ReLU should zero out negative values"
    print("  [PASS] ReLU test passed")


def test_gelu():
    """Test GELU activation."""
    print("Testing gelu...")

    # Test basic properties
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = gelu(x)

    # GELU should be approximately 0 for large negative values
    assert result[0] < 0.01, "GELU should be near 0 for large negative values"

    # GELU should be approximately equal to input for large positive values
    assert result[-1] > 1.9, "GELU should be near input for large positive values"

    # GELU(0) should be exactly 0
    assert result[2] == 0.0, "GELU(0) should be 0"

    print("  [PASS] GELU test passed")


def test_positional_encoding():
    """Test positional encoding generation."""
    print("Testing positional encoding...")

    seq_len = 10
    d_model = 512

    pe = get_positional_encoding(seq_len, d_model)

    # Check shape
    assert pe.shape == (seq_len, d_model), f"Shape should be ({seq_len}, {d_model})"

    # Check that values are bounded
    assert np.all(pe >= -1.0) and np.all(pe <= 1.0), "Values should be in [-1, 1]"

    # Check that different positions have different encodings
    assert not np.allclose(pe[0], pe[1]), "Different positions should have different encodings"

    print("  [PASS] Positional encoding test passed")


def test_weight_initialization():
    """Test weight initialization."""
    print("Testing weight initialization...")

    shape = (512, 512)

    # Test Xavier initialization
    weights = initialize_weights(shape, method='xavier')
    assert weights.shape == shape, "Shape should match"
    assert not np.allclose(weights, 0), "Weights should not be all zeros"
    print("  [PASS] Xavier initialization test passed")

    # Test He initialization
    weights = initialize_weights(shape, method='he')
    assert weights.shape == shape, "Shape should match"
    assert not np.allclose(weights, 0), "Weights should not be all zeros"
    print("  [PASS] He initialization test passed")

    # Test normal initialization
    weights = initialize_weights(shape, method='normal')
    assert weights.shape == shape, "Shape should match"
    assert np.allclose(weights.std(), 0.01, atol=0.005), "Should have small std dev"
    print("  [PASS] Normal initialization test passed")


def test_bias_initialization():
    """Test bias initialization."""
    print("Testing bias initialization...")

    shape = (512,)
    bias = initialize_bias(shape)

    assert bias.shape == shape, "Shape should match"
    assert np.allclose(bias, 0), "Bias should be initialized to zeros"
    print("  [PASS] Bias initialization test passed")


if __name__ == "__main__":
    print("\n=== Running Utility Function Tests ===\n")

    test_softmax()
    test_relu()
    test_gelu()
    test_positional_encoding()
    test_weight_initialization()
    test_bias_initialization()

    print("\n=== All Tests Passed! ===\n")
