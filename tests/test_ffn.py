"""
Unit tests for feed-forward networks.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import src module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ffn import PositionWiseFFN


def test_position_wise_ffn():
    """Test position-wise feed-forward network."""
    print("Testing PositionWiseFFN...")

    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048

    ffn = PositionWiseFFN(d_model=d_model, d_ff=d_ff)

    # Create dummy input
    x = np.random.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = ffn.forward(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("  [PASS] Output shape correct")

    # Check that output is different from input (non-identity transformation)
    assert not np.allclose(output, x), "FFN should transform the input"
    print("  [PASS] Output is transformed")


def test_position_wise_ffn_relu():
    """Test position-wise feed-forward network with ReLU activation."""
    print("Testing PositionWiseFFN with ReLU...")

    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048

    ffn = PositionWiseFFN(d_model=d_model, d_ff=d_ff, activation='relu')

    # Create dummy input
    x = np.random.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = ffn.forward(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("  [PASS] Output shape correct with ReLU")


def test_position_wise_ffn_gelu():
    """Test position-wise feed-forward network with GELU activation."""
    print("Testing PositionWiseFFN with GELU...")

    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048

    ffn = PositionWiseFFN(d_model=d_model, d_ff=d_ff, activation='gelu')

    # Create dummy input
    x = np.random.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = ffn.forward(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("  [PASS] Output shape correct with GELU")


def test_position_wise_ffn_expansion_ratio():
    """Test that FFN uses the specified expansion ratio."""
    print("Testing FFN expansion ratio...")

    d_model = 512
    d_ff = 2048  # 4x expansion

    ffn = PositionWiseFFN(d_model=d_model, d_ff=d_ff)

    # Check that hidden layer has correct dimension
    assert ffn.W1.shape == (d_model, d_ff), f"W1 shape should be ({d_model}, {d_ff})"
    assert ffn.W2.shape == (d_ff, d_model), f"W2 shape should be ({d_ff}, {d_model})"
    print("  [PASS] Expansion ratio correct")


if __name__ == "__main__":
    print("\n=== Running Feed-Forward Network Tests ===\n")

    test_position_wise_ffn()
    test_position_wise_ffn_relu()
    test_position_wise_ffn_gelu()
    test_position_wise_ffn_expansion_ratio()

    print("\n=== All FFN Tests Passed! ===\n")
