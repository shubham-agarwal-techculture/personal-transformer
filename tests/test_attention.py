"""
Unit tests for attention mechanisms.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import src module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.attention import ScaledDotProductAttention, MultiHeadAttention


def test_scaled_dot_product_attention():
    """Test scaled dot-product attention."""
    print("Testing ScaledDotProductAttention...")

    batch_size = 2
    seq_len = 10
    d_k = 64

    attention = ScaledDotProductAttention(d_k=d_k)

    # Create dummy inputs
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)

    # Forward pass
    output, weights = attention.forward(Q, K, V)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_k), f"Expected shape {(batch_size, seq_len, d_k)}, got {output.shape}"
    print("  [PASS] Output shape correct")

    # Check attention weights shape
    assert weights.shape == (batch_size, seq_len, seq_len), f"Expected shape {(batch_size, seq_len, seq_len)}, got {weights.shape}"
    print("  [PASS] Attention weights shape correct")

    # Check that attention weights sum to 1 along the last axis
    assert np.allclose(weights.sum(axis=-1), 1.0), "Attention weights should sum to 1"
    print("  [PASS] Attention weights sum to 1")

    # Check that weights are non-negative
    assert np.all(weights >= 0), "Attention weights should be non-negative"
    print("  [PASS] Attention weights are non-negative")


def test_scaled_dot_product_attention_with_mask():
    """Test scaled dot-product attention with masking."""
    print("Testing ScaledDotProductAttention with mask...")

    batch_size = 2
    seq_len = 10
    d_k = 64

    attention = ScaledDotProductAttention(d_k=d_k)

    # Create dummy inputs
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)

    # Create a mask (mask out the last 3 positions)
    mask = np.zeros((batch_size, seq_len, seq_len))
    mask[:, :, -3:] = 1  # Mask last 3 key positions

    # Forward pass with mask
    output, weights = attention.forward(Q, K, V, mask)

    # Check that masked positions have near-zero attention
    # The last 3 columns of attention weights should be close to 0
    assert np.allclose(weights[:, :, -3:], 0, atol=1e-5), "Masked positions should have near-zero attention"
    print("  [PASS] Mask correctly applied")


def test_multi_head_attention():
    """Test multi-head attention."""
    print("Testing MultiHeadAttention...")

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # Create dummy inputs
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)

    # Forward pass
    output, weights = mha.forward(Q, K, V)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("  [PASS] Output shape correct")

    # Check attention weights shape
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len), \
        f"Expected shape {(batch_size, num_heads, seq_len, seq_len)}, got {weights.shape}"
    print("  [PASS] Attention weights shape correct")

    # Check that attention weights sum to 1 for each head
    assert np.allclose(weights.sum(axis=-1), 1.0), "Attention weights should sum to 1 for each head"
    print("  [PASS] Attention weights sum to 1")


def test_multi_head_attention_different_seq_lengths():
    """Test multi-head attention with different query and key sequence lengths."""
    print("Testing MultiHeadAttention with different sequence lengths...")

    batch_size = 2
    seq_len_q = 5
    seq_len_k = 10
    d_model = 512
    num_heads = 8

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # Create dummy inputs with different sequence lengths
    Q = np.random.randn(batch_size, seq_len_q, d_model)
    K = np.random.randn(batch_size, seq_len_k, d_model)
    V = np.random.randn(batch_size, seq_len_k, d_model)

    # Forward pass
    output, weights = mha.forward(Q, K, V)

    # Check output shape (should match query sequence length)
    assert output.shape == (batch_size, seq_len_q, d_model), \
        f"Expected shape {(batch_size, seq_len_q, d_model)}, got {output.shape}"
    print("  [PASS] Output shape matches query sequence length")

    # Check attention weights shape
    assert weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k), \
        f"Expected shape {(batch_size, num_heads, seq_len_q, seq_len_k)}, got {weights.shape}"
    print("  [PASS] Attention weights shape correct")


def test_multi_head_attention_with_mask():
    """Test multi-head attention with masking."""
    print("Testing MultiHeadAttention with mask...")

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # Create dummy inputs
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)

    # Create a mask
    mask = np.zeros((batch_size, seq_len, seq_len))
    mask[:, :, -3:] = 1

    # Forward pass with mask
    output, weights = mha.forward(Q, K, V, mask)

    # Check that masked positions have near-zero attention for all heads
    assert np.allclose(weights[:, :, :, -3:], 0, atol=1e-5), "Masked positions should have near-zero attention"
    print("  [PASS] Mask correctly applied to all heads")


if __name__ == "__main__":
    print("\n=== Running Attention Tests ===\n")

    test_scaled_dot_product_attention()
    test_scaled_dot_product_attention_with_mask()
    test_multi_head_attention()
    test_multi_head_attention_different_seq_lengths()
    test_multi_head_attention_with_mask()

    print("\n=== All Attention Tests Passed! ===\n")
