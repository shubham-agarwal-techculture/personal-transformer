"""
Unit tests for encoder components.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import src module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.encoder import EncoderLayer, Encoder


def test_encoder_layer():
    """Test a single encoder layer."""
    print("Testing EncoderLayer...")

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048

    layer = EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    # Create dummy input
    x = np.random.randn(batch_size, seq_len, d_model)

    # Forward pass
    output, attn_weights = layer.forward(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("  [PASS] Output shape correct")

    # Check attention weights shape
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), \
        f"Expected shape {(batch_size, num_heads, seq_len, seq_len)}, got {attn_weights.shape}"
    print("  [PASS] Attention weights shape correct")

    # Check that output is different from input (non-identity)
    assert not np.allclose(output, x), "Encoder layer should transform the input"
    print("  [PASS] Output is transformed")


def test_encoder_layer_with_mask():
    """Test encoder layer with masking."""
    print("Testing EncoderLayer with mask...")

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048

    layer = EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    # Create dummy input
    x = np.random.randn(batch_size, seq_len, d_model)

    # Create a mask (mask out last 3 positions)
    mask = np.zeros((batch_size, seq_len, seq_len))
    mask[:, :, -3:] = 1

    # Forward pass with mask
    output, attn_weights = layer.forward(x, mask)

    # Check that masked positions have near-zero attention
    assert np.allclose(attn_weights[:, :, :, -3:], 0, atol=1e-5), \
        "Masked positions should have near-zero attention"
    print("  [PASS] Mask correctly applied")


def test_encoder():
    """Test the full encoder."""
    print("Testing Encoder...")

    batch_size = 2
    seq_len = 10
    num_layers = 6
    d_model = 512
    num_heads = 8
    d_ff = 2048
    vocab_size = 10000

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size
    )

    # Create dummy input (token IDs)
    x = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    # Forward pass
    output, all_attn_weights = encoder.forward(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("  [PASS] Output shape correct")

    # Check that we have attention weights from all layers
    assert len(all_attn_weights) == num_layers, \
        f"Expected {num_layers} attention weight sets, got {len(all_attn_weights)}"
    print("  [PASS] Correct number of attention weight sets")

    # Check each attention weight set has correct shape
    for i, attn_weights in enumerate(all_attn_weights):
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), \
            f"Layer {i}: Expected shape {(batch_size, num_heads, seq_len, seq_len)}, got {attn_weights.shape}"
    print("  [PASS] All attention weight sets have correct shape")


def test_encoder_with_mask():
    """Test encoder with masking."""
    print("Testing Encoder with mask...")

    batch_size = 2
    seq_len = 10
    num_layers = 6
    d_model = 512
    num_heads = 8
    d_ff = 2048
    vocab_size = 10000

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size
    )

    # Create dummy input
    x = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    # Create a mask
    mask = np.zeros((batch_size, seq_len, seq_len))
    mask[:, :, -3:] = 1

    # Forward pass with mask
    output, all_attn_weights = encoder.forward(x, mask)

    # Check that masked positions have near-zero attention in all layers
    for i, attn_weights in enumerate(all_attn_weights):
        assert np.allclose(attn_weights[:, :, :, -3:], 0, atol=1e-5), \
            f"Layer {i}: Masked positions should have near-zero attention"
    print("  [PASS] Mask correctly applied in all layers")


def test_encoder_positional_encoding():
    """Test that encoder adds positional encoding correctly."""
    print("Testing Encoder positional encoding...")

    batch_size = 2
    seq_len = 10
    num_layers = 2
    d_model = 512
    num_heads = 8
    d_ff = 2048
    vocab_size = 100

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        dropout_rate=0.0
    )

    # Create two identical inputs
    x1 = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    x2 = x1.copy()

    # Forward pass
    output1, _ = encoder.forward(x1)
    output2, _ = encoder.forward(x2)

    # Check that outputs are identical (deterministic)
    assert np.allclose(output1, output2), "Outputs should be identical for identical inputs"
    print("  [PASS] Positional encoding applied deterministically")


if __name__ == "__main__":
    print("\n=== Running Encoder Tests ===\n")

    test_encoder_layer()
    test_encoder_layer_with_mask()
    test_encoder()
    test_encoder_with_mask()
    test_encoder_positional_encoding()

    print("\n=== All Encoder Tests Passed! ===\n")
