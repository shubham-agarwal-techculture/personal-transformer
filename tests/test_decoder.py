"""
Unit tests for decoder components.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import src module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.decoder import DecoderLayer, Decoder, create_look_ahead_mask


def test_look_ahead_mask():
    """Test look-ahead mask creation."""
    print("Testing create_look_ahead_mask...")

    seq_len = 5
    mask = create_look_ahead_mask(seq_len)

    # Check shape
    assert mask.shape == (seq_len, seq_len), f"Expected shape ({seq_len}, {seq_len}), got {mask.shape}"
    print("  [PASS] Mask shape correct")

    # Check that diagonal and below are 0 (can attend)
    assert np.allclose(np.tril(mask), 0), "Lower triangle should be 0"
    print("  [PASS] Lower triangle is 0")

    # Check that above diagonal is 1 (cannot attend)
    # Check specific positions: position i should not be able to attend to position j where j > i
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert mask[i, j] == 1, f"Position {i} should not attend to {j}"
    print("  [PASS] Upper triangle is 1")


def test_decoder_layer():
    """Test a single decoder layer."""
    print("Testing DecoderLayer...")

    batch_size = 2
    seq_len = 10
    enc_seq_len = 15
    d_model = 512
    num_heads = 8
    d_ff = 2048

    layer = DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    # Create dummy inputs
    x = np.random.randn(batch_size, seq_len, d_model)
    encoder_output = np.random.randn(batch_size, enc_seq_len, d_model)

    # Forward pass
    output, (self_attn, cross_attn) = layer.forward(x, encoder_output)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("  [PASS] Output shape correct")

    # Check self-attention weights shape
    assert self_attn.shape == (batch_size, num_heads, seq_len, seq_len), \
        f"Expected shape {(batch_size, num_heads, seq_len, seq_len)}, got {self_attn.shape}"
    print("  [PASS] Self-attention weights shape correct")

    # Check cross-attention weights shape
    assert cross_attn.shape == (batch_size, num_heads, seq_len, enc_seq_len), \
        f"Expected shape {(batch_size, num_heads, seq_len, enc_seq_len)}, got {cross_attn.shape}"
    print("  [PASS] Cross-attention weights shape correct")


def test_decoder_layer_with_masks():
    """Test decoder layer with masks."""
    print("Testing DecoderLayer with masks...")

    batch_size = 2
    seq_len = 10
    enc_seq_len = 15
    d_model = 512
    num_heads = 8
    d_ff = 2048

    layer = DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    # Create dummy inputs
    x = np.random.randn(batch_size, seq_len, d_model)
    encoder_output = np.random.randn(batch_size, enc_seq_len, d_model)

    # Create look-ahead mask
    look_ahead_mask = create_look_ahead_mask(seq_len)

    # Create padding mask for cross-attention
    padding_mask = np.zeros((batch_size, seq_len, enc_seq_len))
    padding_mask[:, :, -3:] = 1

    # Forward pass with masks
    # Note: padding_mask is only used for cross-attention, not self-attention
    output, (self_attn, cross_attn) = layer.forward(
        x, encoder_output, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask
    )

    # Check that look-ahead mask is applied to self-attention
    # Upper triangle should be near-zero
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert np.allclose(self_attn[:, :, i, j], 0, atol=1e-5), \
                f"Self-attention should be masked for position {i} attending to {j}"
    print("  [PASS] Look-ahead mask applied to self-attention")

    # Check that padding mask is applied to cross-attention
    assert np.allclose(cross_attn[:, :, :, -3:], 0, atol=1e-5), \
        "Padding mask should be applied to cross-attention"
    print("  [PASS] Padding mask applied to cross-attention")


def test_decoder():
    """Test the full decoder."""
    print("Testing Decoder...")

    batch_size = 2
    seq_len = 10
    enc_seq_len = 15
    num_layers = 6
    d_model = 512
    num_heads = 8
    d_ff = 2048
    vocab_size = 10000

    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size
    )

    # Create dummy inputs
    x = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    encoder_output = np.random.randn(batch_size, enc_seq_len, d_model)

    # Forward pass
    output, (all_self_attn, all_cross_attn) = decoder.forward(x, encoder_output)

    # Check output shape
    assert output.shape == (batch_size, seq_len, vocab_size), \
        f"Expected shape {(batch_size, seq_len, vocab_size)}, got {output.shape}"
    print("  [PASS] Output shape correct")

    # Check that we have attention weights from all layers
    assert len(all_self_attn) == num_layers, \
        f"Expected {num_layers} self-attention sets, got {len(all_self_attn)}"
    assert len(all_cross_attn) == num_layers, \
        f"Expected {num_layers} cross-attention sets, got {len(all_cross_attn)}"
    print("  [PASS] Correct number of attention weight sets")


def test_decoder_with_masks():
    """Test decoder with masks."""
    print("Testing Decoder with masks...")

    batch_size = 2
    seq_len = 10
    enc_seq_len = 15
    num_layers = 2
    d_model = 512
    num_heads = 8
    d_ff = 2048
    vocab_size = 10000

    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        dropout_rate=0.0
    )

    # Create dummy inputs
    x = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    encoder_output = np.random.randn(batch_size, enc_seq_len, d_model)

    # Create masks
    look_ahead_mask = create_look_ahead_mask(seq_len)
    padding_mask = np.zeros((batch_size, seq_len, enc_seq_len))
    padding_mask[:, :, -3:] = 1

    # Forward pass with masks
    output, (all_self_attn, all_cross_attn) = decoder.forward(
        x, encoder_output, look_ahead_mask, padding_mask
    )

    # Check that look-ahead mask is applied in all layers
    for i, self_attn in enumerate(all_self_attn):
        for row in range(seq_len):
            for col in range(row + 1, seq_len):
                assert np.allclose(self_attn[:, :, row, col], 0, atol=1e-5), \
                    f"Layer {i}: Look-ahead mask not applied"
    print("  [PASS] Look-ahead mask applied in all layers")

    # Check that padding mask is applied in all layers
    for i, cross_attn in enumerate(all_cross_attn):
        assert np.allclose(cross_attn[:, :, :, -3:], 0, atol=1e-5), \
            f"Layer {i}: Padding mask not applied"
    print("  [PASS] Padding mask applied in all layers")


if __name__ == "__main__":
    print("\n=== Running Decoder Tests ===\n")

    test_look_ahead_mask()
    test_decoder_layer()
    test_decoder_layer_with_masks()
    test_decoder()
    test_decoder_with_masks()

    print("\n=== All Decoder Tests Passed! ===\n")
