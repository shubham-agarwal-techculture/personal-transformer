"""
Unit tests for the complete Transformer model.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import src module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.transformer import Transformer
from src.decoder import create_look_ahead_mask


def test_transformer_forward():
    """Test the complete Transformer forward pass."""
    print("Testing Transformer forward pass...")

    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    num_layers = 2
    d_model = 512
    num_heads = 8
    d_ff = 2048
    vocab_size = 10000

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        dropout_rate=0.0
    )

    # Create dummy inputs
    src = np.random.randint(0, vocab_size, size=(batch_size, src_seq_len))
    tgt = np.random.randint(0, vocab_size, size=(batch_size, tgt_seq_len))

    # Forward pass
    output, enc_attn, dec_attn = transformer.forward(src, tgt)

    # Check output shape
    assert output.shape == (batch_size, tgt_seq_len, vocab_size), \
        f"Expected shape {(batch_size, tgt_seq_len, vocab_size)}, got {output.shape}"
    print("  [PASS] Output shape correct")

    # Check encoder attention weights
    assert len(enc_attn) == num_layers, \
        f"Expected {num_layers} encoder attention sets, got {len(enc_attn)}"
    print("  [PASS] Encoder attention weights collected")

    # Check decoder attention weights
    assert len(dec_attn) == 2, "Decoder should return (self_attn, cross_attn)"
    assert len(dec_attn[0]) == num_layers, \
        f"Expected {num_layers} decoder self-attention sets, got {len(dec_attn[0])}"
    assert len(dec_attn[1]) == num_layers, \
        f"Expected {num_layers} decoder cross-attention sets, got {len(dec_attn[1])}"
    print("  [PASS] Decoder attention weights collected")


def test_transformer_with_masks():
    """Test Transformer with masks."""
    print("Testing Transformer with masks...")

    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    num_layers = 2
    d_model = 512
    num_heads = 8
    d_ff = 2048
    vocab_size = 10000

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        dropout_rate=0.0
    )

    # Create dummy inputs
    src = np.random.randint(0, vocab_size, size=(batch_size, src_seq_len))
    tgt = np.random.randint(0, vocab_size, size=(batch_size, tgt_seq_len))

    # Create masks
    src_mask = np.zeros((batch_size, src_seq_len, src_seq_len))
    src_mask[:, :, -2:] = 1

    tgt_mask = create_look_ahead_mask(tgt_seq_len)

    # Forward pass with masks
    output, enc_attn, dec_attn = transformer.forward(src, tgt, src_mask, tgt_mask)

    # Check output shape
    assert output.shape == (batch_size, tgt_seq_len, vocab_size), \
        f"Expected shape {(batch_size, tgt_seq_len, vocab_size)}, got {output.shape}"
    print("  [PASS] Output shape correct with masks")


def test_transformer_encode_decode():
    """Test separate encode and decode methods."""
    print("Testing Transformer encode/decode...")

    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    num_layers = 2
    d_model = 512
    num_heads = 8
    d_ff = 2048
    vocab_size = 10000

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        dropout_rate=0.0
    )

    # Create dummy inputs
    src = np.random.randint(0, vocab_size, size=(batch_size, src_seq_len))
    tgt = np.random.randint(0, vocab_size, size=(batch_size, tgt_seq_len))

    # Encode
    encoder_output, enc_attn = transformer.encode(src)

    # Check encoder output shape
    assert encoder_output.shape == (batch_size, src_seq_len, d_model), \
        f"Expected shape {(batch_size, src_seq_len, d_model)}, got {encoder_output.shape}"
    print("  [PASS] Encoder output shape correct")

    # Decode
    tgt_mask = create_look_ahead_mask(tgt_seq_len)
    output, dec_attn = transformer.decode(tgt, encoder_output, tgt_mask)

    # Check decoder output shape
    assert output.shape == (batch_size, tgt_seq_len, vocab_size), \
        f"Expected shape {(batch_size, tgt_seq_len, vocab_size)}, got {output.shape}"
    print("  [PASS] Decoder output shape correct")


if __name__ == "__main__":
    print("\n=== Running Transformer Tests ===\n")

    test_transformer_forward()
    test_transformer_with_masks()
    test_transformer_encode_decode()

    print("\n=== All Transformer Tests Passed! ===\n")
