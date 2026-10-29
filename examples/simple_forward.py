"""
Simple forward pass example.

This script demonstrates how to use the Transformer model for a basic
forward pass with dummy data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src import Transformer


def main():
    print("=" * 60)
    print("Simple Transformer Forward Pass Example")
    print("=" * 60)
    print()

    # Model hyperparameters
    num_layers = 2
    d_model = 512
    num_heads = 8
    d_ff = 2048
    vocab_size = 10000
    max_seq_len = 512

    print(f"Creating Transformer with:")
    print(f"  - Layers: {num_layers}")
    print(f"  - Model dimension: {d_model}")
    print(f"  - Attention heads: {num_heads}")
    print(f"  - FFN dimension: {d_ff}")
    print(f"  - Vocabulary size: {vocab_size}")
    print()

    # Initialize the model
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dropout_rate=0.0  # Disable dropout for deterministic output
    )

    print("Model created successfully!")
    print()

    # Create dummy input data
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8

    print(f"Creating dummy input data:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Source sequence length: {src_seq_len}")
    print(f"  - Target sequence length: {tgt_seq_len}")
    print()

    # Random token IDs
    src = np.random.randint(0, vocab_size, size=(batch_size, src_seq_len))
    tgt = np.random.randint(0, vocab_size, size=(batch_size, tgt_seq_len))

    print(f"Source tokens shape: {src.shape}")
    print(f"Target tokens shape: {tgt.shape}")
    print()

    # Forward pass
    print("Running forward pass...")
    output, enc_attn, dec_attn = transformer.forward(src, tgt)

    print()
    print("Forward pass completed!")
    print()
    print(f"Output logits shape: {output.shape}")
    print(f"  - Batch size: {output.shape[0]}")
    print(f"  - Sequence length: {output.shape[1]}")
    print(f"  - Vocabulary size: {output.shape[2]}")
    print()

    print(f"Encoder attention weights: {len(enc_attn)} layers")
    for i, attn in enumerate(enc_attn):
        print(f"  Layer {i}: shape {attn.shape}")
    print()

    print(f"Decoder attention weights:")
    print(f"  - Self-attention: {len(dec_attn[0])} layers")
    for i, attn in enumerate(dec_attn[0]):
        print(f"    Layer {i}: shape {attn.shape}")
    print(f"  - Cross-attention: {len(dec_attn[1])} layers")
    for i, attn in enumerate(dec_attn[1]):
        print(f"    Layer {i}: shape {attn.shape}")
    print()

    # Get predicted tokens
    predictions = np.argmax(output, axis=-1)
    print(f"Predicted tokens shape: {predictions.shape}")
    print(f"Sample predictions (first sequence):")
    print(f"  {predictions[0]}")
    print()

    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
