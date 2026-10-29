"""
Attention visualization example.

This script demonstrates how to extract and visualize attention weights
from the Transformer model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src import Transformer, create_look_ahead_mask


def print_attention_matrix(attention, title="Attention Weights"):
    """
    Print a simplified view of an attention matrix.

    Args:
        attention: Attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        title: Title for the visualization
    """
    print(f"\n{title}")
    print("=" * 60)

    # Average over batch and heads for simplicity
    avg_attention = attention.mean(axis=(0, 1))

    print(f"Attention matrix shape (averaged): {avg_attention.shape}")
    print()

    # Print the matrix with simple formatting
    seq_len_q, seq_len_k = avg_attention.shape

    print("Attention weights (rows=query positions, cols=key positions):")
    print()

    # Print header
    print("    ", end="")
    for j in range(min(seq_len_k, 10)):
        print(f"K{j:2d}  ", end="")
    if seq_len_k > 10:
        print("...")
    print()

    # Print rows
    for i in range(min(seq_len_q, 10)):
        print(f"Q{i:2d} ", end="")
        for j in range(min(seq_len_k, 10)):
            weight = avg_attention[i, j]
            # Use simple text-based visualization with ASCII
            if weight > 0.5:
                marker = "###"
            elif weight > 0.3:
                marker = "+++"
            elif weight > 0.1:
                marker = "..."
            else:
                marker = "   "
            print(f"{marker} ", end="")
        if seq_len_k > 10:
            print("...")
        else:
            print()

    if seq_len_q > 10:
        print("...")

    print()

    # Print some statistics
    print(f"Statistics:")
    print(f"  - Max attention: {avg_attention.max():.4f}")
    print(f"  - Min attention: {avg_attention.min():.4f}")
    print(f"  - Mean attention: {avg_attention.mean():.4f}")
    print()


def main():
    print("=" * 60)
    print("Attention Visualization Example")
    print("=" * 60)
    print()

    # Model hyperparameters (smaller for faster execution)
    num_layers = 2
    d_model = 128
    num_heads = 4
    d_ff = 512
    vocab_size = 1000
    max_seq_len = 32

    print(f"Creating Transformer with smaller model for visualization:")
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
        dropout_rate=0.0
    )

    print("Model created successfully!")
    print()

    # Create dummy input data
    batch_size = 1  # Single example for clearer visualization
    src_seq_len = 8
    tgt_seq_len = 6

    print(f"Creating dummy input data:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Source sequence length: {src_seq_len}")
    print(f"  - Target sequence length: {tgt_seq_len}")
    print()

    # Random token IDs
    src = np.random.randint(0, vocab_size, size=(batch_size, src_seq_len))
    tgt = np.random.randint(0, vocab_size, size=(batch_size, tgt_seq_len))

    print(f"Source tokens: {src[0]}")
    print(f"Target tokens: {tgt[0]}")
    print()

    # Forward pass
    print("Running forward pass...")
    output, enc_attn, dec_attn = transformer.forward(src, tgt)

    print("Forward pass completed!")
    print()

    # Visualize encoder self-attention
    print("=" * 60)
    print("ENCODER SELF-ATTENTION")
    print("=" * 60)

    for layer_idx, attn in enumerate(enc_attn):
        print(f"\nEncoder Layer {layer_idx}:")
        print_attention_matrix(attn, f"Encoder Layer {layer_idx} Self-Attention")

    # Visualize decoder self-attention
    print("=" * 60)
    print("DECODER SELF-ATTENTION (Masked)")
    print("=" * 60)

    dec_self_attn = dec_attn[0]
    for layer_idx, attn in enumerate(dec_self_attn):
        print(f"\nDecoder Layer {layer_idx}:")
        print_attention_matrix(attn, f"Decoder Layer {layer_idx} Self-Attention (Masked)")

    # Visualize decoder cross-attention
    print("=" * 60)
    print("DECODER CROSS-ATTENTION (Encoder-Decoder)")
    print("=" * 60)

    dec_cross_attn = dec_attn[1]
    for layer_idx, attn in enumerate(dec_cross_attn):
        print(f"\nDecoder Layer {layer_idx}:")
        print_attention_matrix(attn, f"Decoder Layer {layer_idx} Cross-Attention")

    print("=" * 60)
    print("Visualization completed!")
    print("=" * 60)
    print()
    print("Notes:")
    print("- Encoder self-attention: Each position can attend to all positions")
    print("- Decoder self-attention: Masked, so positions can only attend to previous positions")
    print("- Decoder cross-attention: Query positions attend to encoder key/value positions")
    print()


if __name__ == "__main__":
    main()
