# Educational Vanilla Transformer

A from-scratch implementation of the Transformer architecture using only NumPy, built for educational purposes.

## Purpose

This project implements the complete encoder-decoder Transformer architecture (as described in "Attention Is All You Need") using pure NumPy. The goal is to provide a clear, well-documented implementation that helps you understand:

- How attention mechanisms work mathematically
- The role of each component in the architecture
- Matrix dimensions and data flow through the network
- Why transformers are effective for sequence modeling

## Features

- **Pure NumPy** - No deep learning framework magic, just the mathematics
- **Full Architecture** - Complete encoder-decoder with multi-head attention
- **Educational Focus** - Heavy comments, dimension tracking, and explanations
- **Step-by-Step** - Each component built incrementally for learning

## Architecture

The implementation follows the original Transformer paper:

```
Input -> Encoder Stack -> Decoder Stack -> Output
         (N layers)       (N layers)
```

Each encoder layer contains:
- Multi-head self-attention
- Position-wise feed-forward network
- Residual connections and layer normalization

Each decoder layer contains:
- Masked multi-head self-attention
- Multi-head cross-attention (encoder-decoder attention)
- Position-wise feed-forward network
- Residual connections and layer normalization

## Project Structure

```
.
├── src/
│   ├── utils.py           # Core utility functions
│   ├── attention.py       # Attention mechanisms
│   ├── ffn.py            # Feed-forward networks
│   ├── normalization.py   # Layer norm and dropout
│   ├── encoder.py        # Encoder components
│   ├── decoder.py        # Decoder components
│   ├── transformer.py    # Complete model
│   └── training.py       # Training utilities
├── tests/                # Unit tests
├── examples/             # Educational examples
└── ARCHITECTURE.md       # Detailed architecture documentation
```

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

See `examples/` directory for usage examples.

## Educational Notes

This implementation prioritizes clarity over performance:
- Forward pass only (no automatic differentiation)
- Detailed comments explaining matrix operations
- Dimension tracking at each step
- Mathematical formulas in docstrings

For production use, consider PyTorch or TensorFlow implementations.

## References

- "Attention Is All You Need" (Vaswani et al., 2017)
- The Annotated Transformer (Harvard NLP)
