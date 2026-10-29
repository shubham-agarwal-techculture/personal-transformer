---
agent: devin-local
session: placid-digestion
created: 2026-10-29T08:46:28Z
---
# Build Educational Vanilla Transformer with NumPy

Create a step-by-step implementation of a full encoder-decoder transformer using only NumPy, prioritizing educational clarity and understanding of the underlying mathematics.

## Overview
Build a complete vanilla transformer (encoder-decoder architecture) from scratch using only NumPy. This educational implementation will break down each component with clear explanations, focusing on understanding the mathematics rather than performance optimization.

## Implementation Steps

### Phase 1: Project Setup and Foundations
1. **Create project structure**
   - Set up Python virtual environment
   - Create `requirements.txt` (numpy only)
   - Create directory structure: `src/`, `tests/`, `examples/`
   - Add README with educational goals

2. **Core utility functions** (`src/utils.py`)
   - `softmax` - stable softmax implementation
   - `relu` - activation function
   - `gelu` - modern activation used in transformers
   - `get_positional_encoding` - sinusoidal position encoding
   - Helper functions for initialization

### Phase 2: Building Blocks

3. **Attention Mechanism** (`src/attention.py`)
   - `ScaledDotProductAttention` - core attention computation
   - `MultiHeadAttention` - splitting into multiple heads
   - Clear explanations of Q, K, V matrices
   - Visual comments showing matrix dimensions

4. **Feed-Forward Networks** (`src/ffn.py`)
   - `PositionWiseFFN` - two-layer feed-forward network
   - Explain expansion factor and non-linearity

5. **Normalization and Dropout** (`src/normalization.py`)
   - `LayerNormalization` - layer norm implementation
   - `Dropout` - dropout for regularization
   - Explain why layer norm is used in transformers

### Phase 3: Encoder Components

6. **Encoder Layer** (`src/encoder.py`)
   - `EncoderLayer` - single encoder block
   - Self-attention + feed-forward + residual connections
   - Clear data flow documentation

7. **Full Encoder** (`src/encoder.py`)
   - `Encoder` - stack of encoder layers
   - Embedding layer + positional encoding
   - Combine all components

### Phase 4: Decoder Components

8. **Decoder Layer** (`src/decoder.py`)
   - `DecoderLayer` - single decoder block
   - Masked self-attention
   - Cross-attention (encoder-decoder attention)
   - Feed-forward network
   - Explain the three types of attention

9. **Full Decoder** (`src/decoder.py`)
   - `Decoder` - stack of decoder layers
   - Embedding layer + positional encoding
   - Output projection

### Phase 5: Complete Transformer

10. **Transformer Model** (`src/transformer.py`)
    - `Transformer` - complete encoder-decoder model
    - Combine encoder and decoder
    - Forward pass with all components
    - Input/output processing

11. **Model Training Utilities** (`src/training.py`)
    - Simple loss function (cross-entropy)
    - Basic training loop skeleton
    - Note: Full training requires gradient computation (will implement forward pass first)

### Phase 6: Testing and Examples

12. **Unit Tests** (`tests/`)
    - Test each component independently
    - Verify shape correctness
    - Test numerical stability

13. **Educational Examples** (`examples/`)
    - Simple forward pass demo
    - Visualize attention weights
    - Example showing matrix dimensions at each step
    - Interactive exploration notebook (optional)

14. **Documentation**
    - Add detailed docstrings to each class/function
    - Create `ARCHITECTURE.md` explaining the full architecture
    - Add mathematical formulas in comments
    - Create data flow diagrams in markdown

## Files to Create

- `requirements.txt` - NumPy dependency
- `README.md` - Project overview and educational goals
- `src/utils.py` - Utility functions
- `src/attention.py` - Attention mechanisms
- `src/ffn.py` - Feed-forward networks
- `src/normalization.py` - Layer norm and dropout
- `src/encoder.py` - Encoder components
- `src/decoder.py` - Decoder components
- `src/transformer.py` - Complete model
- `src/training.py` - Training utilities
- `tests/test_*.py` - Unit tests for each module
- `examples/simple_forward.py` - Basic usage example
- `examples/visualize_attention.py` - Attention visualization
- `ARCHITECTURE.md` - Detailed architecture documentation

## Educational Approach

Each component will include:
- **Clear variable names** matching the original paper
- **Dimension comments** showing matrix shapes at each step
- **Mathematical explanations** in docstrings
- **Inline comments** explaining the "why" not just the "how"
- **Simple test cases** to verify understanding

## Verification

- [ ] Run unit tests for each component
- [ ] Verify forward pass produces correct output shapes
- [ ] Test with small dummy inputs
- [ ] Visualize attention weights to confirm they make sense
- [ ] Document matrix dimensions throughout the pipeline

## Notes

- This implementation focuses on the **forward pass** and understanding the architecture
- Full training would require automatic differentiation (not implemented in pure NumPy)
- The goal is educational clarity, not production performance
- All code will be heavily commented for learning purposes
