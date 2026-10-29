# Transformer Architecture Documentation

This document provides a detailed explanation of the Transformer architecture implemented in this educational project.

## Overview

The Transformer is a sequence-to-sequence model that relies entirely on attention mechanisms, dispensing with recurrence and convolutions. It was introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017).

## High-Level Architecture

```
Input Tokens → Encoder → Contextualized Representations
                                    ↓
Output Tokens ← Decoder ← Cross-Attention
```

### Encoder

Processes the input sequence and produces contextualized representations where each position's representation incorporates information from all other positions.

### Decoder

Generates the output sequence autoregressively, using:
- Masked self-attention to prevent attending to future positions
- Cross-attention to incorporate information from the encoder

## Component Breakdown

### 1. Input/Output Embeddings

**Purpose**: Convert discrete token IDs to continuous vector representations.

**Operation**: Token IDs are used to index into an embedding matrix.

**Shape Transformation**:
- Input: `(batch_size, seq_len)` with integer values
- Output: `(batch_size, seq_len, d_model)`

**Formula**: `embedded = embedding[token_ids] * sqrt(d_model)`

The scaling by `sqrt(d_model)` ensures embeddings and positional encodings have similar magnitudes.

### 2. Positional Encoding

**Purpose**: Inject information about the position of tokens in the sequence, since the model has no inherent notion of order.

**Implementation**: Sinusoidal functions of different frequencies.

**Formula**:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Properties**:
- Each dimension corresponds to a sinusoid with different frequency
- Allows the model to learn relative positions easily
- Can extrapolate to sequence lengths longer than those seen during training

**Shape**: `(seq_len, d_model)`

### 3. Scaled Dot-Product Attention

**Purpose**: Compute attention weights that determine how much each query position should attend to each key position.

**Formula**:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

**Components**:
- **Q (Query)**: What we're looking for
- **K (Key)**: What the input contains
- **V (Value)**: The actual content we want to retrieve
- **d_k**: Dimension of keys (used for scaling)

**Scaling**: The division by `sqrt(d_k)` prevents dot products from becoming too large, which would push softmax into regions with tiny gradients.

**Shape Transformations**:
- Q, K, V: `(batch_size, seq_len, d_k)`
- QK^T: `(batch_size, seq_len, seq_len)`
- Output: `(batch_size, seq_len, d_v)`

### 4. Multi-Head Attention

**Purpose**: Allow the model to attend to different representation subspaces at different positions simultaneously.

**Formula**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)
```

**Benefits**:
- Each head can learn different aspects of the relationships
- Provides multiple "representation subspaces"
- Increases the model's capacity to capture complex patterns

**Parameters**:
- `h`: Number of heads (typically 8)
- `d_k = d_v = d_model / h`: Dimension per head

**Shape Transformations**:
- Input: `(batch_size, seq_len, d_model)`
- After projection: `(batch_size, seq_len, d_k * h)`
- After splitting heads: `(batch_size, h, seq_len, d_k)`
- After attention: `(batch_size, h, seq_len, d_v)`
- After combining: `(batch_size, seq_len, d_model)`

### 5. Position-Wise Feed-Forward Network

**Purpose**: Apply non-linear transformations to each position independently.

**Formula**:
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

Or with GELU activation:
```
FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
```

**Architecture**:
- Two linear transformations with a non-linearity in between
- Expands dimension from `d_model` to `d_ff` (typically 4x)
- Projects back to `d_model`

**Shape Transformations**:
- Input: `(batch_size, seq_len, d_model)`
- After first layer: `(batch_size, seq_len, d_ff)`
- After activation: `(batch_size, seq_len, d_ff)`
- Output: `(batch_size, seq_len, d_model)`

### 6. Layer Normalization

**Purpose**: Stabilize hidden state dynamics and improve training speed.

**Formula**:
```
LayerNorm(x) = gamma * (x - mu) / sqrt(sigma^2 + epsilon) + beta
```

**Properties**:
- Normalizes across features (last dimension) for each sample independently
- Unlike batch normalization, doesn't depend on batch statistics
- Learnable scale (`gamma`) and shift (`beta`) parameters

**Shape**: Preserves input shape

### 7. Residual Connections

**Purpose**: Enable gradient flow through deep networks and ease training.

**Formula**:
```
output = x + SubLayer(x)
```

**Benefits**:
- Allows gradients to flow directly through the network
- Helps with training very deep networks
- Enables the model to learn residual functions

### 8. Dropout

**Purpose**: Regularization to prevent overfitting.

**Operation**: Randomly sets a fraction of units to zero during training.

**Formula** (during training):
```
output = mask * x / (1 - p)
```

Where `p` is the dropout rate and the scaling maintains expected value.

**During inference**: No dropout is applied.

## Encoder Layer Structure

Each encoder layer contains:

1. **Multi-Head Self-Attention**
   - Queries, keys, and values all come from the same input
   - Each position can attend to all positions in the input

2. **Residual Connection + Layer Normalization**

3. **Position-Wise Feed-Forward Network**

4. **Residual Connection + Layer Normalization**

**Data Flow**:
```
x → LayerNorm(x) → Attention → Dropout → x + → LayerNorm(x) → FFN → Dropout → x +
```

## Decoder Layer Structure

Each decoder layer contains three sub-layers:

1. **Masked Multi-Head Self-Attention**
   - Queries, keys, and values from decoder input
   - Masked to prevent attending to future positions (autoregressive)

2. **Residual Connection + Layer Normalization**

3. **Multi-Head Cross-Attention**
   - Queries from decoder
   - Keys and values from encoder output
   - Allows decoder to attend to encoder representations

4. **Residual Connection + Layer Normalization**

5. **Position-Wise Feed-Forward Network**

6. **Residual Connection + Layer Normalization**

**Data Flow**:
```
x → LayerNorm(x) → MaskedAttention → Dropout → x + →
LayerNorm(x) → CrossAttention → Dropout → x + →
LayerNorm(x) → FFN → Dropout → x +
```

## Complete Model

### Encoder Stack

- N encoder layers (typically N=6)
- Input embedding + positional encoding
- Output: Contextualized representations of input sequence

### Decoder Stack

- N decoder layers (typically N=6)
- Output embedding + positional encoding
- Cross-attention to encoder output
- Output projection to vocabulary

### Full Forward Pass

1. **Encoder Path**:
   - Input tokens → Embedding → Positional Encoding
   - Pass through N encoder layers
   - Output: Encoder representations

2. **Decoder Path**:
   - Output tokens → Embedding → Positional Encoding
   - Pass through N decoder layers (with cross-attention to encoder)
   - Output projection to vocabulary
   - Output: Logits over vocabulary

## Matrix Dimensions Throughout

For a batch of size B, source sequence length S, target sequence length T, model dimension D, H heads:

| Component | Shape |
|-----------|-------|
| Input tokens | (B, S) |
| Embedding | (vocab_size, D) |
| Embedded input | (B, S, D) |
| Positional encoding | (S, D) |
| Encoder layer input | (B, S, D) |
| Self-attention Q, K, V | (B, S, D) |
| Attention weights | (B, H, S, S) |
| Attention output | (B, S, D) |
| FFN hidden | (B, S, 4*D) |
| Encoder output | (B, S, D) |
| Decoder self-attention | (B, H, T, T) |
| Decoder cross-attention | (B, H, T, S) |
| Decoder output | (B, T, D) |
| Output logits | (B, T, vocab_size) |

## Key Design Choices

### Why Self-Attention?

- Allows each position to directly attend to all other positions
- Path length between any two positions is O(1)
- More efficient than RNNs for long sequences
- Captures long-range dependencies effectively

### Why Multi-Head?

- Different heads can learn different types of relationships
- Increases model capacity without increasing dimension
- Enables parallel computation of different attention patterns

### Why Positional Encoding?

- Attention is permutation-invariant by default
- Injects position information without using recurrence
- Sinusoidal encoding allows extrapolation to longer sequences

### Why Layer Norm?

- More stable than batch norm for sequence models
- Independent of batch size
- Works well with residual connections

### Why Residual Connections?

- Enables training of very deep networks
- Provides gradient highways
- Allows learning of residual functions

## Educational Notes

This implementation prioritizes:
- **Clarity over performance**: Heavy comments and explanations
- **Understanding over optimization**: No framework magic
- **Forward pass focus**: Automatic differentiation not implemented
- **Educational value**: Dimension tracking and mathematical explanations

For production use, consider frameworks like PyTorch or TensorFlow which provide:
- Automatic differentiation
- Optimized implementations
- GPU acceleration
- Distributed training support
