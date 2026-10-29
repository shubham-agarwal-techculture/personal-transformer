"""
Decoder components for the Transformer implementation.

This module implements:
- DecoderLayer: A single decoder block with three sub-layers
- Decoder: A stack of decoder layers with embedding and positional encoding

The decoder generates the output sequence autoregressively, using:
1. Masked self-attention to prevent attending to future positions
2. Cross-attention to incorporate information from the encoder
3. Feed-forward network for non-linear transformations
"""

import numpy as np
from .attention import MultiHeadAttention
from .ffn import PositionWiseFFN
from .normalization import LayerNormalization, Dropout
from .utils import get_positional_encoding, initialize_weights, initialize_bias


def create_look_ahead_mask(seq_len):
    """
    Create a look-ahead mask for masked self-attention.

    This mask prevents positions from attending to future positions,
    ensuring the autoregressive property (each position can only see
    previous positions).

    The mask is a lower triangular matrix where:
    - Mask[i, j] = 0 if j <= i (can attend)
    - Mask[i, j] = 1 if j > i (cannot attend)

    Args:
        seq_len: Length of the sequence

    Returns:
        Mask of shape (seq_len, seq_len)
    """
    # Create a lower triangular matrix
    # np.tri returns a matrix with 1s on and below the diagonal
    mask = 1 - np.tri(seq_len, seq_len)
    return mask.astype(float)


class DecoderLayer:
    """
    A single decoder layer.

    Each decoder layer consists of three sub-layers:
    1. Masked multi-head self-attention mechanism
    2. Multi-head cross-attention (encoder-decoder attention)
    3. Position-wise feed-forward network

    Each sub-layer is followed by:
    - Residual connection (add the input to the output)
    - Layer normalization

    The formula is:
        output = LayerNorm(x + SubLayer(x))

    Where SubLayer is one of the three attention mechanisms or FFN.

    The key difference from the encoder is:
    - Self-attention is masked to prevent attending to future positions
    - Cross-attention uses encoder outputs as keys and values

    Attributes:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of the feed-forward network
        dropout_rate: Dropout probability
        self_attention: Masked multi-head self-attention
        cross_attention: Multi-head cross-attention
        feed_forward: Position-wise feed-forward network
        norm1, norm2, norm3: Layer normalization for each sub-layer
        dropout1, dropout2, dropout3: Dropout for each sub-layer
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        """
        Initialize a decoder layer.

        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network (typically 4x d_model)
            dropout_rate: Dropout probability (default: 0.1)
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # Masked multi-head self-attention
        # Queries, keys, and values all come from the decoder input
        # Masked to prevent attending to future positions
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # Multi-head cross-attention (encoder-decoder attention)
        # Queries come from decoder, keys and values from encoder
        self.cross_attention = MultiHeadAttention(d_model, num_heads)

        # Feed-forward network
        self.feed_forward = PositionWiseFFN(d_model, d_ff)

        # Layer normalization for each sub-layer
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)

        # Dropout for each sub-layer
        if dropout_rate > 0:
            self.dropout1 = Dropout(dropout_rate)
            self.dropout2 = Dropout(dropout_rate)
            self.dropout3 = Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None
            self.dropout3 = None

    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None):
        """
        Apply the decoder layer to the input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            encoder_output: Output from encoder of shape (batch_size, enc_seq_len, d_model)
            look_ahead_mask: Mask for self-attention of shape (seq_len, seq_len)
                            Prevents attending to future positions
            padding_mask: Mask for cross-attention of shape (batch_size, seq_len, enc_seq_len)
                         Prevents attending to padding tokens in encoder output
                         (Only used for cross-attention, not self-attention)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
            Tuple of (self_attention_weights, cross_attention_weights)

        Example:
            >>> layer = DecoderLayer(d_model=512, num_heads=8, d_ff=2048)
            >>> x = np.random.randn(2, 10, 512)
            >>> enc_output = np.random.randn(2, 15, 512)
            >>> output, (self_attn, cross_attn) = layer.forward(x, enc_output)
            >>> output.shape
            (2, 10, 512)
        """
        # Sub-layer 1: Masked self-attention with residual connection and normalization
        # Architecture: x -> LayerNorm(x + Dropout(MaskedAttention(x)))

        # Apply layer normalization before attention (pre-norm)
        normed_x = self.norm1.forward(x)

        # Apply masked self-attention
        # Look-ahead mask prevents attending to future positions
        # Padding mask (if provided for self-attention) prevents attending to padding
        if look_ahead_mask is not None:
            mask = look_ahead_mask[np.newaxis, :, :]
        else:
            mask = None

        self_attn_output, self_attn_weights = self.self_attention.forward(
            normed_x, normed_x, normed_x, mask
        )

        # Apply dropout
        if self.dropout1 is not None:
            self_attn_output = self.dropout1.forward(self_attn_output)

        # Residual connection
        x = x + self_attn_output

        # Sub-layer 2: Cross-attention with residual connection and normalization
        # Architecture: x -> LayerNorm(x + Dropout(CrossAttention(x, encoder_output)))

        # Apply layer normalization before cross-attention (pre-norm)
        normed_x = self.norm2.forward(x)

        # Apply cross-attention
        # Queries from decoder, keys and values from encoder
        cross_attn_output, cross_attn_weights = self.cross_attention.forward(
            normed_x, encoder_output, encoder_output, padding_mask
        )

        # Apply dropout
        if self.dropout2 is not None:
            cross_attn_output = self.dropout2.forward(cross_attn_output)

        # Residual connection
        x = x + cross_attn_output

        # Sub-layer 3: Feed-forward network with residual connection and normalization
        # Architecture: x -> LayerNorm(x + Dropout(FFN(x)))

        # Apply layer normalization before FFN (pre-norm)
        normed_x = self.norm3.forward(x)

        # Apply feed-forward network
        ffn_output = self.feed_forward.forward(normed_x)

        # Apply dropout
        if self.dropout3 is not None:
            ffn_output = self.dropout3.forward(ffn_output)

        # Residual connection
        x = x + ffn_output

        return x, (self_attn_weights, cross_attn_weights)


class Decoder:
    """
    The complete decoder stack.

    The decoder consists of:
    1. Output embedding layer (projects token IDs to d_model)
    2. Positional encoding (adds position information)
    3. A stack of N decoder layers
    4. Output projection layer (projects to vocabulary size)

    The formula is:
        Decoder(x, encoder_output) = OutputProjection(Layer_N(...Layer_1(
            Embedding(x) + PositionalEncoding, encoder_output)))

    Where each Layer_i is a DecoderLayer.

    The decoder generates the output sequence autoregressively, using
    masked self-attention to ensure each position only attends to
    previous positions, and cross-attention to incorporate information
    from the encoder.

    Attributes:
        num_layers: Number of decoder layers
        d_model: Dimension of the model
        num_heads: Number of attention heads per layer
        d_ff: Dimension of the feed-forward network
        dropout_rate: Dropout probability
        vocab_size: Size of the vocabulary
        max_seq_len: Maximum sequence length for positional encoding
        embedding: Token embedding matrix
        positional_encoding: Pre-computed positional encodings
        layers: List of decoder layers
        output_projection: Final projection to vocabulary
        dropout: Dropout after embedding + positional encoding
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size,
                 max_seq_len=512, dropout_rate=0.1):
        """
        Initialize the decoder.

        Args:
            num_layers: Number of decoder layers (typically 6)
            d_model: Dimension of the model (typically 512)
            num_heads: Number of attention heads (typically 8)
            d_ff: Dimension of the feed-forward network (typically 2048)
            vocab_size: Size of the vocabulary
            max_seq_len: Maximum sequence length for positional encoding
            dropout_rate: Dropout probability (default: 0.1)
        """
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate

        # Token embedding layer
        # Shape: (vocab_size, d_model)
        self.embedding = initialize_weights((vocab_size, d_model), method='xavier')

        # Pre-compute positional encodings
        # Shape: (max_seq_len, d_model)
        self.positional_encoding = get_positional_encoding(max_seq_len, d_model)

        # Create stack of decoder layers
        self.layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]

        # Output projection layer
        # Projects from d_model to vocab_size
        # Shape: (d_model, vocab_size)
        self.output_projection = initialize_weights((d_model, vocab_size), method='xavier')

        # Dropout for embedding + positional encoding
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None

        # Set dropout to inference mode for consistency
        if self.dropout is not None:
            self.dropout.set_training_mode(False)
        for layer in self.layers:
            if hasattr(layer, 'dropout1') and layer.dropout1 is not None:
                layer.dropout1.set_training_mode(False)
            if hasattr(layer, 'dropout2') and layer.dropout2 is not None:
                layer.dropout2.set_training_mode(False)
            if hasattr(layer, 'dropout3') and layer.dropout3 is not None:
                layer.dropout3.set_training_mode(False)

    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None):
        """
        Apply the decoder to the input.

        Args:
            x: Input token IDs of shape (batch_size, seq_len)
            encoder_output: Output from encoder of shape (batch_size, enc_seq_len, d_model)
            look_ahead_mask: Mask for self-attention of shape (seq_len, seq_len)
            padding_mask: Mask for cross-attention of shape (batch_size, seq_len, enc_seq_len)
                         (Only used for cross-attention, not self-attention)

        Returns:
            Output logits of shape (batch_size, seq_len, vocab_size)
            Lists of attention weights from each layer

        Example:
            >>> decoder = Decoder(num_layers=6, d_model=512, num_heads=8, d_ff=2048, vocab_size=10000)
            >>> x = np.random.randint(0, 10000, size=(2, 10))
            >>> enc_output = np.random.randn(2, 15, 512)
            >>> output, attn_weights = decoder.forward(x, enc_output)
            >>> output.shape
            (2, 10, 10000)
        """
        batch_size, seq_len = x.shape

        # Convert token IDs to embeddings
        # x: (batch_size, seq_len) with integer values
        # embedding: (vocab_size, d_model)
        # embedded: (batch_size, seq_len, d_model)
        embedded = self.embedding[x]

        # Scale embeddings by sqrt(d_model)
        embedded = embedded * np.sqrt(self.d_model)

        # Add positional encoding
        pe = self.positional_encoding[:seq_len, :]
        pe = pe[np.newaxis, :, :]
        embedded = embedded + pe

        # Apply dropout
        if self.dropout is not None:
            embedded = self.dropout.forward(embedded)

        # Pass through each decoder layer
        x = embedded
        all_self_attn_weights = []
        all_cross_attn_weights = []

        for layer in self.layers:
            x, (self_attn_weights, cross_attn_weights) = layer.forward(
                x, encoder_output, look_ahead_mask, padding_mask
            )
            all_self_attn_weights.append(self_attn_weights)
            all_cross_attn_weights.append(cross_attn_weights)

        # Apply output projection
        # x: (batch_size, seq_len, d_model)
        # output_projection: (d_model, vocab_size)
        # output: (batch_size, seq_len, vocab_size)
        output = np.dot(x, self.output_projection)

        return output, (all_self_attn_weights, all_cross_attn_weights)
