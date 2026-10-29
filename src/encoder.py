"""
Encoder components for the Transformer implementation.

This module implements:
- EncoderLayer: A single encoder block with self-attention and FFN
- Encoder: A stack of encoder layers with embedding and positional encoding

The encoder processes the input sequence and produces contextualized representations
that capture relationships between all positions in the sequence.
"""

import numpy as np
from .attention import MultiHeadAttention
from .ffn import PositionWiseFFN
from .normalization import LayerNormalization, Dropout
from .utils import get_positional_encoding, initialize_weights, initialize_bias


class EncoderLayer:
    """
    A single encoder layer.

    Each encoder layer consists of two sub-layers:
    1. Multi-head self-attention mechanism
    2. Position-wise feed-forward network

    Each sub-layer is followed by:
    - Residual connection (add the input to the output)
    - Layer normalization

    The formula is:
        output = LayerNorm(x + SubLayer(x))

    Where SubLayer is either the self-attention or feed-forward network.

    This architecture allows gradients to flow directly through the network
    via the residual connections, making it easier to train very deep networks.

    Attributes:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of the feed-forward network
        dropout_rate: Dropout probability
        self_attention: Multi-head self-attention layer
        feed_forward: Position-wise feed-forward network
        norm1: Layer normalization after attention
        norm2: Layer normalization after FFN
        dropout1: Dropout after attention residual
        dropout2: Dropout after FFN residual
    """

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        """
        Initialize an encoder layer.

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

        # Multi-head self-attention
        # This allows each position to attend to all positions in the input
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # Feed-forward network
        self.feed_forward = PositionWiseFFN(d_model, d_ff)

        # Layer normalization
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

        # Dropout (only if rate > 0)
        if dropout_rate > 0:
            self.dropout1 = Dropout(dropout_rate)
            self.dropout2 = Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x, mask=None):
        """
        Apply the encoder layer to the input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask of shape (batch_size, seq_len, seq_len)
                  Used to prevent attention to padding tokens

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
            Attention weights of shape (batch_size, num_heads, seq_len, seq_len)

        Example:
            >>> layer = EncoderLayer(d_model=512, num_heads=8, d_ff=2048)
            >>> x = np.random.randn(2, 10, 512)
            >>> output, attn_weights = layer.forward(x)
            >>> output.shape
            (2, 10, 512)
        """
        # Sub-layer 1: Multi-head self-attention with residual connection and normalization
        # Architecture: x -> LayerNorm(x + Dropout(Attention(x)))
        # Note: The original paper applies normalization after the residual connection,
        # but modern implementations (like BERT) apply it before. We'll use the modern approach.

        # Apply layer normalization before attention (pre-norm)
        normed_x = self.norm1.forward(x)

        # Apply self-attention
        # In self-attention, queries, keys, and values all come from the same input
        attn_output, attn_weights = self.self_attention.forward(normed_x, normed_x, normed_x, mask)

        # Apply dropout if enabled
        if self.dropout1 is not None:
            attn_output = self.dropout1.forward(attn_output)

        # Residual connection: add the original input
        x = x + attn_output

        # Sub-layer 2: Feed-forward network with residual connection and normalization
        # Architecture: x -> LayerNorm(x + Dropout(FFN(x)))

        # Apply layer normalization before FFN (pre-norm)
        normed_x = self.norm2.forward(x)

        # Apply feed-forward network
        ffn_output = self.feed_forward.forward(normed_x)

        # Apply dropout if enabled
        if self.dropout2 is not None:
            ffn_output = self.dropout2.forward(ffn_output)

        # Residual connection: add the previous output
        x = x + ffn_output

        return x, attn_weights


class Encoder:
    """
    The complete encoder stack.

    The encoder consists of:
    1. Input embedding layer (projects token IDs to d_model)
    2. Positional encoding (adds position information)
    3. A stack of N encoder layers

    The formula is:
        Encoder(x) = Layer_N(...Layer_2(Layer_1(Embedding(x) + PositionalEncoding)))

    Where each Layer_i is an EncoderLayer.

    The encoder produces contextualized representations of the input sequence,
    where each position's representation incorporates information from all
    other positions through the self-attention mechanism.

    Attributes:
        num_layers: Number of encoder layers
        d_model: Dimension of the model
        num_heads: Number of attention heads per layer
        d_ff: Dimension of the feed-forward network
        dropout_rate: Dropout probability
        vocab_size: Size of the vocabulary
        max_seq_len: Maximum sequence length for positional encoding
        embedding: Token embedding matrix
        positional_encoding: Pre-computed positional encodings
        layers: List of encoder layers
        dropout: Dropout after embedding + positional encoding
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size,
                 max_seq_len=512, dropout_rate=0.1):
        """
        Initialize the encoder.

        Args:
            num_layers: Number of encoder layers (typically 6)
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
        # Projects token IDs (integers) to d_model-dimensional vectors
        # Shape: (vocab_size, d_model)
        self.embedding = initialize_weights((vocab_size, d_model), method='xavier')

        # Pre-compute positional encodings
        # Shape: (max_seq_len, d_model)
        self.positional_encoding = get_positional_encoding(max_seq_len, d_model)

        # Dropout for embedding + positional encoding
        # Only use dropout if rate > 0
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None

        # Create stack of encoder layers
        self.layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]

        # Set dropout to inference mode for consistency
        if self.dropout is not None:
            self.dropout.set_training_mode(False)
        for layer in self.layers:
            if hasattr(layer, 'dropout1') and layer.dropout1 is not None:
                layer.dropout1.set_training_mode(False)
            if hasattr(layer, 'dropout2') and layer.dropout2 is not None:
                layer.dropout2.set_training_mode(False)

    def forward(self, x, mask=None):
        """
        Apply the encoder to the input.

        Args:
            x: Input token IDs of shape (batch_size, seq_len)
            mask: Optional mask of shape (batch_size, seq_len, seq_len)
                  Used to prevent attention to padding tokens

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
            List of attention weights from each layer (for visualization)

        Example:
            >>> encoder = Encoder(num_layers=6, d_model=512, num_heads=8, d_ff=2048, vocab_size=10000)
            >>> x = np.random.randint(0, 10000, size=(2, 10))
            >>> output, attn_weights = encoder.forward(x)
            >>> output.shape
            (2, 10, 512)
        """
        batch_size, seq_len = x.shape

        # Convert token IDs to embeddings
        # x: (batch_size, seq_len) with integer values
        # embedding: (vocab_size, d_model)
        # embedded: (batch_size, seq_len, d_model)
        embedded = self.embedding[x]

        # Scale embeddings by sqrt(d_model)
        # This helps the embeddings and positional encodings have similar magnitudes
        embedded = embedded * np.sqrt(self.d_model)

        # Add positional encoding
        # positional_encoding: (max_seq_len, d_model)
        # We take the first seq_len positions
        pe = self.positional_encoding[:seq_len, :]
        # Broadcast to batch: (batch_size, seq_len, d_model)
        pe = pe[np.newaxis, :, :]
        embedded = embedded + pe

        # Apply dropout if enabled
        if self.dropout is not None:
            embedded = self.dropout.forward(embedded)

        # Pass through each encoder layer
        x = embedded
        all_attention_weights = []

        for layer in self.layers:
            x, attn_weights = layer.forward(x, mask)
            all_attention_weights.append(attn_weights)

        return x, all_attention_weights
