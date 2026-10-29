"""
Complete Transformer model implementation.

This module implements the full Transformer architecture that combines
the encoder and decoder for sequence-to-sequence tasks.

The Transformer takes an input sequence, processes it through the encoder,
and generates an output sequence through the decoder using cross-attention
to incorporate information from the encoder.
"""

import numpy as np
from .encoder import Encoder
from .decoder import Decoder, create_look_ahead_mask


class Transformer:
    """
    The complete Transformer model.

    The Transformer consists of:
    1. Encoder: Processes the input sequence
    2. Decoder: Generates the output sequence using encoder information

    The architecture follows the original "Attention Is All You Need" paper:
    - Input tokens -> Encoder -> Contextualized representations
    - Output tokens + Encoder output -> Decoder -> Output logits

    The model supports:
    - Masking for padding tokens
    - Look-ahead masking for autoregressive generation
    - Multiple encoder and decoder layers

    Attributes:
        num_layers: Number of encoder/decoder layers
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of the feed-forward network
        dropout_rate: Dropout probability
        vocab_size: Size of the vocabulary (assumed same for input and output)
        max_seq_len: Maximum sequence length
        encoder: The encoder stack
        decoder: The decoder stack
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size,
                 max_seq_len=512, dropout_rate=0.1):
        """
        Initialize the Transformer.

        Args:
            num_layers: Number of encoder and decoder layers (typically 6)
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

        # Initialize encoder and decoder
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            dropout_rate=dropout_rate
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            dropout_rate=dropout_rate
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass through the Transformer.

        Args:
            src: Source (input) token IDs of shape (batch_size, src_seq_len)
            tgt: Target (output) token IDs of shape (batch_size, tgt_seq_len)
            src_mask: Optional mask for encoder attention of shape (batch_size, src_seq_len, src_seq_len)
                     Used to prevent attention to padding tokens
            tgt_mask: Optional mask for decoder self-attention of shape (tgt_seq_len, tgt_seq_len)
                     Typically a look-ahead mask to prevent attending to future positions

        Returns:
            output: Output logits of shape (batch_size, tgt_seq_len, vocab_size)
            encoder_attention_weights: List of attention weights from encoder layers
            decoder_attention_weights: Tuple of (self_attn_weights, cross_attn_weights) from decoder layers

        Example:
            >>> transformer = Transformer(num_layers=6, d_model=512, num_heads=8, d_ff=2048, vocab_size=10000)
            >>> src = np.random.randint(0, 10000, size=(2, 10))
            >>> tgt = np.random.randint(0, 10000, size=(2, 8))
            >>> output, enc_attn, dec_attn = transformer.forward(src, tgt)
            >>> output.shape
            (2, 8, 10000)
        """
        # Pass source through encoder
        encoder_output, encoder_attention_weights = self.encoder.forward(src, src_mask)

        # Create look-ahead mask for decoder if not provided
        if tgt_mask is None:
            tgt_seq_len = tgt.shape[1]
            tgt_mask = create_look_ahead_mask(tgt_seq_len)

        # Pass target and encoder output through decoder
        output, decoder_attention_weights = self.decoder.forward(
            tgt, encoder_output, look_ahead_mask=tgt_mask, padding_mask=None
        )

        return output, encoder_attention_weights, decoder_attention_weights

    def encode(self, src, src_mask=None):
        """
        Encode the source sequence.

        This is useful when you want to encode once and decode multiple times
        (e.g., during generation with beam search).

        Args:
            src: Source token IDs of shape (batch_size, src_seq_len)
            src_mask: Optional mask for encoder attention

        Returns:
            encoder_output: Encoded representations of shape (batch_size, src_seq_len, d_model)
            encoder_attention_weights: List of attention weights from encoder layers
        """
        return self.encoder.forward(src, src_mask)

    def decode(self, tgt, encoder_output, tgt_mask=None, padding_mask=None):
        """
        Decode the target sequence given encoder output.

        Args:
            tgt: Target token IDs of shape (batch_size, tgt_seq_len)
            encoder_output: Output from encoder of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Optional look-ahead mask of shape (tgt_seq_len, tgt_seq_len)
            padding_mask: Optional padding mask for cross-attention

        Returns:
            output: Output logits of shape (batch_size, tgt_seq_len, vocab_size)
            decoder_attention_weights: Tuple of (self_attn_weights, cross_attn_weights)
        """
        return self.decoder.forward(tgt, encoder_output, tgt_mask, padding_mask)
