"""
Attention mechanisms for the Transformer implementation.

This module implements the core attention components:
- Scaled Dot-Product Attention: The fundamental attention operation
- Multi-Head Attention: Multiple attention heads in parallel

The attention mechanism allows the model to focus on different parts of the
input sequence when producing each part of the output sequence.
"""

import numpy as np
from .utils import softmax, initialize_weights, initialize_bias


class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention.

    This is the core attention operation from the Transformer paper.
    It computes attention weights using the dot product of queries and keys,
    scaled by the square root of the key dimension.

    The formula is:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Where:
    - Q (Query): What we're looking for
    - K (Key): What the input contains
    - V (Value): The actual content we want to retrieve
    - d_k: Dimension of the keys (used for scaling)

    The scaling by sqrt(d_k) prevents the dot products from becoming too large,
    which would push the softmax into regions with extremely small gradients.

    Attributes:
        d_k: Dimension of the keys (and queries)
    """

    def __init__(self, d_k):
        """
        Initialize the attention mechanism.

        Args:
            d_k: Dimension of the keys/queries
        """
        self.d_k = d_k

    def forward(self, queries, keys, values, mask=None):
        """
        Compute scaled dot-product attention.

        Args:
            queries: Query matrix of shape (batch_size, seq_len_q, d_k)
            keys: Key matrix of shape (batch_size, seq_len_k, d_k)
            values: Value matrix of shape (batch_size, seq_len_v, d_v)
                    Note: seq_len_k should equal seq_len_v
            mask: Optional mask of shape (batch_size, seq_len_q, seq_len_k)
                  Used to prevent attention to certain positions (e.g., padding)
                  Masked positions should have value -inf before softmax

        Returns:
            output: Attention output of shape (batch_size, seq_len_q, d_v)
            attention_weights: Attention weights of shape (batch_size, seq_len_q, seq_len_k)

        Example:
            >>> attention = ScaledDotProductAttention(d_k=64)
            >>> Q = np.random.randn(2, 10, 64)  # batch=2, seq_len=10, d_k=64
            >>> K = np.random.randn(2, 10, 64)
            >>> V = np.random.randn(2, 10, 64)
            >>> output, weights = attention.forward(Q, K, V)
            >>> output.shape
            (2, 10, 64)
        """
        # Compute dot product of queries and keys
        # Q: (batch_size, seq_len_q, d_k)
        # K: (batch_size, seq_len_k, d_k)
        # QK^T: (batch_size, seq_len_q, seq_len_k)
        # We use einsum for clarity: 'bqd,bkd->bqk' means:
        #   b: batch dimension (same for all)
        #   q: query sequence length
        #   k: key sequence length
        #   d: feature dimension (summed over)
        scores = np.einsum('bqd,bkd->bqk', queries, keys)

        # Scale by sqrt(d_k) to prevent vanishing gradients
        # This is crucial - without scaling, large dot products would
        # push softmax into regions with tiny gradients
        scores = scores / np.sqrt(self.d_k)

        # Apply mask if provided
        # Mask positions with -inf so they become 0 after softmax
        if mask is not None:
            scores = scores + (mask * -1e9)

        # Apply softmax to get attention weights
        # These represent how much each query position should attend to each key position
        attention_weights = softmax(scores, axis=-1)

        # Compute weighted sum of values
        # attention_weights: (batch_size, seq_len_q, seq_len_k)
        # V: (batch_size, seq_len_v, d_v) where seq_len_v = seq_len_k
        # output: (batch_size, seq_len_q, d_v)
        output = np.einsum('bqk,bkd->bqd', attention_weights, values)

        return output, attention_weights


class MultiHeadAttention:
    """
    Multi-Head Attention.

    Instead of performing a single attention function, multi-head attention
    runs multiple attention operations in parallel, allowing the model to
    attend to different representation subspaces at different positions.

    The formula is:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
        where head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)

    Where:
    - h: Number of attention heads
    - W_i^Q, W_i^K, W_i^V: Projection matrices for each head
    - W^O: Output projection matrix

    This allows the model to jointly attend to information from different
    representation subspaces at different positions.

    Attributes:
        d_model: Total dimension of the model
        num_heads: Number of attention heads
        d_k: Dimension of each head (d_model / num_heads)
        d_v: Dimension of values per head (usually equal to d_k)
        W_q: Query projection weights
        W_k: Key projection weights
        W_v: Value projection weights
        W_o: Output projection weights
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Total dimension of the model
            num_heads: Number of attention heads (must divide d_model evenly)
        """
        self.d_model = d_model
        self.num_heads = num_heads

        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Dimension per head
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Initialize projection matrices
        # W_q: (d_model, d_k) - projects queries to d_k per head
        # W_k: (d_model, d_k) - projects keys to d_k per head
        # W_v: (d_model, d_v) - projects values to d_v per head
        # W_o: (d_model, d_model) - projects concatenated heads back to d_model
        self.W_q = initialize_weights((d_model, self.d_k * num_heads), method='xavier')
        self.W_k = initialize_weights((d_model, self.d_k * num_heads), method='xavier')
        self.W_v = initialize_weights((d_model, self.d_v * num_heads), method='xavier')
        self.W_o = initialize_weights((d_model, d_model), method='xavier')

        # Initialize bias terms (optional, but common)
        self.b_q = initialize_bias(self.d_k * num_heads)
        self.b_k = initialize_bias(self.d_k * num_heads)
        self.b_v = initialize_bias(self.d_v * num_heads)
        self.b_o = initialize_bias(d_model)

        # Scaled dot-product attention for each head
        self.attention = ScaledDotProductAttention(self.d_k)

    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, depth).

        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape

        # Reshape to split the last dimension
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose to bring num_heads before seq_len
        # (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x):
        """
        Combine the heads back.

        Transpose and reshape to merge the last two dimensions.

        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, d_k)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape

        # Transpose back
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(0, 2, 1, 3)

        # Reshape to merge the last two dimensions
        # (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        return x.reshape(batch_size, seq_len, self.d_model)

    def forward(self, queries, keys, values, mask=None):
        """
        Compute multi-head attention.

        Args:
            queries: Query matrix of shape (batch_size, seq_len_q, d_model)
            keys: Key matrix of shape (batch_size, seq_len_k, d_model)
            values: Value matrix of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask of shape (batch_size, seq_len_q, seq_len_k)

        Returns:
            output: Attention output of shape (batch_size, seq_len_q, d_model)
            attention_weights: Attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)

        Example:
            >>> mha = MultiHeadAttention(d_model=512, num_heads=8)
            >>> Q = np.random.randn(2, 10, 512)
            >>> K = np.random.randn(2, 10, 512)
            >>> V = np.random.randn(2, 10, 512)
            >>> output, weights = mha.forward(Q, K, V)
            >>> output.shape
            (2, 10, 512)
        """
        batch_size = queries.shape[0]

        # Linear projections
        # Project Q, K, V to d_k * num_heads dimension (split across heads)
        # (batch_size, seq_len, d_model) @ (d_model, d_k * num_heads)
        # -> (batch_size, seq_len, d_k * num_heads)
        Q = np.dot(queries, self.W_q) + self.b_q
        K = np.dot(keys, self.W_k) + self.b_k
        V = np.dot(values, self.W_v) + self.b_v

        # Split into multiple heads
        # (batch_size, seq_len, d_k * num_heads) -> (batch_size, num_heads, seq_len, d_k)
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Apply scaled dot-product attention to each head
        # For each head, compute attention independently
        # After splitting, we have shape (batch_size, num_heads, seq_len, d_k)
        # We need to reshape to apply attention to all heads at once
        batch_size, num_heads, seq_len_q, d_k = Q.shape
        _, _, seq_len_k, _ = K.shape

        # Reshape: (batch_size * num_heads, seq_len, d_k)
        Q_flat = Q.reshape(batch_size * num_heads, seq_len_q, d_k)
        K_flat = K.reshape(batch_size * num_heads, seq_len_k, d_k)
        V_flat = V.reshape(batch_size * num_heads, seq_len_k, self.d_v)

        # Apply attention
        # Reshape mask if needed
        if mask is not None:
            # Broadcast mask to all heads
            # mask shape: (batch_size, seq_len_q, seq_len_k) or (1, seq_len_q, seq_len_k)
            # We need: (batch_size * num_heads, seq_len_q, seq_len_k)
            if mask.shape[0] == 1:
                # Broadcast mask to batch size
                mask = np.repeat(mask, batch_size, axis=0)
            mask = np.repeat(mask[:, np.newaxis, :, :], num_heads, axis=1)
            mask = mask.reshape(batch_size * num_heads, seq_len_q, seq_len_k)

        attention_output, attention_weights = self.attention.forward(Q_flat, K_flat, V_flat, mask)

        # Reshape back to separate heads
        # (batch_size * num_heads, seq_len_q, d_v) -> (batch_size, num_heads, seq_len_q, d_v)
        attention_output = attention_output.reshape(batch_size, num_heads, seq_len_q, self.d_v)
        attention_weights = attention_weights.reshape(batch_size, num_heads, seq_len_q, seq_len_k)

        # Combine heads
        # (batch_size, num_heads, seq_len, d_v) -> (batch_size, seq_len, d_model)
        output = self._combine_heads(attention_output)

        # Final linear projection
        # (batch_size, seq_len, d_model) @ (d_model, d_model) -> (batch_size, seq_len, d_model)
        output = np.dot(output, self.W_o) + self.b_o

        return output, attention_weights
