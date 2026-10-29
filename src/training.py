"""
Training utilities for the Transformer.

This module provides basic training utilities including loss functions
and a simple training loop skeleton.

Note: This is a forward-only implementation for educational purposes.
For actual training, you would need automatic differentiation (e.g., PyTorch,
TensorFlow, or JAX) to compute gradients and update weights.
"""

import numpy as np


def cross_entropy_loss(logits, targets, padding_idx=None):
    """
    Compute cross-entropy loss.

    Cross-entropy measures the difference between the predicted probability
    distribution and the true distribution (one-hot encoded targets).

    The formula is:
        Loss = -sum(target * log(softmax(logits)))

    Args:
        logits: Predicted logits of shape (batch_size, seq_len, vocab_size)
        targets: Target token IDs of shape (batch_size, seq_len)
        padding_idx: Optional padding token index (loss is not computed for these)

    Returns:
        Scalar loss value

    Example:
        >>> logits = np.random.randn(2, 10, 1000)
        >>> targets = np.random.randint(0, 1000, size=(2, 10))
        >>> loss = cross_entropy_loss(logits, targets)
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Apply softmax to get probabilities
    # logits: (batch_size, seq_len, vocab_size)
    # Shift by max for numerical stability
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Create one-hot encoding of targets
    # targets: (batch_size, seq_len)
    # target_one_hot: (batch_size, seq_len, vocab_size)
    target_one_hot = np.zeros((batch_size, seq_len, vocab_size))
    for b in range(batch_size):
        for t in range(seq_len):
            target_one_hot[b, t, targets[b, t]] = 1

    # Compute cross-entropy loss
    # Loss = -sum(target * log(probs))
    # Add small epsilon to log to avoid log(0)
    epsilon = 1e-10
    log_probs = np.log(probs + epsilon)
    loss = -np.sum(target_one_hot * log_probs)

    # Average over all non-padding tokens
    if padding_idx is not None:
        # Create mask for non-padding tokens
        mask = (targets != padding_idx).astype(float)
        num_valid = np.sum(mask)
        if num_valid > 0:
            loss = loss / num_valid
    else:
        loss = loss / (batch_size * seq_len)

    return loss


def accuracy(logits, targets, padding_idx=None):
    """
    Compute prediction accuracy.

    Args:
        logits: Predicted logits of shape (batch_size, seq_len, vocab_size)
        targets: Target token IDs of shape (batch_size, seq_len)
        padding_idx: Optional padding token index (not counted in accuracy)

    Returns:
        Accuracy value between 0 and 1
    """
    # Get predictions (argmax over vocabulary dimension)
    predictions = np.argmax(logits, axis=-1)

    # Compare with targets
    correct = (predictions == targets).astype(float)

    # Mask out padding tokens if specified
    if padding_idx is not None:
        mask = (targets != padding_idx).astype(float)
        correct = correct * mask
        num_valid = np.sum(mask)
    else:
        num_valid = correct.size

    if num_valid > 0:
        acc = np.sum(correct) / num_valid
    else:
        acc = 0.0

    return acc


class TrainingLoop:
    """
    Simple training loop skeleton.

    Note: This is a skeleton for educational purposes. For actual training,
    you would need to implement:
    1. Automatic differentiation to compute gradients
    2. Optimizer (e.g., SGD, Adam) to update weights
    3. Learning rate scheduling
    4. Gradient clipping
    5. Checkpointing

    This class shows the structure of a training loop, but the actual
    weight updates are not implemented since we're using pure NumPy.
    """

    def __init__(self, model, learning_rate=0.001):
        """
        Initialize the training loop.

        Args:
            model: The Transformer model
            learning_rate: Learning rate for weight updates
        """
        self.model = model
        self.learning_rate = learning_rate

    def train_step(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Perform a single training step.

        Args:
            src: Source token IDs
            tgt: Target token IDs
            src_mask: Optional source mask
            tgt_mask: Optional target mask

        Returns:
            Dictionary with loss and metrics
        """
        # Forward pass
        logits, enc_attn, dec_attn = self.model.forward(src, tgt, src_mask, tgt_mask)

        # Compute loss
        loss = cross_entropy_loss(logits, tgt)

        # Compute accuracy
        acc = accuracy(logits, tgt)

        # Note: In a real implementation, you would:
        # 1. Compute gradients of loss w.r.t. parameters
        # 2. Update parameters using optimizer
        # 3. Apply gradient clipping if needed

        return {
            'loss': loss,
            'accuracy': acc
        }

    def train_epoch(self, dataloader):
        """
        Train for one epoch.

        Args:
            dataloader: Iterable that yields batches of (src, tgt)

        Returns:
            Dictionary with average loss and accuracy for the epoch
        """
        total_loss = 0
        total_acc = 0
        num_batches = 0

        for batch in dataloader:
            src, tgt = batch

            # Training step
            metrics = self.train_step(src, tgt)

            total_loss += metrics['loss']
            total_acc += metrics['accuracy']
            num_batches += 1

        # Compute averages
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0

        return {
            'loss': avg_loss,
            'accuracy': avg_acc
        }
