"""
Unit tests for normalization and dropout.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import src module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.normalization import LayerNormalization, Dropout


def test_layer_normalization():
    """Test layer normalization."""
    print("Testing LayerNormalization...")

    batch_size = 2
    seq_len = 10
    d_model = 512

    ln = LayerNormalization(d_model=d_model)

    # Create dummy input
    x = np.random.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = ln.forward(x)

    # Check output shape
    assert output.shape == x.shape, f"Output shape should match input shape, got {output.shape}"
    print("  [PASS] Output shape correct")

    # Check that each position has mean close to 0 and std close to 1
    # (after normalization)
    mean = np.mean(output, axis=-1)
    std = np.std(output, axis=-1)

    assert np.allclose(mean, 0, atol=1e-5), "Mean should be close to 0"
    assert np.allclose(std, 1, atol=1e-5), "Std should be close to 1"
    print("  [PASS] Normalization correct (mean=0, std=1)")


def test_layer_normalization_preserves_shape():
    """Test that layer normalization preserves the input shape."""
    print("Testing LayerNormalization shape preservation...")

    d_model = 512
    ln = LayerNormalization(d_model=d_model)

    # Test with different batch sizes and sequence lengths
    for batch_size in [1, 2, 4]:
        for seq_len in [5, 10, 20]:
            x = np.random.randn(batch_size, seq_len, d_model)
            output = ln.forward(x)
            assert output.shape == x.shape, \
                f"Shape mismatch for batch={batch_size}, seq_len={seq_len}"

    print("  [PASS] Shape preserved for various input sizes")


def test_dropout_training_mode():
    """Test dropout in training mode."""
    print("Testing Dropout in training mode...")

    dropout = Dropout(dropout_rate=0.5, training=True)

    # Create dummy input
    x = np.ones((1000,))  # Use ones to easily check dropout

    # Forward pass
    output = dropout.forward(x)

    # Check that some values are zeroed (dropped)
    assert np.any(output == 0), "Some values should be dropped"
    assert np.any(output != 0), "Some values should be kept"

    # Check that approximately 50% are dropped
    zero_fraction = np.mean(output == 0)
    assert 0.4 < zero_fraction < 0.6, f"Expected ~50% dropout, got {zero_fraction * 100}%"

    print("  [PASS] Dropout applied correctly in training mode")


def test_dropout_inference_mode():
    """Test dropout in inference mode."""
    print("Testing Dropout in inference mode...")

    dropout = Dropout(dropout_rate=0.5, training=False)

    # Create dummy input
    x = np.random.randn(100, 10, 512)

    # Forward pass
    output = dropout.forward(x)

    # Check that output is identical to input (no dropout)
    assert np.allclose(output, x), "Output should be identical to input in inference mode"

    print("  [PASS] No dropout in inference mode")


def test_dropout_scale_correction():
    """Test that dropout scales correctly to maintain expected value."""
    print("Testing Dropout scale correction...")

    dropout = Dropout(dropout_rate=0.5, training=True)

    # Create input with known mean
    x = np.ones((10000,)) * 10.0  # Mean = 10

    # Forward pass
    output = dropout.forward(x)

    # Check that the mean is preserved (approximately)
    # The scaling by 1/(1-p) should maintain the expected value
    output_mean = np.mean(output)
    assert 9.5 < output_mean < 10.5, f"Expected mean ~10, got {output_mean}"

    print("  [PASS] Scale correction maintains expected value")


def test_dropout_training_mode_switch():
    """Test switching between training and inference modes."""
    print("Testing Dropout mode switching...")

    dropout = Dropout(dropout_rate=0.5, training=True)

    x = np.random.randn(100, 10, 512)

    # Training mode
    output_train = dropout.forward(x)
    assert not np.allclose(output_train, x), "Training mode should apply dropout"

    # Switch to inference mode
    dropout.set_training_mode(False)
    output_infer = dropout.forward(x)
    assert np.allclose(output_infer, x), "Inference mode should not apply dropout"

    # Switch back to training mode
    dropout.set_training_mode(True)
    output_train2 = dropout.forward(x)
    assert not np.allclose(output_train2, x), "Training mode should apply dropout again"

    print("  [PASS] Mode switching works correctly")


if __name__ == "__main__":
    print("\n=== Running Normalization Tests ===\n")

    test_layer_normalization()
    test_layer_normalization_preserves_shape()
    test_dropout_training_mode()
    test_dropout_inference_mode()
    test_dropout_scale_correction()
    test_dropout_training_mode_switch()

    print("\n=== All Normalization Tests Passed! ===\n")
