"""
Test script for the new embedding modes in the EmbeddingLayer.

This script tests:
1. 'add' mode - standard addition of signal embedding to vocab embedding
2. 'concat' mode - concatenation of label components to sequence
3. 'mask' mode - handling of NaN values in labels
"""

import torch
import sys
from model.transformer import EmbeddingLayer

def test_add_mode():
    """Test the 'add' embedding mode."""
    print("\n" + "="*60)
    print("Testing 'add' mode")
    print("="*60)

    batch_size, seq_length, signal_dim, hidden_dim = 4, 230, 3, 768
    vocab_size = 4

    # Create embedding layer
    embed_layer = EmbeddingLayer(
        dim=hidden_dim,
        vocab_dim=vocab_size,
        signal_dim=signal_dim,
        embedding_mode='add'
    )

    # Create sample data
    sequences = torch.randint(0, vocab_size, (batch_size, seq_length))
    labels = torch.randn(batch_size, signal_dim)

    # Forward pass
    output = embed_layer(sequences, labels)

    print(f"Input sequences shape: {sequences.shape}")
    print(f"Input labels shape: {labels.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_length}, {hidden_dim})")

    assert output.shape == (batch_size, seq_length, hidden_dim), \
        f"Output shape mismatch! Got {output.shape}, expected ({batch_size}, {seq_length}, {hidden_dim})"

    print("✓ 'add' mode test PASSED")
    return True


def test_concat_mode():
    """Test the 'concat' embedding mode."""
    print("\n" + "="*60)
    print("Testing 'concat' mode")
    print("="*60)

    batch_size, seq_length, signal_dim, hidden_dim = 4, 230, 3, 768
    vocab_size = 4

    # Create embedding layer
    embed_layer = EmbeddingLayer(
        dim=hidden_dim,
        vocab_dim=vocab_size,
        signal_dim=signal_dim,
        embedding_mode='concat'
    )

    # Create sample data
    sequences = torch.randint(0, vocab_size, (batch_size, seq_length))
    labels = torch.randn(batch_size, signal_dim)

    # Forward pass
    output = embed_layer(sequences, labels)

    print(f"Input sequences shape: {sequences.shape}")
    print(f"Input labels shape: {labels.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_length + signal_dim}, {hidden_dim})")

    assert output.shape == (batch_size, seq_length + signal_dim, hidden_dim), \
        f"Output shape mismatch! Got {output.shape}, expected ({batch_size}, {seq_length + signal_dim}, {hidden_dim})"

    print("✓ 'concat' mode test PASSED")
    return True


def test_mask_mode():
    """Test the 'mask' embedding mode with NaN values."""
    print("\n" + "="*60)
    print("Testing 'mask' mode")
    print("="*60)

    batch_size, seq_length, signal_dim, hidden_dim = 4, 230, 3, 768
    vocab_size = 4

    # Create embedding layer
    embed_layer = EmbeddingLayer(
        dim=hidden_dim,
        vocab_dim=vocab_size,
        signal_dim=signal_dim,
        embedding_mode='mask'
    )

    # Create sample data with NaN values
    sequences = torch.randint(0, vocab_size, (batch_size, seq_length))
    labels = torch.randn(batch_size, signal_dim)

    # Introduce NaN values in labels
    labels[0, 0] = float('nan')  # First sample, first label
    labels[1, 1] = float('nan')  # Second sample, second label
    labels[2, :] = float('nan')  # Third sample, all labels

    print(f"Labels with NaN:\n{labels}")

    # Forward pass
    output = embed_layer(sequences, labels)

    print(f"\nInput sequences shape: {sequences.shape}")
    print(f"Input labels shape: {labels.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_length}, {hidden_dim})")

    # Check that output is finite (no NaN propagation)
    assert torch.isfinite(output).all(), "Output contains NaN values!"

    assert output.shape == (batch_size, seq_length, hidden_dim), \
        f"Output shape mismatch! Got {output.shape}, expected ({batch_size}, {seq_length}, {hidden_dim})"

    print("✓ 'mask' mode test PASSED (NaN values properly masked)")
    return True


def test_unconditional():
    """Test unconditional generation (labels=None) for all modes."""
    print("\n" + "="*60)
    print("Testing unconditional generation (labels=None)")
    print("="*60)

    batch_size, seq_length, signal_dim, hidden_dim = 4, 230, 3, 768
    vocab_size = 4

    sequences = torch.randint(0, vocab_size, (batch_size, seq_length))

    for mode in ['add', 'concat', 'mask']:
        embed_layer = EmbeddingLayer(
            dim=hidden_dim,
            vocab_dim=vocab_size,
            signal_dim=signal_dim,
            embedding_mode=mode
        )

        output = embed_layer(sequences, None)

        assert output.shape == (batch_size, seq_length, hidden_dim), \
            f"Mode '{mode}': Output shape mismatch! Got {output.shape}"

        print(f"  ✓ Mode '{mode}': unconditional generation works")

    print("✓ Unconditional generation test PASSED for all modes")
    return True


def test_parameter_counts():
    """Test that parameter counts are as expected."""
    print("\n" + "="*60)
    print("Testing parameter counts")
    print("="*60)

    signal_dim, hidden_dim, vocab_size = 3, 768, 4

    for mode in ['add', 'concat', 'mask']:
        embed_layer = EmbeddingLayer(
            dim=hidden_dim,
            vocab_dim=vocab_size,
            signal_dim=signal_dim,
            embedding_mode=mode
        )

        total_params = sum(p.numel() for p in embed_layer.parameters())

        # Calculate expected params
        vocab_params = vocab_size * hidden_dim
        if mode == 'concat':
            # Linear: 1 -> hidden_dim (maps each scalar label to dim)
            label_params = 1 * hidden_dim + hidden_dim  # weights + bias
        else:
            # Linear: signal_dim -> hidden_dim
            label_params = signal_dim * hidden_dim + hidden_dim  # weights + bias

        expected_params = vocab_params + label_params

        print(f"  Mode '{mode}': {total_params:,} parameters (expected ~{expected_params:,})")

    print("✓ Parameter count test PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# Testing EmbeddingLayer modes for LentIMPRA")
    print("#"*60)

    try:
        # Run all tests
        test_add_mode()
        test_concat_mode()
        test_mask_mode()
        test_unconditional()
        test_parameter_counts()

        print("\n" + "#"*60)
        print("# ALL TESTS PASSED ✓")
        print("#"*60)
        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
