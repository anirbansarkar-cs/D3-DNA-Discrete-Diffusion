"""
Quick test to verify embedding modes work correctly.
"""
import torch
from model.transformer import EmbeddingLayer

def test_embedding_mode(mode, signal_dim=3):
    """Test a specific embedding mode."""
    print(f"\n{'='*60}")
    print(f"Testing mode: {mode} with signal_dim={signal_dim}")
    print(f"{'='*60}")

    # Create embedding layer
    dim = 768
    vocab_dim = 5  # A, C, G, T + absorb
    seq_length = 230
    batch_size = 4

    embedding_layer = EmbeddingLayer(
        dim=dim,
        vocab_dim=vocab_dim,
        signal_dim=signal_dim,
        embedding_mode=mode
    )

    # Create test data
    x = torch.randint(0, vocab_dim, (batch_size, seq_length))
    y = torch.randn(batch_size, signal_dim)

    # Forward pass
    try:
        output = embedding_layer(x, y)
        print(f"✓ Input shapes:")
        print(f"  - x (sequence): {x.shape}")
        print(f"  - y (labels): {y.shape}")
        print(f"✓ Output shape: {output.shape}")

        if mode == 'concat':
            expected_seq_len = seq_length + signal_dim
            assert output.shape == (batch_size, expected_seq_len, dim), \
                f"Expected {(batch_size, expected_seq_len, dim)}, got {output.shape}"
            print(f"✓ Concat mode: sequence extended from {seq_length} to {expected_seq_len}")
        else:
            assert output.shape == (batch_size, seq_length, dim), \
                f"Expected {(batch_size, seq_length, dim)}, got {output.shape}"
            print(f"✓ Add/mask mode: sequence length preserved at {seq_length}")

        print(f"✓ SUCCESS: {mode} mode works correctly!")
        return True

    except Exception as e:
        print(f"✗ FAILED: {mode} mode")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\nTesting Embedding Modes")
    print("="*60)

    results = {}

    # Test all modes
    for mode in ['add', 'concat', 'mask']:
        results[mode] = test_embedding_mode(mode, signal_dim=3)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for mode, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {mode} mode")

    all_passed = all(results.values())
    if all_passed:
        print(f"\n{'='*60}")
        print("ALL TESTS PASSED!")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("SOME TESTS FAILED")
        print(f"{'='*60}\n")
        exit(1)
