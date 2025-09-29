#!/usr/bin/env python3
"""
Simple DeepSTARR Sequence Sampling Script

Sample sequences from a trained D3 model and save as one-hot encodings in H5 format.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from typing import Optional, Tuple
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts import sampling
from model_zoo.deepstarr.models import load_trained_model
from model_zoo.deepstarr.data import get_deepstarr_datasets


def sample_sequences_onehot(model_checkpoint: str,
                           config_path: str,
                           num_samples: int = 1000,
                           num_steps: int = 100,
                           architecture: str = 'transformer',
                           conditioning_labels: Optional[torch.Tensor] = None,
                           batch_size: int = 128,
                           device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample sequences from trained D3 model and return as one-hot encodings.

    Args:
        model_checkpoint: Path to trained D3 model checkpoint
        config_path: Path to model configuration file
        num_samples: Number of sequences to sample
        num_steps: Number of sampling steps (default: 100)
        architecture: Model architecture ('transformer' or 'convolutional')
        conditioning_labels: Optional conditioning labels (N, 2) for DeepSTARR
        batch_size: Sampling batch size
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Tuple of (sequences_onehot, labels) where:
        - sequences_onehot: (N, L, 4) one-hot encoded sequences
        - labels: (N, 2) conditioning labels used for sampling
    """
    device = torch.device(device)

    # Load configuration
    config = OmegaConf.load(config_path)

    # Load trained model
    print(f"Loading model from: {model_checkpoint}")
    model, graph, noise = load_trained_model(model_checkpoint, config, architecture, device)
    model.eval()

    # Generate conditioning labels if not provided
    if conditioning_labels is None:
        print(f"Generating random conditioning labels for {num_samples} samples...")
        # DeepSTARR: 2 activities (Dev, HK) - generate realistic range
        conditioning_labels = torch.randn(num_samples, 2, device=device) * 2.0  # Scale to ±2
    else:
        conditioning_labels = conditioning_labels.to(device)
        num_samples = len(conditioning_labels)

    # Sample sequences in batches
    sequence_length = 249  # DeepSTARR sequence length
    sampled_sequences = []
    sampled_labels = []

    num_batches = (num_samples + batch_size - 1) // batch_size
    print(f"Sampling {num_samples} sequences in {num_batches} batches...")

    for i in tqdm(range(num_batches), desc="Sampling batches"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx

        batch_labels = conditioning_labels[start_idx:end_idx]

        # Create PC sampler for this batch
        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (current_batch_size, sequence_length), 'analytic',
            num_steps, device=device
        )

        # Sample sequences (returns token indices)
        with torch.amp.autocast('cuda', enabled=True):
            sample_indices = sampling_fn(model, batch_labels)

        # Convert to one-hot encoding: (B, L) -> (B, L, 4)
        sample_onehot = F.one_hot(sample_indices, num_classes=4).float()

        sampled_sequences.append(sample_onehot.cpu())
        sampled_labels.append(batch_labels.cpu())

        # Clear GPU cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()

    # Concatenate all batches
    final_sequences = torch.cat(sampled_sequences, dim=0)  # (N, L, 4)
    final_labels = torch.cat(sampled_labels, dim=0)  # (N, 2)

    print(f"✓ Sampled {len(final_sequences)} sequences")
    print(f"  Sequence shape: {final_sequences.shape}")
    print(f"  Labels shape: {final_labels.shape}")

    return final_sequences, final_labels


def save_sequences_h5(sequences_onehot: torch.Tensor,
                     labels: torch.Tensor,
                     output_path: str,
                     metadata: Optional[dict] = None):
    """
    Save one-hot encoded sequences and labels to H5 file.

    Args:
        sequences_onehot: (N, L, 4) one-hot encoded sequences
        labels: (N, 2) conditioning labels
        output_path: Output H5 file path
        metadata: Optional metadata dictionary
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving sequences to: {output_path}")

    with h5py.File(output_path, 'w') as f:
        # Save sequences as one-hot (N, L, 4)
        f.create_dataset('sequences_onehot', data=sequences_onehot.numpy(),
                        compression='gzip', compression_opts=9)

        # Save labels
        f.create_dataset('labels', data=labels.numpy(),
                        compression='gzip', compression_opts=9)

        # Save metadata
        f.attrs['num_samples'] = len(sequences_onehot)
        f.attrs['sequence_length'] = sequences_onehot.shape[1]
        f.attrs['vocab_size'] = sequences_onehot.shape[2]
        f.attrs['label_dim'] = labels.shape[1]
        f.attrs['format'] = 'one_hot_ACGT'  # A=0, C=1, G=2, T=3

        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value

    print(f"✓ Saved {len(sequences_onehot)} one-hot sequences to {output_path}")


def load_conditioning_from_test_set(data_path: str, num_samples: int) -> torch.Tensor:
    """Load conditioning labels from DeepSTARR test set."""
    print(f"Loading conditioning labels from test set: {data_path}")

    try:
        train_ds, val_ds, test_ds = get_deepstarr_datasets(data_path)

        # Extract test labels
        test_labels = []
        for _, label in test_ds:
            test_labels.append(label)
        test_labels = torch.stack(test_labels)

        print(f"Found {len(test_labels)} test samples")

        # Sample or repeat to get desired number
        if len(test_labels) >= num_samples:
            indices = torch.randperm(len(test_labels))[:num_samples]
            sampled_labels = test_labels[indices]
        else:
            # Repeat test labels to reach target
            repeats = (num_samples + len(test_labels) - 1) // len(test_labels)
            repeated_labels = test_labels.repeat(repeats, 1)
            sampled_labels = repeated_labels[:num_samples]

        print(f"Using {len(sampled_labels)} conditioning labels from test set")
        return sampled_labels

    except Exception as e:
        print(f"Error loading test set: {e}")
        print("Falling back to random conditioning labels")
        return None


def main():
    parser = argparse.ArgumentParser(description='Sample sequences from trained D3 model and save as one-hot H5')
    parser.add_argument('--checkpoint', required=True, help='Path to trained D3 model checkpoint')
    parser.add_argument('--config', required=True, help='Path to model config file')
    parser.add_argument('--output', required=True, help='Output H5 file path')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of sequences to sample')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of sampling steps')
    parser.add_argument('--architecture', choices=['transformer', 'convolutional'],
                       default='transformer', help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=128, help='Sampling batch size')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--data_path', help='DeepSTARR data file for test set conditioning (optional)')
    parser.add_argument('--random_labels', action='store_true',
                       help='Use random labels instead of test set conditioning')

    args = parser.parse_args()

    # Determine conditioning strategy
    conditioning_labels = None
    if not args.random_labels and args.data_path:
        conditioning_labels = load_conditioning_from_test_set(args.data_path, args.num_samples)

    # Sample sequences
    sequences_onehot, labels = sample_sequences_onehot(
        model_checkpoint=args.checkpoint,
        config_path=args.config,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        architecture=args.architecture,
        conditioning_labels=conditioning_labels,
        batch_size=args.batch_size,
        device=args.device
    )

    # Save to H5 file
    metadata = {
        'model_checkpoint': args.checkpoint,
        'config_path': args.config,
        'num_steps': args.num_steps,
        'architecture': args.architecture,
        'conditioning_source': 'test_set' if conditioning_labels is not None else 'random'
    }

    save_sequences_h5(sequences_onehot, labels, args.output, metadata)

    print(f"\n✓ Sampling completed successfully!")
    print(f"Output: {args.output}")
    print(f"Sequences: {sequences_onehot.shape} (one-hot format)")
    print(f"Labels: {labels.shape}")


if __name__ == '__main__':
    main()