#!/usr/bin/env python3
"""
Minimal function to sample sequences with a trained LentIMPRA model.

Based on model_zoo/lentimpra/sp_mse_callback.py
"""

import argparse
import os
import torch
import h5py
from omegaconf import OmegaConf, DictConfig
from typing import Optional, cast
import numpy as np

from model_zoo.lentimpra.models import load_trained_model
from scripts import sampling



def sample_lentimpra_sequences(
    checkpoint_path: str,
    config_path: str,
    num_samples: int,
    labels: Optional[torch.Tensor] = None,
    architecture: str = 'transformer',
    sampling_steps: int = 230,
    device: str = 'cuda',
    batch_size: int = 256
) -> torch.Tensor:
    """
    Sample sequences from a trained LentIMPRA model.

    Args:
        checkpoint_path: Path to trained model checkpoint
        config_path: Path to model config file
        num_samples: Number of sequences to generate
        labels: Optional conditioning labels (num_samples, signal_dim). If None, uses random labels.
        architecture: Model architecture ('transformer' or 'convolutional')
        sampling_steps: Number of diffusion steps (default: 230, sequence length)
        device: Device to run on ('cuda' or 'cpu')
        batch_size: Batch size for sampling (default: 256)

    Returns:
        Generated sequences as indices (num_samples, 230)

    Example:
        >>> config_path = 'model_zoo/lentimpra/configs/transformer.yaml'
        >>> checkpoint_path = 'path/to/checkpoint.ckpt'
        >>> sequences = sample_lentimpra_sequences(checkpoint_path, config_path, num_samples=100)
        >>> print(sequences.shape)  # (100, 230)
    """
    # Load config
    config = cast(DictConfig, OmegaConf.load(config_path))

    # Set device (torch)
    torch_device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load trained model
    model, graph, noise = load_trained_model(
        checkpoint_path,
        config,
        architecture,
        device=str(torch_device)
    )
    model.eval()

    # Generate or validate labels
    if labels is None:
        # Random regulatory activity values
        signal_dim = config.dataset.get('signal_dim', 1)
        labels = torch.randn(num_samples, signal_dim, device=torch_device)
    else:
        if labels.shape[0] != num_samples:
            raise ValueError(f"labels.shape[0] ({labels.shape[0]}) must match num_samples ({num_samples})")
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        labels = labels.to(torch_device)

    # Sample in batches to avoid memory issues with flash attention
    seq_length = 230  # LentIMPRA sequence length
    all_sequences = []

    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        current_batch_size = end_idx - start_idx

        # Get labels for this batch
        batch_labels = labels[start_idx:end_idx]

        # Create sampling function for this batch
        sampling_fn = sampling.get_pc_sampler(
            graph,
            noise,
            (current_batch_size, seq_length),
            'analytic',  # Predictor type
            sampling_steps,
            device=torch_device
        )

        # Generate sequences for this batch
        with torch.no_grad():
            batch_sequences = sampling_fn(model, batch_labels)

        all_sequences.append(batch_sequences)

        print(f"Completed batch {i+1}/{num_batches} ({end_idx}/{num_samples} sequences)")

    # Concatenate all batches
    sequences = torch.cat(all_sequences, dim=0)

    return sequences



def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description='Sample sequences from a trained LentIMPRA diffusion model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of sequences to generate')
    parser.add_argument('--architecture', type=str, default='transformer',
                        choices=['transformer', 'convolutional', 'transformer_multi_class'], help='Model architecture')
    parser.add_argument('--steps', type=int, default=230, help='Number of sampling steps')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for sampling')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--output', type=str, help='Optional output HDF5 file to save sequences (.h5)')
    parser.add_argument('--onehot', action='store_true', help='Also save one-hot encoded sequences in the HDF5 file')

    args = parser.parse_args()

    # Set device (string)
    device_str = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available, using CPU")

    print(f"Sampling {args.num_samples} sequences from {args.checkpoint}")
    print(f"Device: {device_str}")

    # Generate sequences
    sequences = sample_lentimpra_sequences(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_samples=args.num_samples,
        architecture=args.architecture,
        sampling_steps=args.steps,
        device=device_str,
        batch_size=args.batch_size
    )

    print(f"Generated sequences shape: {sequences.shape}")
    print(f"Sample sequence (first 50 positions): {sequences[0, :50]}")

    # Save if requested
    if args.output:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Write HDF5 file
        with h5py.File(args.output, 'w') as f:
            # Always save indexed sequences
            f.create_dataset('sequences', data=sequences.cpu().numpy(), compression='gzip', compression_opts=4)
            if args.onehot:
                # Convert to one-hot (N, L, 4) and save
                onehot = torch.nn.functional.one_hot(sequences.long(), num_classes=4).to(torch.uint8)
                f.create_dataset('sequences_onehot', data=onehot.cpu().numpy(), compression='gzip', compression_opts=4)
                # Save vocab mapping as ASCII strings
                dt = h5py.string_dtype(encoding='ascii', length=1)
                f.create_dataset('vocab_mapping', data=np.array(['A','C','G','T'], dtype=dt))
        print(f"Saved HDF5 sequences to {args.output}")


if __name__ == '__main__':
    main()
