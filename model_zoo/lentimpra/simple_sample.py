#!/usr/bin/env python3
"""
Minimal function to sample sequences with a trained LentIMPRA model.

Based on model_zoo/lentimpra/sp_mse_callback.py
"""

import argparse
import torch
from omegaconf import OmegaConf
from typing import Optional
from model_zoo.lentimpra.models import load_trained_model
from scripts import sampling


def sample_lentimpra_sequences(
    checkpoint_path: str,
    config_path: str,
    num_samples: int,
    labels: Optional[torch.Tensor] = None,
    architecture: str = 'transformer',
    sampling_steps: int = 230,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Sample sequences from a trained LentIMPRA model.

    Args:
        checkpoint_path: Path to trained model checkpoint
        config_path: Path to model config file
        num_samples: Number of sequences to generate
        labels: Optional conditioning labels (num_samples, 1). If None, uses random labels.
        architecture: Model architecture ('transformer' or 'convolutional')
        sampling_steps: Number of diffusion steps (default: 230, sequence length)
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Generated sequences as indices (num_samples, 230)

    Example:
        >>> config_path = 'model_zoo/lentimpra/configs/transformer.yaml'
        >>> checkpoint_path = 'path/to/checkpoint.ckpt'
        >>> sequences = sample_lentimpra_sequences(checkpoint_path, config_path, num_samples=100)
        >>> print(sequences.shape)  # (100, 230)
    """
    # Load config
    config = OmegaConf.load(config_path)

    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load trained model
    model, graph, noise = load_trained_model(
        checkpoint_path,
        config,
        architecture,
        device=str(device)
    )
    model.eval()

    # Generate or validate labels
    if labels is None:
        # Random regulatory activity values
        labels = torch.randn(num_samples, 1, device=device)
    else:
        if labels.shape[0] != num_samples:
            raise ValueError(f"labels.shape[0] ({labels.shape[0]}) must match num_samples ({num_samples})")
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        labels = labels.to(device)

    # Create sampling function
    seq_length = 230  # LentIMPRA sequence length
    sampling_fn = sampling.get_pc_sampler(
        graph,
        noise,
        (num_samples, seq_length),
        'analytic',  # Predictor type
        sampling_steps,
        device=device
    )

    # Generate sequences
    with torch.no_grad():
        sequences = sampling_fn(model, labels)

    return sequences


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description='Sample sequences from a trained LentIMPRA diffusion model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of sequences to generate')
    parser.add_argument('--architecture', type=str, default='transformer',
                        choices=['transformer', 'convolutional'], help='Model architecture')
    parser.add_argument('--steps', type=int, default=230, help='Number of sampling steps')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--output', type=str, help='Optional output file to save sequences (.pt)')

    args = parser.parse_args()

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available, using CPU")

    print(f"Sampling {args.num_samples} sequences from {args.checkpoint}")
    print(f"Device: {device}")

    # Generate sequences
    sequences = sample_lentimpra_sequences(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_samples=args.num_samples,
        architecture=args.architecture,
        sampling_steps=args.steps,
        device=device
    )

    print(f"Generated sequences shape: {sequences.shape}")
    print(f"Sample sequence (first 50 positions): {sequences[0, :50]}")

    # Save if requested
    if args.output:
        torch.save(sequences, args.output)
        print(f"Saved sequences to {args.output}")


if __name__ == '__main__':
    main()
