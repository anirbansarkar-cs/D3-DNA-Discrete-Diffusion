#!/usr/bin/env python3
"""
Promoter Sampling Script

Inherits from base sampling framework while using Promoter-specific models directly.
Uses proper PC sampling methodology.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from typing import Optional
import numpy as np
import h5py

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import base framework and Promoter-specific components
from scripts.sample import BaseSampler, parse_base_args, main_sample
from model_zoo.promoter.data import get_promoter_datasets


class PromoterSampler(BaseSampler):
    """Promoter-specific sampler that inherits from base framework."""
    
    def __init__(self):
        super().__init__("Promoter")
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """Load Promoter model using dataset-specific model loading."""
        from model_zoo.promoter.models import load_trained_model
        
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        """Get Promoter sequence length."""
        if hasattr(config, 'model') and hasattr(config.model, 'length'):
            return config.model.length
        return 1024  # Promoter default sequence length
    
    def generate_conditioning_labels(self, num_samples: int, config: OmegaConf) -> torch.Tensor:
        """
        Generate conditioning labels for Promoter sampling.

        Promoter uses per-position regulatory activity labels by default,
        where each position has a signal_dim-dimensional regulatory signal.
        For promoter: signal_dim=1 and per-position conditioning is used.

        Args:
            num_samples: Number of samples to generate labels for
            config: Configuration object containing dataset and model parameters

        Returns:
            Conditioning labels tensor
        """
        seq_length = self.get_sequence_length(config)

        # Get signal_dim from dataset config (dimensionality of regulatory signal per position)
        if hasattr(config, 'dataset') and hasattr(config.dataset, 'signal_dim'):
            signal_dim = config.dataset.signal_dim
        else:
            signal_dim = 1  # Default for promoter

        # Check if global conditioning is requested (single value for entire sequence)
        # Otherwise, use per-position conditioning (default for promoter)
        use_global = getattr(config.model, 'use_global_conditioning', False) if hasattr(config, 'model') else False

        if use_global:
            # Global conditioning: single regulatory value for entire sequence
            # TODO: should include a check for the architecture (should have been trained with same shape of labels)
            # Shape: (num_samples, signal_dim)
            labels = torch.randn(num_samples, signal_dim, device=self.device) * 2.0
        else:
            # Per-position conditioning: regulatory value at each position
            # Shape: (num_samples, seq_length, signal_dim)
            labels = torch.randn(num_samples, seq_length, signal_dim, device=self.device) * 2.0

        return labels


def load_config(architecture: str):
    """Load Promoter configuration."""
    config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main sampling function using base framework."""
    # Parse arguments using base framework
    parser = parse_base_args()
    # Add Promoter-specific conditioning arguments
    parser.add_argument('--expression_target', type=float, help='Expression target value (if not provided, uses random)')
    parser.add_argument('--unconditional', action='store_true', help='Sample unconditionally (ignoring any labels)')
    parser.add_argument('--use_test_set', action='store_true', default=False, help='Use test set labels from dataset as conditioning labels')
    parser.add_argument('--save_elements', type=str, nargs='+', default=None,
                       choices=['sequence', 'score', 'stag_score', 'prob'],
                       help='List of elements to save during sampling: sequence, score, stag_score, prob. '
                            'Each will be saved as (N, L, T, 4) tensor in HDF5 format.')
    args = parser.parse_args()
    
    # Load config if not provided
    if not args.config:
        try:
            config_path = Path(__file__).parent / 'configs' / 'transformer.yaml'  # Default to transformer
            if config_path.exists():
                args.config = str(config_path)
                print(f"Using default config: {args.config}")
            else:
                print(f"Error: No config provided and default config not found: {config_path}")
                print("Please provide a config file with --config")
                return 1
        except Exception as e:
            print(f"Error loading default config: {e}")
            return 1
    
    config = OmegaConf.load(args.config)
    sampler = PromoterSampler()

    # Get sequence length
    seq_length = sampler.get_sequence_length(config)

    # Set default steps to sequence length if not provided
    steps = args.steps if args.steps is not None else seq_length
    print(f"Using {steps} sampling steps")

    # Generate conditioning labels for all samples
    conditioning_labels = None
    num_samples = args.num_samples

    if not args.unconditional:
        if args.use_test_set:
            # Use test set labels from dataset
            if not args.data_path:
                print("Error: --data_path is required when using --use_test_set")
                return 1

            # Load test dataset to get labels
            from model_zoo.promoter.data import PromoterDataset
            test_dataset = PromoterDataset(args.data_path, split='test')
            conditioning_labels = test_dataset.y.to(sampler.device)  # Shape: (N, 1024, 1)
            num_samples = len(test_dataset)
            print(f"Using test set labels: {num_samples} samples with shape {conditioning_labels.shape}")

        elif args.expression_target is not None:
            # User-specified expression target - replicate across all positions
            # Shape: (num_samples, seq_length, 1) for per-position conditioning
            conditioning_labels = torch.full((num_samples, seq_length, 1), args.expression_target, device=sampler.device)
            print(f"Using specified expression target: {args.expression_target}")
        else:
            # Random expression targets (default behavior)
            conditioning_labels = sampler.generate_conditioning_labels(num_samples, config)
            print(f"Using random expression targets with shape {conditioning_labels.shape}")
    else:
        print("Sampling unconditionally (no conditioning labels)")

    # Use base class batched sampling (handles flash attention memory issues automatically)
    # Smart defaults: automatically batches with size 256 when num_samples > 512
    print(f"Loading Promoter {args.architecture} model from {args.checkpoint}")
    result = sampler.sample_sequences_with_pc_sampler(
        checkpoint_path=args.checkpoint,
        config=config,
        num_samples=num_samples,
        steps=steps,
        architecture=args.architecture,
        conditioning_labels=conditioning_labels,
        save_elements_list=args.save_elements
        # No sampling_batch_size - uses automatic smart defaults from base class
    )
    
    # Handle returned result (may be just sequences or (sequences, saved_elements))
    if isinstance(result, tuple):
        sequences, saved_elements = result
    else:
        sequences = result
        saved_elements = None

    # Save sequences if output path provided
    if args.output:
        sampler.save_sequences(sequences, args.output, args.format, args.sequence_encoding)
        results = {
            'num_sequences': len(sequences),
            'sequence_length': seq_length,
            'output_file': args.output,
            'encoding': args.sequence_encoding
        }
    else:
        results = {
            'num_sequences': len(sequences),
            'sequence_length': seq_length
        }
    
    # Save elements if requested
    if saved_elements:
        # Determine output directory (use same directory as sequence output if provided)
        if args.output:
            output_dir = Path(args.output).parent
            base_name = Path(args.output).stem
        else:
            output_dir = Path('.')
            base_name = 'promoter_samples'
        
        # Save all elements as datasets in a single HDF5 file
        output_file = output_dir / f"{base_name}_elements.h5"
        
        print(f"\nSaving sampling elements to {output_file}...")
        with h5py.File(output_file, 'w') as f:
            for elem_name, elem_tensor in saved_elements.items():
                # elem_tensor shape: (N, L, T, 4)
                f.create_dataset(elem_name, data=elem_tensor.numpy(), compression='gzip')
                print(f"  Saved dataset '{elem_name}': shape {elem_tensor.shape}")
            
            # Save metadata as attributes
            first_elem = list(saved_elements.values())[0]
            f.attrs['num_samples'] = first_elem.shape[0]
            f.attrs['sequence_length'] = first_elem.shape[1]
            f.attrs['num_timesteps'] = first_elem.shape[2]
            f.attrs['num_classes'] = first_elem.shape[3]
            f.attrs['saved_elements'] = list(saved_elements.keys())
        
        print(f"  All elements saved to: {output_file}")
        results['saved_elements_file'] = str(output_file)
        results['saved_elements'] = list(saved_elements.keys())
    
    # Print results
    print(f"\nPromoter Sampling Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")
    
    print(f"\n✓ Promoter sampling completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())