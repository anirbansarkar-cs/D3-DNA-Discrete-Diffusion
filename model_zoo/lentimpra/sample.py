#!/usr/bin/env python3
"""
LentIMPRA Sampling Script

Inherits from base sampling framework while using LentIMPRA-specific models directly.
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

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import base framework and LentIMPRA-specific components
from scripts.sample import BaseSampler, parse_base_args, main_sample
from model_zoo.lentimpra.data import get_lentimpra_datasets


class LentIMPRASampler(BaseSampler):
    """LentIMPRA-specific sampler that inherits from base framework."""
    
    def __init__(self):
        super().__init__("LentIMPRA")
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """Load LentIMPRA model using dataset-specific model loading."""
        from model_zoo.lentimpra.models import load_trained_model
        
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        """Get LentIMPRA sequence length."""
        return 230  # LentIMPRA fixed sequence length
    
    def generate_conditioning_labels(self, num_samples: int, config: OmegaConf) -> torch.Tensor:
        """Generate conditioning labels for LentIMPRA sampling.

        Supports both single-class (N, 1) and multi-class (N, 3) based on config.
        """
        # Check signal dimension from config (1 for single-class, 3 for multi-class)
        signal_dim = config.dataset.get('signal_dim', 1)

        # Generate random activities in a reasonable range
        labels = torch.randn(num_samples, signal_dim, device=self.device)
        return labels


def load_default_config():
    """Load LentIMPRA default configuration (transformer)."""
    config_file = Path(__file__).parent / 'configs' / 'transformer.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main sampling function using base framework."""
    # Parse arguments using base framework
    parser = parse_base_args()
    # Add LentIMPRA-specific conditioning arguments
    parser.add_argument('--activity', type=float, help='Regulatory activity value for single-class models (if not provided, uses random)')
    parser.add_argument('--k562_activity', type=float, help='K562 activity value for multi-class models')
    parser.add_argument('--hepg2_activity', type=float, help='HepG2 activity value for multi-class models')
    parser.add_argument('--wtc11_activity', type=float, help='WTC11 activity value for multi-class models')
    parser.add_argument('--unconditional', action='store_true', help='Sample unconditionally (ignoring any labels)')
    parser.add_argument('--use_test_set', action='store_true', default=False, help='Use test set labels from dataset as conditioning labels')
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
    sampler = LentIMPRASampler()

    # Determine signal dimension from config (1 for single-class, 3 for multi-class)
    signal_dim = config.dataset.get('signal_dim', 1)

    # Generate conditioning labels based on arguments
    conditioning_labels = None
    num_samples = args.num_samples

    if not args.unconditional:
        if args.use_test_set:
            # Use test set labels from dataset
            if not args.data_path:
                print("Error: --data_path is required when using --use_test_set")
                return 1

            # Load test dataset to get labels
            from model_zoo.lentimpra.data import LentIMPRADataset
            test_dataset = LentIMPRADataset(args.data_path, split='test')
            conditioning_labels = test_dataset.y.to(sampler.device)
            num_samples = len(test_dataset)
            print(f"Using test set labels: {num_samples} samples with shape {conditioning_labels.shape}")

        elif signal_dim == 1 and args.activity is not None:
            # Single-class: user-specified activity
            conditioning_labels = torch.tensor([[args.activity]], device=sampler.device).expand(num_samples, -1)
            print(f"Using specified activity: {args.activity}")

        elif signal_dim == 3:
            # Multi-class: check if all three activities are specified
            multi_class_activities = [args.k562_activity, args.hepg2_activity, args.wtc11_activity]
            if all(a is not None for a in multi_class_activities):
                # All three activities specified
                conditioning_labels = torch.tensor(
                    [[args.k562_activity, args.hepg2_activity, args.wtc11_activity]],
                    device=sampler.device
                ).expand(num_samples, -1)
                print(f"Using specified activities - K562: {args.k562_activity}, HepG2: {args.hepg2_activity}, WTC11: {args.wtc11_activity}")
            elif any(a is not None for a in multi_class_activities):
                # Some but not all activities specified - this is an error
                print("Error: For multi-class models, either specify all three activities (--k562_activity, --hepg2_activity, --wtc11_activity) or none")
                return 1
            else:
                # No activities specified, use random
                conditioning_labels = sampler.generate_conditioning_labels(num_samples, config)
                print(f"Using random activities (multi-class, {signal_dim} dimensions)")
        else:
            # Random activity (default behavior)
            conditioning_labels = sampler.generate_conditioning_labels(num_samples, config)
            print(f"Using random activities ({signal_dim} dimension{'s' if signal_dim > 1 else ''})")
    else:
        print("Sampling unconditionally (no conditioning labels)")
    
    # Set default steps to sequence length if not provided
    steps = args.steps
    if steps is None:
        steps = sampler.get_sequence_length(config)
        print(f"Using default steps: {steps} (sequence length)")

    # Auto-detect architecture for multi-class models
    architecture = args.architecture
    if signal_dim > 1 and architecture == 'transformer':
        architecture = 'transformer_multi_class'
        print(f"Auto-detected multi-class model (signal_dim={signal_dim}), using architecture: {architecture}")

    # Run sampling only (no evaluation)
    results = sampler.sample_and_save(
        checkpoint_path=args.checkpoint,
        config=config,
        num_samples=num_samples,
        steps=steps,
        architecture=architecture,
        conditioning_labels=conditioning_labels,
        output_path=args.output,
        format=args.format,
        encoding=args.sequence_encoding
    )
    
    # Print results
    print(f"\nLentIMPRA Sampling Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")
    
    print(f"\n✓ LentIMPRA sampling completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())