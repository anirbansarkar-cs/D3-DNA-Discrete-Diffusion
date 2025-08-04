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
        """Generate conditioning labels for LentIMPRA sampling."""
        # LentIMPRA has single regression target for regulatory activity
        # Generate random activities in a reasonable range
        labels = torch.randn(num_samples, 1, device=self.device)
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
    parser.add_argument('--activity', type=float, help='Regulatory activity value (if not provided, uses random)')
    parser.add_argument('--unconditional', action='store_true', help='Sample unconditionally (ignoring any labels)')
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
    
    # Generate conditioning labels based on arguments
    conditioning_labels = None
    if not args.unconditional:
        if args.activity is not None:
            # User-specified activity
            conditioning_labels = torch.tensor([[args.activity]], device=sampler.device).expand(args.num_samples, -1)
            print(f"Using specified activity: {args.activity}")
        else:
            # Random activity (default behavior)
            conditioning_labels = sampler.generate_conditioning_labels(args.num_samples, config)
            print("Using random activities")
    else:
        print("Sampling unconditionally (no conditioning labels)")
    
    # Set default steps to sequence length if not provided
    steps = args.steps
    if steps is None:
        steps = sampler.get_sequence_length(config)
        print(f"Using default steps: {steps} (sequence length)")
    
    # Run sampling only (no evaluation)
    results = sampler.sample_and_save(
        checkpoint_path=args.checkpoint,
        config=config,
        num_samples=args.num_samples,
        steps=steps,
        architecture=args.architecture,
        conditioning_labels=conditioning_labels,
        output_path=args.output,
        format=args.format
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