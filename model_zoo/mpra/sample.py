#!/usr/bin/env python3
"""
MPRA Sampling Script

Inherits from base sampling framework while using MPRA-specific models directly.
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

# Import base framework and MPRA-specific components
from scripts.sample import BaseSampler, parse_base_args, main_sample
from model_zoo.mpra.data import get_mpra_datasets


class MPRASampler(BaseSampler):
    """MPRA-specific sampler that inherits from base framework."""
    
    def __init__(self):
        super().__init__("MPRA")
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """Load MPRA model using dataset-specific model loading."""
        from model_zoo.mpra.models import load_trained_model
        
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        """Get MPRA sequence length."""
        if hasattr(config, 'model') and hasattr(config.model, 'length'):
            return config.model.length
        return 200  # MPRA default sequence length
    
    def generate_conditioning_labels(self, num_samples: int, config: OmegaConf) -> torch.Tensor:
        """Generate conditioning labels for MPRA sampling."""
        # MPRA typically has 3 labels for different regulatory activities
        # Generate random activities in a reasonable range
        labels = torch.randn(num_samples, 3, device=self.device) * 1.5  # Adjust scale as needed
        return labels


def load_config(architecture: str):
    """Load MPRA configuration."""
    config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main sampling function using base framework."""
    # Parse arguments using base framework
    parser = parse_base_args()
    # Add MPRA-specific conditioning arguments
    parser.add_argument('--activity1', type=float, help='Activity 1 value (if not provided, uses random)')
    parser.add_argument('--activity2', type=float, help='Activity 2 value (if not provided, uses random)')
    parser.add_argument('--activity3', type=float, help='Activity 3 value (if not provided, uses random)')
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
    sampler = MPRASampler()
    
    # Generate conditioning labels based on arguments
    conditioning_labels = None
    if not args.unconditional:
        if args.activity1 is not None and args.activity2 is not None and args.activity3 is not None:
            # User-specified activities
            conditioning_labels = torch.tensor([[args.activity1, args.activity2, args.activity3]], device=sampler.device).expand(args.num_samples, -1)
            print(f"Using specified activities: Activity1={args.activity1}, Activity2={args.activity2}, Activity3={args.activity3}")
        else:
            # Random activities (default behavior)
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
        model_path=args.model_path,
        config=config,
        num_samples=args.num_samples,
        steps=steps,
        conditioning_labels=conditioning_labels,
        output_path=args.output,
        format=args.format
    )
    
    # Print results
    print(f"\nMPRA Sampling Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")
    
    print(f"\n✓ MPRA sampling completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())