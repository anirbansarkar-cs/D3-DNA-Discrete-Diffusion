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
    parser.add_argument('--save_elements', type=str, nargs='+', default=None,
                       choices=['sequence', 'score', 'stag_score', 'prob'],
                       help='List of elements to save during sampling: sequence, score, stag_score, prob. '
                            'Each will be saved as (N, L, T, 4) tensor in HDF5 format.')
    args = parser.parse_args()

    # Load config using shared utility
    config, _ = BaseSampler.load_config_with_fallback(
        args.config, Path(__file__).parent, 'transformer.yaml'
    )
    sampler = MPRASampler()

    # Generate conditioning labels based on arguments
    conditioning_labels = None
    if not args.unconditional:
        if args.activity1 is not None and args.activity2 is not None and args.activity3 is not None:
            # All three activities must be specified together
            conditioning_labels = torch.tensor(
                [[args.activity1, args.activity2, args.activity3]],
                device=sampler.device
            ).expand(args.num_samples, -1)
            print(f"Using specified activities: Activity1={args.activity1}, Activity2={args.activity2}, Activity3={args.activity3}")
        elif any([args.activity1, args.activity2, args.activity3]):
            # Partial specification is an error
            print("Error: For MPRA, either specify all three activities or none")
            return 1
        else:
            # Random activities (default behavior)
            conditioning_labels = sampler.generate_conditioning_labels(args.num_samples, config)
            print("Using random activities")
    else:
        print("Sampling unconditionally (no conditioning labels)")

    # Set default steps to sequence length if not provided
    steps = args.steps if args.steps is not None else sampler.get_sequence_length(config)
    print(f"Using {steps} sampling steps")

    # Sample using PC sampler with optional element saving
    print(f"Loading MPRA {args.architecture} model from {args.checkpoint}")
    result = sampler.sample_sequences_with_pc_sampler(
        checkpoint_path=args.checkpoint,
        config=config,
        num_samples=args.num_samples,
        steps=steps,
        architecture=args.architecture,
        conditioning_labels=conditioning_labels,
        save_elements_list=args.save_elements
    )

    # Handle result using shared utility
    sequences, saved_elements, results = sampler.handle_sample_result(
        result, args.output, args.format, args.sequence_encoding
    )

    # Save elements if requested using shared utility
    if saved_elements:
        elements_file = sampler.save_sampling_elements(
            saved_elements, args.output, 'mpra_samples'
        )
        results['saved_elements_file'] = elements_file
        results['saved_elements'] = list(saved_elements.keys())

    # Print results
    print(f"\nMPRA Sampling Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")

    print(f"\n✓ MPRA sampling completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())