#!/usr/bin/env python3
"""
cCRE Sampling Script

Inherits from base sampling framework while using cCRE-specific models directly.
Uses proper PC sampling methodology for unconditional generation.
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

# Import base framework and cCRE-specific components
from scripts.sample import BaseSampler, parse_base_args, main_sample
from model_zoo.ccre.data import get_ccre_datasets


class cCRESampler(BaseSampler):
    """cCRE-specific sampler that inherits from base framework."""
    
    def __init__(self):
        super().__init__("cCRE")
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """Load cCRE model using dataset-specific model loading."""
        from model_zoo.ccre.models import load_trained_model
        
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        """Get cCRE sequence length."""
        return 512  # cCRE fixed sequence length
    
    def generate_conditioning_labels(self, num_samples: int, config: OmegaConf) -> Optional[torch.Tensor]:
        """Generate conditioning labels for cCRE sampling.
        
        For cCRE, we return None since this is unconditional generation
        (no labels available).
        """
        return None


def load_default_config():
    """Load cCRE default configuration (transformer)."""
    config_file = Path(__file__).parent / 'configs' / 'transformer.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main sampling function using base framework."""
    # Parse arguments using base framework
    parser = parse_base_args()
    # Add cCRE-specific arguments
    parser.add_argument('--unconditional', action='store_true', default=True,
                       help='Sample unconditionally (default for cCRE since no labels)')
    parser.add_argument('--save_elements', type=str, nargs='+', default=None,
                       choices=['sequence', 'score', 'stag_score', 'prob'],
                       help='List of elements to save during sampling: sequence, score, stag_score, prob. '
                            'Each will be saved as (N, L, T, 4) tensor in HDF5 format.')
    args = parser.parse_args()

    # Load config using shared utility
    config, _ = BaseSampler.load_config_with_fallback(
        args.config, Path(__file__).parent, 'transformer.yaml'
    )
    sampler = cCRESampler()

    # For cCRE, we always do unconditional sampling (no labels available)
    print("Sampling unconditionally (cCRE has no conditioning labels)")

    # Set default steps to sequence length if not provided
    steps = args.steps if args.steps is not None else sampler.get_sequence_length(config)
    print(f"Using {steps} sampling steps")

    # Sample using PC sampler with optional element saving
    print(f"Loading cCRE {args.architecture} model from {args.checkpoint}")
    result = sampler.sample_sequences_with_pc_sampler(
        checkpoint_path=args.checkpoint,
        config=config,
        num_samples=args.num_samples,
        steps=steps,
        architecture=args.architecture,
        conditioning_labels=None,  # Always None for cCRE
        save_elements_list=args.save_elements
    )

    # Handle result using shared utility
    sequences, saved_elements, results = sampler.handle_sample_result(
        result, args.output, args.format, args.sequence_encoding
    )

    # Save elements if requested using shared utility
    if saved_elements:
        elements_file = sampler.save_sampling_elements(
            saved_elements, args.output, 'ccre_samples'
        )
        results['saved_elements_file'] = elements_file
        results['saved_elements'] = list(saved_elements.keys())

    # Print results
    print(f"\ncCRE Sampling Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")

    print(f"\n✓ cCRE sampling completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())