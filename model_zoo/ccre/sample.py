#!/usr/bin/env python3
"""
cCRE Sampling Script. Inherits from base sampling framework while using cCRE-specific models directly.
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
        from model_zoo.ccre.models import load_trained_model

        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        return 512  # cCRE fixed sequence length
    
    def generate_conditioning_labels(self, num_samples: int, config: OmegaConf) -> Optional[torch.Tensor]:
        return None  # cCRE is unconditional generation


def load_default_config():
    config_file = Path(__file__).parent / 'configs' / 'transformer.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():

    parser = parse_base_args()
    # Add cCRE-specific arguments
    parser.add_argument('--unconditional', action='store_true', default=True,
                       help='Sample unconditionally (default for cCRE since no labels)')
    args = parser.parse_args()

    config, _ = BaseSampler.load_config_with_fallback(
        args.config, Path(__file__).parent, 'transformer.yaml'
    )
    sampler = cCRESampler()

    print("Sampling unconditionally (cCRE has no conditioning labels)")

    # Setup wandb if enabled
    if args.use_wandb:
        sampler.setup_wandb(args, config)

    steps = args.steps if args.steps is not None else sampler.get_sequence_length(config)

    print(f"Loading cCRE {args.architecture} model from {args.checkpoint}")
    result = sampler.sample_sequences_with_pc_sampler(
        checkpoint_path=args.checkpoint,
        config=config,
        num_samples=args.num_samples,
        steps=steps,
        architecture=args.architecture,
        conditioning_labels=None,
        save_elements_list=args.save_elements
    )

    sequences, saved_elements, results = sampler.handle_sample_result(
        result, args.output, args.format, args.sequence_encoding
    )

    results.update(sampler.handle_saved_elements(saved_elements, args.output, 'ccre_samples'))

    # Log to wandb if enabled
    if sampler.wandb_enabled:
        try:
            sampler.log_to_wandb(
                sequences=sequences,
                activity_labels=None,
                saved_elements=saved_elements
            )
        except Exception as e:
            print(f"Warning: Error logging to wandb: {e}")
        finally:
            sampler.cleanup_wandb()

    print(f"\ncCRE Sampling Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")

    print(f"\n✓ cCRE sampling completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())