#!/usr/bin/env python3
"""
Promoter Sampling Script. Inherits from base sampling framework while using Promoter-specific models directly.
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
        from model_zoo.promoter.models import load_trained_model

        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        if hasattr(config, 'model') and hasattr(config.model, 'length'):
            return config.model.length
        return 1024  # Promoter default sequence length
    
    def generate_conditioning_labels(self, num_samples: int, config: OmegaConf) -> torch.Tensor:
        seq_length = self.get_sequence_length(config)

        # Get signal_dim from dataset config (dimensionality of regulatory signal per position)
        if hasattr(config, 'dataset') and hasattr(config.dataset, 'signal_dim'):
            signal_dim = config.dataset.signal_dim
        else:
            signal_dim = 1  # Default for promoter

        # Check if global conditioning is requested (single value for entire sequence)
        use_global = getattr(config.model, 'use_global_conditioning', False) if hasattr(config, 'model') else False

        if use_global:
            # Global conditioning: single regulatory value for entire sequence
            labels = torch.randn(num_samples, signal_dim, device=self.device) * 2.0
        else:
            # Per-position conditioning: regulatory value at each position
            labels = torch.randn(num_samples, seq_length, signal_dim, device=self.device) * 2.0

        return labels


def load_config(architecture: str):
    config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():

    parser = parse_base_args()
    # Add Promoter-specific conditioning arguments
    parser.add_argument('--expression_target', type=float, help='Expression target value (if not provided, uses random)')
    parser.add_argument('--unconditional', action='store_true', help='Sample unconditionally (ignoring any labels)')
    args = parser.parse_args()

    config, _ = BaseSampler.load_config_with_fallback(
        args.config, Path(__file__).parent, 'transformer.yaml'
    )
    sampler = PromoterSampler()

    seq_length = sampler.get_sequence_length(config)

    steps = args.steps if args.steps is not None else seq_length

    conditioning_labels = None
    num_samples = args.num_samples

    if not args.unconditional:
        if args.use_test_set:
            if not args.data_path:
                print("Error: --data_path is required when using --use_test_set")
                return 1

            from model_zoo.promoter.data import PromoterDataset
            test_dataset = PromoterDataset(args.data_path, split='test')
            conditioning_labels = test_dataset.y.to(sampler.device)
            num_samples = len(test_dataset)
            print(f"Using test set labels: {num_samples} samples with shape {conditioning_labels.shape}")

        elif args.expression_target is not None:
            conditioning_labels = torch.full((num_samples, seq_length, 1), args.expression_target, device=sampler.device)
            print(f"Using specified expression target: {args.expression_target}")
        else:
            conditioning_labels = sampler.generate_conditioning_labels(num_samples, config)
            print(f"Using random expression targets with shape {conditioning_labels.shape}")
    else:
        print("Sampling unconditionally (no conditioning labels)")

    # Setup wandb if enabled
    if args.use_wandb:
        sampler.setup_wandb(args, config)

    print(f"Loading Promoter {args.architecture} model from {args.checkpoint}")
    result = sampler.sample_sequences_with_pc_sampler(
        checkpoint_path=args.checkpoint,
        config=config,
        num_samples=num_samples,
        steps=steps,
        architecture=args.architecture,
        conditioning_labels=conditioning_labels,
        save_elements_list=args.save_elements
    )

    sequences, saved_elements, results = sampler.handle_sample_result(
        result, args.output, args.format, args.sequence_encoding
    )

    results.update(sampler.handle_saved_elements(saved_elements, args.output, 'promoter_samples'))

    # Log to wandb if enabled
    if sampler.wandb_enabled:
        try:
            sampler.log_to_wandb(
                sequences=sequences,
                activity_labels=conditioning_labels,
                saved_elements=saved_elements
            )
        except Exception as e:
            print(f"Warning: Error logging to wandb: {e}")
        finally:
            sampler.cleanup_wandb()

    print(f"\nPromoter Sampling Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")

    print(f"\n✓ Promoter sampling completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())