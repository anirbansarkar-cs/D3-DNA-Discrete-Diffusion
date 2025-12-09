#!/usr/bin/env python3
"""
DeepSTARR Sampling Script. Inherits from base sampling framework while using DeepSTARR-specific models directly.
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

# Import base framework and DeepSTARR-specific components
from scripts.sample import BaseSampler, parse_base_args, main_sample
from model_zoo.deepstarr.data import get_deepstarr_datasets
from model_zoo.deepstarr.deepstarr import PL_DeepSTARR


class DeepSTARRSampler(BaseSampler):
    """DeepSTARR-specific sampler that inherits from base framework."""
    
    def __init__(self):
        super().__init__("DeepSTARR")
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        from model_zoo.deepstarr.models import load_trained_model

        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        return 249  # DeepSTARR fixed sequence length
    
    def generate_conditioning_labels(self, num_samples: int, config: OmegaConf) -> torch.Tensor:
        # DeepSTARR has 2 activities: Dev and HK enhancer activities
        labels = torch.randn(num_samples, 2, device=self.device)
        return labels
    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None):
        train_ds, val_ds, test_ds = get_deepstarr_datasets(config.paths.data_file)

        if split == 'train':
            dataset = train_ds
        elif split == 'val':
            dataset = val_ds
        elif split == 'test':
            dataset = test_ds
        else:
            raise ValueError(f"Unknown split: {split}")

        if batch_size is None:
            batch_size = getattr(config, 'batch_size', 32)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )


def load_default_config():
    config_file = Path(__file__).parent / 'configs' / 'transformer.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():

    parser = parse_base_args()
    # Add DeepSTARR-specific conditioning arguments
    parser.add_argument('--dev_activity', type=float, help='Dev enhancer activity value (if not provided, uses random)')
    parser.add_argument('--hk_activity', type=float, help='HK enhancer activity value (if not provided, uses random)')
    parser.add_argument('--unconditional', action='store_true', help='Sample unconditionally (ignoring any labels)')
    args = parser.parse_args()

    config, _ = BaseSampler.load_config_with_fallback(
        args.config, Path(__file__).parent, 'transformer.yaml'
    )
    sampler = DeepSTARRSampler()

    if args.save_rep:
        print(f"Saving representation of the model to {args.data_path}")
        results = sampler.save_representation(
            checkpoint_path=args.checkpoint,
            config=config,
            split=args.split,
            save_rep_timestamp=args.save_rep_timestamp,
            batch_size=args.batch_size,
            architecture=args.architecture,
            output_path=args.output,
            format=args.format
        )

        print(f"\n{sampler.dataset_name} Representation Results:")
        print("=" * 40)
        for key, value in results.items():
            print(f"{key}: {value}")

        print(f"\n✓ {sampler.dataset_name} saving representation completed successfully!")
        sys.exit(0)

    conditioning_labels = None
    if not args.unconditional:
        if args.dev_activity is not None and args.hk_activity is not None:
            conditioning_labels = torch.tensor([[args.dev_activity, args.hk_activity]], device=sampler.device).expand(args.num_samples, -1)
            print(f"Using specified activities: Dev={args.dev_activity}, HK={args.hk_activity}")
        else:
            conditioning_labels = sampler.generate_conditioning_labels(args.num_samples, config)
            print("Using random activities")
    else:
        print("Sampling unconditionally (no conditioning labels)")

    steps = args.steps
    if steps is None:
        steps = sampler.get_sequence_length(config)

    print(f"Loading DeepSTARR {args.architecture} model from {args.checkpoint}")
    result = sampler.sample_sequences_with_pc_sampler(
        checkpoint_path=args.checkpoint,
        config=config,
        num_samples=args.num_samples,
        steps=steps,
        architecture=args.architecture,
        conditioning_labels=conditioning_labels,
        save_elements_list=args.save_elements
    )

    sequences, saved_elements, results = sampler.handle_sample_result(
        result, args.output, args.format, args.sequence_encoding
    )

    results.update(sampler.handle_saved_elements(saved_elements, args.output, 'deepstarr_samples'))

    print(f"\nDeepSTARR Sampling Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")

    print(f"\n✓ DeepSTARR sampling completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())