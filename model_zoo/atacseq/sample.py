#!/usr/bin/env python3
"""
ATAC-seq Sampling Script. Inherits from base sampling framework while using ATAC-seq-specific models directly.
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

# Import base framework and ATAC-seq-specific components
from scripts.sample import BaseSampler, parse_base_args, main_sample


class ATACseqSampler(BaseSampler):
    """ATAC-seq-specific sampler that inherits from base framework."""

    def __init__(self):
        super().__init__("ATACseq")

    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        from model_zoo.atacseq.models import load_trained_model

        return load_trained_model(checkpoint_path, config, architecture, self.device)

    def get_sequence_length(self, config: OmegaConf) -> int:
        if hasattr(config, 'model') and hasattr(config.model, 'length'):
            return config.model.length
        return 249  # ATAC-seq fixed sequence length

    def generate_conditioning_labels(self, num_samples: int, config: OmegaConf) -> torch.Tensor:
        # Get signal_dim from config (number of cell types)
        if hasattr(config, 'dataset') and hasattr(config.dataset, 'signal_dim'):
            signal_dim = config.dataset.signal_dim
        else:
            signal_dim = 18  # Default for ATAC-seq

        labels = torch.randn(num_samples, signal_dim, device=self.device) * 2.0
        return labels


def main():

    parser = parse_base_args()
    # Add ATAC-seq-specific conditioning arguments
    parser.add_argument('--unconditional', action='store_true', help='Sample unconditionally (ignoring any labels)')
    parser.add_argument('--save_elements', type=str, nargs='+', default=None,
                       choices=['sequence', 'score', 'stag_score', 'prob'],
                       help='List of elements to save during sampling: sequence, score, stag_score, prob. '
                            'Each will be saved as (N, L, T, 4) tensor in HDF5 format.')
    args = parser.parse_args()

    config, _ = BaseSampler.load_config_with_fallback(
        args.config, Path(__file__).parent, 'transformer.yaml'
    )
    sampler = ATACseqSampler()

    conditioning_labels = None
    if not args.unconditional:
        conditioning_labels = sampler.generate_conditioning_labels(args.num_samples, config)
        print(f"Using random cell type activities with shape {conditioning_labels.shape}")
    else:
        print("Sampling unconditionally (no conditioning labels)")

    steps = args.steps if args.steps is not None else sampler.get_sequence_length(config)

    print(f"Loading ATAC-seq {args.architecture} model from {args.checkpoint}")
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

    if saved_elements:
        elements_file = sampler.save_sampling_elements(
            saved_elements, args.output, 'atacseq_samples'
        )
        results['saved_elements_file'] = elements_file
        results['saved_elements'] = list(saved_elements.keys())

    print(f"\nATAC-seq Sampling Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")

    print(f"\n✓ ATAC-seq sampling completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
