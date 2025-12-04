#!/usr/bin/env python3
"""
LentIMPRA Sampling Script. Inherits from base sampling framework while using LentIMPRA-specific models directly.
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

# Import base framework and LentIMPRA-specific components
from scripts.sample import BaseSampler, parse_base_args, main_sample
from model_zoo.lentimpra.data import get_lentimpra_datasets


class LentIMPRASampler(BaseSampler):
    """LentIMPRA-specific sampler that inherits from base framework."""
    
    def __init__(self):
        super().__init__("LentIMPRA")
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        from model_zoo.lentimpra.models import load_trained_model
        
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        return 230  # LentIMPRA fixed sequence length
    
    def generate_conditioning_labels(self, num_samples: int, config: OmegaConf) -> torch.Tensor:
        # Check signal dimension from config (1 for single-class, 3 for multi-class)
        signal_dim = config.dataset.get('signal_dim', 1)

        # Generate random activities in a reasonable range
        labels = torch.randn(num_samples, signal_dim, device=self.device)
        return labels


def load_default_config():
    config_file = Path(__file__).parent / 'configs' / 'transformer.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():

    parser = parse_base_args()
    # Add LentIMPRA-specific conditioning arguments
    parser.add_argument('--activity', type=float, help='Regulatory activity value for single-class models (if not provided, uses random)')
    parser.add_argument('--k562_activity', type=float, help='K562 activity value for multi-class models')
    parser.add_argument('--hepg2_activity', type=float, help='HepG2 activity value for multi-class models')
    parser.add_argument('--wtc11_activity', type=float, help='WTC11 activity value for multi-class models')
    parser.add_argument('--unconditional', action='store_true', help='Sample unconditionally (ignoring any labels)')
    parser.add_argument('--use_test_set', action='store_true', default=False, help='Use test set labels from dataset as conditioning labels')
    parser.add_argument('--save_elements', type=str, nargs='+', default=None,
                       choices=['sequence', 'score', 'stag_score', 'prob'],
                       help='List of elements to save during sampling: sequence, score, stag_score, prob. '
                            'Each will be saved as (N, L, T, 4) tensor in HDF5 format.')
    parser.add_argument('--initial_condition', type=str, default='random',
                       choices=['random', 'test', 'dinuc'],
                       help='Initial condition for sampling: random (default), test (use onehot_test sequences), '
                            'or dinuc (use pre-computed onehot_test_dinuc sequences from H5 file). '
                            'Requires --data_path when using test or dinuc.')
    args = parser.parse_args()

    # Load config using shared utility
    config, _ = BaseSampler.load_config_with_fallback(
        args.config, Path(__file__).parent, 'transformer.yaml'
    )
    sampler = LentIMPRASampler()

    # Determine signal dimension from config (1 for single-class, 3 for multi-class)
    signal_dim = config.dataset.get('signal_dim', 1)

    # Handle initial condition loading first to determine num_samples
    initial_x = None
    num_samples = args.num_samples

    if args.initial_condition != 'random':
        if not args.data_path:
            print("Error: --data_path is required when using --initial_condition test or dinuc")
            return 1

        print(f"Loading initial conditions from test set ({args.initial_condition} mode)...")

        # Load one-hot sequences directly from H5 file
        with h5py.File(args.data_path, 'r') as data:
            if args.initial_condition == 'dinuc':
                onehot_test = np.array(data['onehot_test_dinuc'])  # (N, 230, 4)
                print(f"Loaded dinucleotide-shuffled test sequences")
            else:  # args.initial_condition == 'test'
                onehot_test = np.array(data['onehot_test'])  # (N, 230, 4)
                print(f"Loaded test sequences")

        num_samples = len(onehot_test)
        print(f"Loaded {num_samples} sequences with shape {onehot_test.shape}")

        # Convert one-hot to indices: (N, 230, 4) -> (N, 4, 230) -> (N, 230)
        onehot_test = np.transpose(onehot_test, (0, 2, 1))  # (N, 230, 4) -> (N, 4, 230)
        initial_x = torch.tensor(np.argmax(onehot_test, axis=1))  # (N, 4, 230) -> (N, 230)
        print(f"Initial conditions prepared: {initial_x.shape}")

    # Generate conditioning labels based on arguments (now with correct num_samples)
    conditioning_labels = None

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
    
    steps = args.steps
    if steps is None:
        steps = sampler.get_sequence_length(config)

    architecture = args.architecture
    if signal_dim > 1 and architecture == 'transformer':
        architecture = 'transformer_multi_class'
        print(f"Auto-detected multi-class model (signal_dim={signal_dim}), using architecture: {architecture}")

    # Run sampling using PC sampler
    print(f"Loading LentIMPRA {architecture} model from {args.checkpoint}")
    result = sampler.sample_sequences_with_pc_sampler(
        checkpoint_path=args.checkpoint,
        config=config,
        num_samples=num_samples,
        steps=steps,
        architecture=architecture,
        conditioning_labels=conditioning_labels,
        save_elements_list=args.save_elements,
        initial_x=initial_x
    )

    # Handle result using shared utility
    sequences, saved_elements, results = sampler.handle_sample_result(
        result, args.output, args.format, args.sequence_encoding
    )

    # Save elements if requested using shared utility
    if saved_elements:
        elements_file = sampler.save_sampling_elements(
            saved_elements, args.output, 'lentimpra_samples'
        )
        results['saved_elements_file'] = elements_file
        results['saved_elements'] = list(saved_elements.keys())

    # Print results
    print(f"\nLentiMPRA sampling complete. Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")
    return 0


if __name__ == '__main__':
    sys.exit(main())