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
    # LentIMPRA-specific custom initialization arguments
    parser.add_argument('--custom_inits_path', type=str, help='Path to a H5 file with sequences for custom initial conditions')
    parser.add_argument('--custom_inits_step', type=int, default=10, help='Step to use for custom initial conditions')
    args = parser.parse_args()

    # TODO: unify save_elements with the save_rep and other functions in the BaseSampler class

    # Load config using shared utility
    config, _ = BaseSampler.load_config_with_fallback(
        args.config, Path(__file__).parent, 'transformer.yaml'
    )
    sampler = LentIMPRASampler()

    # Determine signal dimension from config (1 for single-class, 3 for multi-class)
    signal_dim = config.dataset.get('signal_dim', 1)

    initial_x = None
    num_samples = args.num_samples

    if args.initial_condition != 'random':
        # TODO: either delete this custom loader later or make it more general
        if args.initial_condition == 'custom':
            if not args.custom_inits_path:
                print("Error: --custom_inits_path is required when using --initial_condition custom")
                return 1
            h5_path = args.custom_inits_path
            dataset_key = 'sequence'
        else:
            if not args.data_path:
                print(f"Error: --data_path is required when using --initial_condition {args.initial_condition}")
                return 1
            h5_path = args.data_path
            dataset_key = 'onehot_test_dinuc' if args.initial_condition == 'dinuc' else 'onehot_test'

        with h5py.File(h5_path, 'r') as data:
            onehot = np.array(data[dataset_key])

            # TODO: remove or generalize
            if args.initial_condition == 'custom':  # then the shape is (samples, 230, steps, 4), do step 10,25,40
                onehot = onehot[:, :, args.custom_inits_step, :]
        
        num_samples = len(onehot)
        # TODO: refine shape checking for standard expected input
        if onehot.shape[1] != 4:
            onehot = np.transpose(onehot, (0, 2, 1))
        initial_x = torch.tensor(np.argmax(onehot, axis=1))
        print(f"Loaded {num_samples} initial sequences: {initial_x.shape}")

    conditioning_labels = None
    if not args.unconditional:
        if args.use_test_set:
            if not args.data_path:
                print("Error: --data_path is required when using --use_test_set")
                return 1
            from model_zoo.lentimpra.data import LentIMPRADataset
            test_dataset = LentIMPRADataset(args.data_path, split='test')
            conditioning_labels = test_dataset.y.to(sampler.device)
            num_samples = len(test_dataset)

        elif signal_dim == 1 and args.activity is not None:
            conditioning_labels = torch.tensor([[args.activity]], device=sampler.device).expand(num_samples, -1)

        elif signal_dim == 3:
            activities = [args.k562_activity, args.hepg2_activity, args.wtc11_activity]
            if all(a is not None for a in activities):
                conditioning_labels = torch.tensor([activities], device=sampler.device).expand(num_samples, -1)
            elif any(a is not None for a in activities):
                print("Error: For multi-class models, specify all three activities or none")
                return 1
            else:
                conditioning_labels = sampler.generate_conditioning_labels(num_samples, config)

        else:
            conditioning_labels = sampler.generate_conditioning_labels(num_samples, config)
    
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
    results.update(sampler.handle_saved_elements(saved_elements, args.output, 'lentimpra_samples'))

    # Print results
    print(f"\nLentiMPRA sampling complete. Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")
    return 0


if __name__ == '__main__':
    sys.exit(main())