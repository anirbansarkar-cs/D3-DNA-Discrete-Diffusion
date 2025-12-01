#!/usr/bin/env python3
"""
DeepSTARR Sampling Script

Inherits from base sampling framework while using DeepSTARR-specific models directly.
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
        """Load DeepSTARR model using dataset-specific model loading."""
        from model_zoo.deepstarr.models import load_trained_model
        
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        """Get DeepSTARR sequence length."""
        return 249  # DeepSTARR fixed sequence length
    
    def generate_conditioning_labels(self, num_samples: int, config: OmegaConf) -> torch.Tensor:
        """Generate conditioning labels for DeepSTARR sampling."""
        # DeepSTARR has 2 activities: Dev and HK enhancer activities
        # Generate random activities in a reasonable range
        labels = torch.randn(num_samples, 2, device=self.device)
        return labels
    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None):
        """Create DeepSTARR dataloader."""
        # Load datasets
        train_ds, val_ds, test_ds = get_deepstarr_datasets(config.paths.data_file)
        
        # Select appropriate dataset
        if split == 'train':
            dataset = train_ds
        elif split == 'val':  # Use val as test for now
            dataset = val_ds
        elif split == 'test':
            dataset = test_ds
        else:
            raise ValueError(f"Unknown split: {split}")
            
        # Use config batch size if not specified
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
    """Load DeepSTARR default configuration (transformer)."""
    config_file = Path(__file__).parent / 'configs' / 'transformer.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main sampling function using base framework."""
    # Parse arguments using base framework
    parser = parse_base_args()
    # Add DeepSTARR-specific conditioning arguments
    parser.add_argument('--dev_activity', type=float, help='Dev enhancer activity value (if not provided, uses random)')
    parser.add_argument('--hk_activity', type=float, help='HK enhancer activity value (if not provided, uses random)')
    parser.add_argument('--unconditional', action='store_true', help='Sample unconditionally (ignoring any labels)')
    parser.add_argument('--save_elements', type=str, nargs='+', default=None,
                       choices=['sequence', 'score', 'stag_score', 'prob'],
                       help='List of elements to save during sampling: sequence, score, stag_score, prob. '
                            'Each will be saved as (N, L, T, 4) tensor in HDF5 format.')
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
    sampler = DeepSTARRSampler()

    if args.save_rep:
        # if not args.data_path:
        #     print("Error: --data_path is required for saving representation")
        #     return 1
        # else:
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

        # Print results
        print(f"\n{sampler.dataset_name} Representation Results:")
        print("=" * 40)
        for key, value in results.items():
            print(f"{key}: {value}")
        
        print(f"\n✓ {sampler.dataset_name} saving representation completed successfully!")
        sys.exit(0)
    
    # Generate conditioning labels based on arguments
    conditioning_labels = None
    if not args.unconditional:
        if args.dev_activity is not None and args.hk_activity is not None:
            # User-specified activities
            conditioning_labels = torch.tensor([[args.dev_activity, args.hk_activity]], device=sampler.device).expand(args.num_samples, -1)
            print(f"Using specified activities: Dev={args.dev_activity}, HK={args.hk_activity}")
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

    # Run sampling using PC sampler
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

    # Handle returned result (may be just sequences or (sequences, saved_elements))
    if isinstance(result, tuple):
        sequences, saved_elements = result
    else:
        sequences = result
        saved_elements = None

    # Save sequences if output path provided
    if args.output:
        sampler.save_sequences(sequences, args.output, args.format, args.sequence_encoding)
        results = {
            'num_sequences': len(sequences),
            'sequence_length': sampler.get_sequence_length(config),
            'output_file': args.output,
            'encoding': args.sequence_encoding
        }
    else:
        results = {
            'num_sequences': len(sequences),
            'sequence_length': sampler.get_sequence_length(config)
        }

    # Save elements if requested
    if saved_elements:
        # Determine output directory (use same directory as sequence output if provided)
        if args.output:
            output_dir = Path(args.output).parent
            base_name = Path(args.output).stem
        else:
            output_dir = Path('.')
            base_name = 'deepstarr_samples'

        # Save all elements as datasets in a single HDF5 file
        output_file = output_dir / f"{base_name}_elements.h5"

        print(f"\nSaving sampling elements to {output_file}...")
        with h5py.File(output_file, 'w') as f:
            for elem_name, elem_tensor in saved_elements.items():
                # elem_tensor shape: (N, L, T, 4)
                f.create_dataset(elem_name, data=elem_tensor.numpy(), compression='gzip')
                print(f"  Saved dataset '{elem_name}': shape {elem_tensor.shape}")

            # Save metadata as attributes
            first_elem = list(saved_elements.values())[0]
            f.attrs['num_samples'] = first_elem.shape[0]
            f.attrs['sequence_length'] = first_elem.shape[1]
            f.attrs['num_timesteps'] = first_elem.shape[2]
            f.attrs['num_classes'] = first_elem.shape[3]
            f.attrs['saved_elements'] = list(saved_elements.keys())

        print(f"  All elements saved to: {output_file}")
        results['saved_elements_file'] = str(output_file)
        results['saved_elements'] = list(saved_elements.keys())

    # Print results
    print(f"\nDeepSTARR Sampling Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")

    print(f"\n✓ DeepSTARR sampling completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())