#!/usr/bin/env python3
"""
MPRA Sampling Script

This script provides sampling functionality specifically for the MPRA dataset,
inheriting from the base sampling framework and implementing MPRA-specific
model loading and data handling.
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import base framework and MPRA-specific components
from scripts.sample import BaseSampler, parse_base_args, main_sample
from model_zoo.mpra.models import load_trained_model
from model_zoo.mpra.data import get_mpra_datasets


class MPRASampler(BaseSampler):
    """MPRA-specific sampler that inherits from base framework."""
    
    def __init__(self):
        super().__init__("MPRA")
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """Load MPRA model using dataset-specific model loading."""
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def get_conditioning_data(self, config: OmegaConf, num_samples: int):
        """
        Get conditioning data for MPRA sampling.
        
        Args:
            config: Configuration object
            num_samples: Number of samples to generate
            
        Returns:
            Conditioning tensor with shape (num_samples, signal_dim)
        """
        # For MPRA, we have 3 cell lines (signal_dim=3)
        # Sample random conditioning values or use specific targets
        signal_dim = config.dataset.signal_dim
        
        # Generate random conditioning values for the 3 cell lines
        # Values are typically in a reasonable range for MPRA data
        conditioning = torch.randn(num_samples, signal_dim) * 0.5
        
        return conditioning.to(self.device)
    
    def create_validation_dataloader(self, config: OmegaConf, batch_size: int = None):
        """Create MPRA validation dataloader for conditional sampling."""
        from torch.utils.data import DataLoader
        
        # Get MPRA datasets
        _, val_ds, _ = get_mpra_datasets(config.paths.data_file)
        
        if batch_size is None:
            batch_size = getattr(config.eval, 'batch_size', 32)
        
        return DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    def sample_conditional(self, checkpoint_path: str, config: OmegaConf, 
                          num_samples: int = 1000, steps: int = None,
                          architecture: str = 'transformer', batch_size: int = 32,
                          output_path: str = None, show_progress: bool = True):
        """
        Sample MPRA sequences conditionally based on target activity values.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration object
            num_samples: Number of sequences to sample
            steps: Number of sampling steps (default: sequence_length)
            architecture: Model architecture ('transformer' or 'convolutional')
            batch_size: Batch size for sampling
            output_path: Path to save samples (default: checkpoint_dir/samples.npz)
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with sampled sequences and conditioning values
        """
        print(f"Sampling {num_samples} MPRA sequences conditionally...")
        
        # Load model
        model, graph, noise = self.load_model(checkpoint_path, config, architecture)
        
        # Get conditioning data
        conditioning = self.get_conditioning_data(config, num_samples)
        
        # Set default steps
        if steps is None:
            steps = config.dataset.sequence_length
        
        # Sample sequences
        sampled_sequences = self.sample_sequences(
            model, graph, noise, conditioning, steps, batch_size, show_progress
        )
        
        # Prepare results
        results = {
            'sequences': sampled_sequences.cpu().numpy(),
            'conditioning': conditioning.cpu().numpy(),
            'num_samples': num_samples,
            'steps': steps,
            'architecture': architecture
        }
        
        # Save results
        if output_path is None:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            output_path = os.path.join(checkpoint_dir, f"mpra_samples_{num_samples}.npz")
        
        self.save_samples(results, output_path)
        
        print(f"✓ Saved {num_samples} MPRA samples to {output_path}")
        
        return results


def load_default_config():
    """Load MPRA default configuration (transformer)."""
    config_file = Path(__file__).parent / 'configs' / 'transformer.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main sampling function using base framework."""
    # Parse arguments using base framework
    parser = parse_base_args()
    parser.add_argument('--conditional', action='store_true', 
                       help='Sample conditionally based on target activity values')
    args = parser.parse_args()
    
    # Load config if not provided
    if not args.config:
        try:
            config_path = Path(__file__).parent / 'configs' / 'transformer.yaml'
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
    
    # Override paths if provided
    if args.data_path:
        config.paths.data_file = args.data_path
        print(f"Overriding paths.data_file with {args.data_path}")
    
    sampler = MPRASampler()
    
    if args.conditional:
        # Run conditional sampling
        results = sampler.sample_conditional(
            checkpoint_path=args.checkpoint,
            config=config,
            num_samples=args.num_samples,
            steps=args.steps,
            architecture=args.architecture,
            batch_size=args.batch_size,
            output_path=args.output,
            show_progress=args.show_progress
        )
        
        print(f"\n✓ MPRA conditional sampling completed successfully!")
        print(f"  - Generated {results['num_samples']} sequences")
        print(f"  - Sampling steps: {results['steps']}")
        print(f"  - Architecture: {results['architecture']}")
    else:
        # Run standard sampling using base framework
        results = main_sample(args, sampler)
        print(f"\n✓ MPRA sampling completed successfully!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())