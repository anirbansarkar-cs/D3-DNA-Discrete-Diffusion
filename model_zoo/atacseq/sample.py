#!/usr/bin/env python3
"""
DeepSTARR Sampling Script

This script provides sampling functionality specifically for the DeepSTARR dataset,
inheriting from the base sampling classes and implementing DeepSTARR-specific
model creation and label generation.
"""

import os
import sys
from pathlib import Path

# Package imports

from scripts.sample import BaseSampler, parse_base_args, main_sample
from model_zoo.deepstarr.models import create_model
from omegaconf import OmegaConf
import torch


class DeepSTARRSampler(BaseSampler):
    """Sampler specifically for DeepSTARR dataset."""
    
    def __init__(self):
        super().__init__('deepstarr')
        
    def create_model(self, config, architecture):
        """Create DeepSTARR-specific model."""
        return create_model(config, architecture)
        
    def get_sequence_length(self, config):
        """Get DeepSTARR sequence length."""
        # DeepSTARR uses 249 bp sequences
        if hasattr(config, 'model') and hasattr(config.model, 'length'):
            return config.model.length
        return 249
    
    def generate_labels(self, num_samples, config):
        """
        Generate conditioning labels for DeepSTARR sampling.
        
        DeepSTARR has two labels:
        - Dev enhancer activity
        - HK enhancer activity  
        """
        # Generate random labels in the range typical for DeepSTARR targets
        # You might want to adjust these ranges based on your specific needs
        dev_labels = torch.randn(num_samples, 1, device=self.device) * 2.0  # Adjust scale as needed
        hk_labels = torch.randn(num_samples, 1, device=self.device) * 2.0   # Adjust scale as needed
        
        # Combine into shape (num_samples, 2)
        labels = torch.cat([dev_labels, hk_labels], dim=1)
        
        return labels
    
    def sample_with_specific_labels(self, checkpoint_path, config, architecture,
                                   dev_activity=None, hk_activity=None, **kwargs):
        """
        Sample sequences with specific enhancer activity targets.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration object
            architecture: Architecture name
            dev_activity: Target developmental enhancer activity (float or list)
            hk_activity: Target housekeeping enhancer activity (float or list)
            **kwargs: Additional sampling arguments
        """
        # Load model
        model = self.load_model_from_checkpoint(checkpoint_path, config, architecture)
        
        num_samples = kwargs.get('num_samples', 1000)
        
        # Create specific labels if provided
        if dev_activity is not None or hk_activity is not None:
            if isinstance(dev_activity, (int, float)):
                dev_activity = [dev_activity] * num_samples
            if isinstance(hk_activity, (int, float)):
                hk_activity = [hk_activity] * num_samples
                
            if dev_activity is None:
                dev_activity = [0.0] * num_samples
            if hk_activity is None:
                hk_activity = [0.0] * num_samples
                
            # Create label tensor
            labels = torch.tensor([
                [dev, hk] for dev, hk in zip(dev_activity, hk_activity)
            ], device=self.device, dtype=torch.float32)
            
            # Override the generate_labels method for this call
            original_generate_labels = self.generate_labels
            self.generate_labels = lambda n, cfg: labels
            
            try:
                sequences = self.sample_sequences(model, config, num_samples, **kwargs)
            finally:
                # Restore original method
                self.generate_labels = original_generate_labels
        else:
            sequences = self.sample_sequences(model, config, num_samples, **kwargs)
        
        return sequences


def load_config(architecture):
    """Load DeepSTARR configuration."""
    config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main sampling function."""
    parser = parse_base_args()
    parser.description = 'DeepSTARR Sampling Script'
    
    # Add DeepSTARR-specific arguments
    parser.add_argument('--dev_activity', type=float, help='Target developmental enhancer activity')
    parser.add_argument('--hk_activity', type=float, help='Target housekeeping enhancer activity')
    
    args = parser.parse_args()
    
    # Create sampler
    sampler = DeepSTARRSampler()
    
    # Load config if not provided
    if not args.config:
        try:
            config = load_config(args.architecture)
        except FileNotFoundError:
            print(f"Error: No config provided and default config not found for architecture: {args.architecture}")
            return 1
    else:
        config = OmegaConf.load(args.config)
    
    # Use specific label sampling if requested
    if args.dev_activity is not None or args.hk_activity is not None:
        sequences = sampler.sample_with_specific_labels(
            checkpoint_path=args.checkpoint,
            config=config,
            architecture=args.architecture,
            dev_activity=args.dev_activity,
            hk_activity=args.hk_activity,
            num_samples=args.num_samples,
            method=args.method,
            num_steps=args.num_steps,
            eta=args.eta,
            temperature=args.temperature
        )
        
        # Save sequences
        output_path = args.output or f"samples/deepstarr_{args.architecture}_{args.method}_specific_samples.{args.format}"
        sampler.save_sequences(sequences, output_path, args.format)
        
        print(f"Sampling completed with specific labels. {args.num_samples} sequences generated.")
    else:
        # Use standard sampling
        sequences = sampler.sample(
            checkpoint_path=args.checkpoint,
            config=config,
            architecture=args.architecture,
            num_samples=args.num_samples,
            method=args.method,
            num_steps=args.num_steps,
            output_path=args.output,
            format=args.format,
            eta=args.eta,
            temperature=args.temperature
        )
        
        print(f"Sampling completed. {args.num_samples} sequences generated.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())