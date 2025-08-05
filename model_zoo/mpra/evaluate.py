#!/usr/bin/env python3
"""
MPRA Evaluation Script

Inherits from base evaluation framework while using MPRA-specific models directly.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from utils.utils import update_cfg_with_unknown_args
from omegaconf import OmegaConf
from typing import Optional
from tqdm import tqdm
import h5py
import numpy as np

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import base framework and MPRA-specific components
from scripts.evaluate import BaseEvaluator, parse_base_args, main_evaluate
from model_zoo.mpra.data import get_mpra_datasets
from model_zoo.mpra.mpra import PL_MPRA


class MPRAEvaluator(BaseEvaluator):
    """MPRA-specific evaluator that inherits from base framework."""
    
    def __init__(self):
        super().__init__("MPRA")
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """Load MPRA model using dataset-specific model loading."""
        from model_zoo.mpra.models import load_trained_model
        
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None):
        """Create MPRA dataloader."""
        # Load datasets
        train_ds, val_ds, test_ds = get_mpra_datasets(config.paths.data_file)
        
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
            batch_size = getattr(config.eval, 'batch_size', 32)
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    def load_oracle_model(self, oracle_checkpoint: str, data_path: str, use_evoaug_oracle: bool = False):
        """Load MPRA oracle model."""
        try:
            import os
            # Check if data_path is empty, a directory, or doesn't exist
            if not data_path or os.path.isdir(data_path) or not os.path.exists(data_path):
                data_path = 'model_zoo/mpra/mpra_data.h5'
                print(f"Using default data path: {data_path}")
            
            # Load standard Lightning oracle model
            oracle = PL_MPRA.load_from_checkpoint(
                oracle_checkpoint, 
                input_h5_file=data_path
            ).eval()
            oracle.to(self.device)
            
            print("✓ Loaded MPRA oracle model")
            return oracle
            
        except Exception as e:
            print(f"Failed to load MPRA oracle model: {e}")
            return None
    
    def get_original_test_data(self, data_path: str) -> torch.Tensor:
        """Get original test data for SP-MSE comparison."""
        try:            
            # Load MPRA test data h5  
            print(f"Loading original test data from: {data_path}")
            with h5py.File(data_path, 'r') as data_file:
                # MPRA uses x_test key
                X = torch.tensor(np.array(data_file['x_test']))
                
            return X
        except Exception as e:
            print(f"Error loading original test data: {e}")
            return torch.zeros(100, 4, 200)  # Default MPRA sequence length
    
    def evaluate_with_sampling(self, checkpoint_path: str, config: OmegaConf, 
                              oracle_checkpoint: str, data_path: str,
                              split: str = 'test', steps: Optional[int] = None, 
                              batch_size: Optional[int] = None, architecture: str = 'transformer',
                              show_progress: bool = False, save_sequences: bool = False):
        """
        Override base method for MPRA-specific evaluation.
        """
        print(f"Evaluating {self.dataset_name} on {split} split with sampling...")
        
        # Set default steps to sequence length if not provided
        if steps is None:
            steps = self.get_sequence_length(config)
            print(f"Using default steps: {steps} (sequence length)")
        
        # Create dataloader
        dataloader = self.create_dataloader(config, split, batch_size)
        
        # Sample sequences using PC sampler
        print(f"Sampling sequences with PC sampler ({steps} steps)...")
        sampled_sequences, target_labels = self.sample_sequences_for_evaluation(
            checkpoint_path, config, dataloader, steps, architecture, show_progress
        )
        
        # Save sequences as NPZ if requested
        if save_sequences:
            # Create output path based on checkpoint directory
            checkpoint_dir = os.path.dirname(checkpoint_path)
            npz_path = os.path.join(checkpoint_dir, "sample.npz")
            self.save_sequences_as_npz(sampled_sequences, npz_path)
        
        # Load oracle model
        print("Loading oracle model for SP-MSE evaluation...")
        oracle_model = self.load_oracle_model(oracle_checkpoint, data_path)
        
        if oracle_model is None:
            return {
                'error': 'oracle_model_not_loaded',
                'num_samples': sampled_sequences.shape[0],
                'sequence_length': sampled_sequences.shape[1],
                'sampling_steps': steps
            }
        
        # Get original test data for comparison
        original_data = self.get_original_test_data(data_path)
        
        # Compute SP-MSE
        print("Computing SP-MSE...")
        sp_mse = self.compute_sp_mse(sampled_sequences, oracle_model, original_data)
        
        results = {
            'dataset': self.dataset_name,
            'split': split,
            'num_samples': sampled_sequences.shape[0],
            'sequence_length': sampled_sequences.shape[1],
            'sampling_steps': steps,
            'sp_mse': sp_mse,
            'oracle_evaluation': 'completed'
        }
        
        print(f"SP-MSE: {sp_mse:.6f}")
        
        return results


def load_default_config():
    """Load MPRA default configuration (transformer)."""
    config_file = Path(__file__).parent / 'configs' / 'transformer.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main evaluation function using base framework."""
    # Parse arguments using base framework
    parser = parse_base_args()
    args = parser.parse_args()
    
    # Set save_sequences default to True for MPRA
    if not hasattr(args, 'save_sequences') or args.save_sequences is False:
        args.save_sequences = True
        print("✓ MPRA: Enabled sequence saving by default")
    
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

    # override paths.data_file with args.data_path if provided
    if args.data_path:
        config.paths.data_file = args.data_path
        print(f"Overriding paths.data_file with {args.data_path}")

    evaluator = MPRAEvaluator()
    
    # Run evaluation (always includes sampling + SP-MSE computation)
    metrics = evaluator.evaluate_with_sampling(
        checkpoint_path=args.checkpoint,
        config=config,
        oracle_checkpoint=args.oracle_checkpoint or config.paths.get('oracle_model'),
        data_path=args.data_path or config.paths.get('data_file'),
        split=args.split,
        steps=args.steps,
        batch_size=args.batch_size,
        architecture=args.architecture,
        show_progress=args.show_progress,
        save_sequences=args.save_sequences
    )
    
    # Print and save results
    evaluator.print_results(metrics)
    
    output_path = args.output or f"evaluation_results/mpra_{args.architecture}_{args.split}_results.json"
    evaluator.save_results(metrics, output_path)
    
    print(f"\n✓ MPRA evaluation completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())