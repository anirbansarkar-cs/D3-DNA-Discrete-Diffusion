#!/usr/bin/env python3
"""
DeepSTARR Evaluation Script

Inherits from base evaluation framework while using DeepSTARR-specific models directly.
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

# Import base framework and DeepSTARR-specific components
from scripts.evaluate import BaseEvaluator, parse_base_args, main_evaluate
from model_zoo.deepstarr.data import get_deepstarr_datasets
from model_zoo.deepstarr.deepstarr import PL_DeepSTARR


class DeepSTARREvaluator(BaseEvaluator):
    """DeepSTARR-specific evaluator that inherits from base framework."""
    
    def __init__(self):
        super().__init__("DeepSTARR")
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """Load DeepSTARR model using dataset-specific model loading."""
        from model_zoo.deepstarr.models import load_trained_model
        
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
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
    
    def load_oracle_model(self, oracle_checkpoint: str, data_path: str):
        """Load DeepSTARR oracle model."""
        try:
            import os
            # Check if data_path is empty, a directory, or doesn't exist
            if not data_path or os.path.isdir(data_path) or not os.path.exists(data_path):
                data_path = 'model_zoo/deepstarr/DeepSTARR_data.h5'
                print(f"Using default data path: {data_path}")
                
            oracle = PL_DeepSTARR.load_from_checkpoint(
                oracle_checkpoint, 
                input_h5_file=data_path
            ).eval()
            oracle.to(self.device)
            
            print("✓ Loaded DeepSTARR oracle model")
            return oracle
            
        except Exception as e:
            print(f"Failed to load DeepSTARR oracle model: {e}")
            return None
    
    def get_original_test_data(self, data_path: str) -> torch.Tensor:
        """Get original test data for SP-MSE comparison."""
        try:
            # Load DeepSTARR test data h5  
            with h5py.File(data_path, 'r') as data_file:
                X = torch.tensor(np.array(data_file['X_test']))
            return X
        except Exception as e:
            print(f"Error loading original test data: {e}")
            return torch.zeros(100, 4, 249)


def load_default_config():
    """Load DeepSTARR default configuration (transformer)."""
    config_file = Path(__file__).parent / 'configs' / 'transformer.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main evaluation function using base framework."""
    # Parse arguments using base framework
    parser = parse_base_args()
    args = parser.parse_args()
    
    # Set save_sequences default to True for DeepSTARR
    if not hasattr(args, 'save_sequences') or args.save_sequences is False:
        args.save_sequences = True
        print("✓ DeepSTARR: Enabled sequence saving by default")
    
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

    evaluator = DeepSTARREvaluator()
    
    # Run evaluation (always includes sampling + SP-MSE computation)
    metrics = evaluator.evaluate_with_sampling(
        checkpoint_path=args.checkpoint,
        config=config,
        oracle_checkpoint=args.oracle_checkpoint,
        data_path=args.data_path,
        split=args.split,
        steps=args.steps,
        batch_size=args.batch_size,
        architecture=args.architecture,
        show_progress=args.show_progress,
        save_sequences=args.save_sequences
    )
    
    # Print and save results
    evaluator.print_results(metrics)
    
    output_path = args.output or f"evaluation_results/deepstarr_{args.architecture}_{args.split}_results.json"
    evaluator.save_results(metrics, output_path)
    
    print(f"\n✓ DeepSTARR evaluation completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())