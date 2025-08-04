#!/usr/bin/env python3
"""
LentIMPRA Evaluation Script

Inherits from base evaluation framework while using LentIMPRA-specific models directly.
Uses the existing LegNet oracle from mpralegnet.py.
"""

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from typing import Optional
import h5py
import numpy as np

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import base framework and LentIMPRA-specific components
from scripts.evaluate import BaseEvaluator, parse_base_args
from model_zoo.lentimpra.data import get_lentimpra_datasets


class LentIMPRAEvaluator(BaseEvaluator):
    """LentIMPRA-specific evaluator that inherits from base framework."""
    
    def __init__(self):
        super().__init__("LentIMPRA")
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """Load LentIMPRA model using dataset-specific model loading."""
        from model_zoo.lentimpra.models import load_trained_model
        
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        """Get LentIMPRA sequence length."""
        return 230  # LentIMPRA fixed sequence length
    
    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None):
        """Create LentIMPRA dataloader."""
        # Load datasets
        train_ds, val_ds, test_ds = get_lentimpra_datasets(config.paths.data_file)
        
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
            batch_size = getattr(config.eval, 'batch_size', 256)
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    def load_oracle_model(self, oracle_checkpoint: str, data_path: str):
        """Load LentIMPRA oracle model using existing LegNet infrastructure."""
        try:
            # Import existing LegNet components from mpralegnet
            from model_zoo.lentimpra.mpralegnet import load_model
            
            # Check if config file exists alongside checkpoint
            oracle_dir = os.path.dirname(oracle_checkpoint)
            config_path = os.path.join(oracle_dir, 'config.json')
            
            if os.path.exists(config_path):
                # Load using existing load_model function
                oracle, config = load_model(oracle_checkpoint, config_path)
                oracle.eval()
                oracle.to(self.device)
                print(f"✓ Loaded LegNet oracle model from {oracle_checkpoint}")
                return oracle
            else:
                # Fallback: create default config and load checkpoint
                print(f"Config file not found at {config_path}, using default config")
                from model_zoo.lentimpra.mpralegnet import get_default_config, LitModel
                
                config = get_default_config()
                oracle = LitModel(config)
                
                # Load checkpoint weights
                checkpoint = torch.load(oracle_checkpoint, map_location=self.device)
                if 'state_dict' in checkpoint:
                    oracle.load_state_dict(checkpoint['state_dict'])
                else:
                    oracle.load_state_dict(checkpoint)
                
                oracle.eval()
                oracle.to(self.device)
                print(f"✓ Loaded LegNet oracle model from {oracle_checkpoint} (default config)")
                return oracle
                
        except Exception as e:
            print(f"Failed to load LentIMPRA oracle model: {e}")
            return None
    
    def get_original_test_data(self, data_path: str) -> torch.Tensor:
        """Get original test data for SP-MSE comparison."""
        try:
            # Load LentIMPRA test data from H5
            with h5py.File(data_path, 'r') as data_file:
                # Load one-hot data: (N, 230, 4)
                X = torch.tensor(np.array(data_file['onehot_test']))
            return X
        except Exception as e:
            print(f"Error loading original test data: {e}")
            return torch.zeros(100, 230, 4)
    
    def compute_sp_mse(self, sampled_sequences: torch.Tensor, oracle_model, 
                      original_data: torch.Tensor) -> float:
        """
        Compute SP-MSE using LentIMPRA oracle model (overrides base implementation).
        
        Args:
            sampled_sequences: Generated sequences (one-hot format)
            oracle_model: LentIMPRA oracle model with predict() method
            original_data: Original test data for comparison
            
        Returns:
            SP-MSE score
        """
        import torch.nn.functional as F
        
        # Convert sampled sequences from one-hot to format expected by oracle
        # sampled_sequences: (batch_size, seq_len, 4) one-hot
        # oracle expects: (batch_size, 4, seq_len) for LegNet
        if sampled_sequences.shape[-1] == 4:  # (batch_size, seq_len, 4)
            sampled_input = sampled_sequences.permute(0, 2, 1).to(self.device)  # -> (batch_size, 4, seq_len)
        else:  # Already (batch_size, 4, seq_len)
            sampled_input = sampled_sequences.to(self.device)
        
        # Convert original data to format expected by oracle
        # original_data: (batch_size, seq_len, 4) one-hot from H5
        if original_data.shape[-1] == 4:  # (batch_size, seq_len, 4)
            original_input = original_data.permute(0, 2, 1).to(self.device)  # -> (batch_size, 4, seq_len)
        else:  # Already (batch_size, 4, seq_len)
            original_input = original_data.to(self.device)
        
        # Get oracle predictions using LentIMPRA's predict method
        val_score = oracle_model.predict(original_input)
        val_pred_score = oracle_model.predict(sampled_input)
        
        # Compute SP-MSE
        sp_mse = (val_score - val_pred_score) ** 2
        mean_sp_mse = torch.mean(sp_mse).cpu().item()
        
        return mean_sp_mse


def load_default_config():
    """Load LentIMPRA default configuration (transformer)."""
    config_file = Path(__file__).parent / 'configs' / 'transformer.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main evaluation function using base framework."""
    # Parse arguments using base framework
    parser = parse_base_args()
    args = parser.parse_args()
    
    # Set save_sequences default to True for LentIMPRA
    if not hasattr(args, 'save_sequences') or args.save_sequences is False:
        args.save_sequences = True
        print("✓ LentIMPRA: Enabled sequence saving by default")
    
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
    evaluator = LentIMPRAEvaluator()
    
    # Use the base framework's evaluate_with_sampling method
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
    
    output_path = args.output or f"evaluation_results/lentimpra_{args.architecture}_{args.split}_results.json"
    evaluator.save_results(metrics, output_path)
    
    print(f"\n✓ LentIMPRA evaluation completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())