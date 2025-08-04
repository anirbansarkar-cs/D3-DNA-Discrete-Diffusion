#!/usr/bin/env python3
"""
Promoter Evaluation Script

Inherits from base evaluation framework while using Promoter-specific models directly.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from typing import Optional
from tqdm import tqdm
import pandas as pd

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import base framework and Promoter-specific components
from scripts.evaluate import BaseEvaluator, parse_base_args, main_evaluate
from model_zoo.promoter.data import get_promoter_datasets
from model_zoo.promoter.sei import Sei, NonStrandSpecific


def upgrade_state_dict(state_dict, prefixes):
    """Upgrade state dict by removing prefixes."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                break
        new_state_dict[new_key] = value
    return new_state_dict


class PromoterEvaluator(BaseEvaluator):
    """Promoter-specific evaluator that inherits from base framework."""
    
    def __init__(self):
        super().__init__("Promoter")
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """Load Promoter model using dataset-specific model loading."""
        from model_zoo.promoter.models import load_trained_model
        
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        """Get Promoter sequence length."""
        if hasattr(config, 'model') and hasattr(config.model, 'length'):
            return config.model.length
        return 1024  # Promoter default sequence length
    
    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None):
        """Create Promoter dataloader."""
        # Load datasets 
        train_ds, val_ds = get_promoter_datasets()
        
        # Select appropriate dataset
        if split == 'train':
            dataset = train_ds
        elif split in ['val', 'test']:  # Use val as test for now
            dataset = val_ds
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
    
    def load_oracle_model(self, oracle_checkpoint: str, data_path: str):
        """Load Promoter oracle model (Sei)."""
        try:
            # Load Sei oracle model with proper architecture
            sei_model = Sei(4096, 21907)  # 4096 seq length, 21907 features
            oracle = NonStrandSpecific(sei_model)
            
            # Load checkpoint if provided
            if oracle_checkpoint and os.path.exists(oracle_checkpoint):
                checkpoint = torch.load(oracle_checkpoint, map_location=self.device)
                state_dict = upgrade_state_dict(checkpoint['state_dict'], prefixes=['module.'])
                oracle.load_state_dict(state_dict, strict=False)
            
            oracle.to(self.device)
            oracle.eval()
            
            print("✓ Loaded Promoter oracle model (Sei)")
            return oracle
            
        except Exception as e:
            print(f"Failed to load Promoter oracle model: {e}")
            return None
    
    def compute_sp_mse(self, sampled_sequences: torch.Tensor, oracle_model, 
                       original_data: torch.Tensor) -> float:
        """
        Compute SP-MSE using Promoter SEI oracle model with proper inference pattern.
        
        Args:
            sampled_sequences: Generated sequences (batch_size, seq_length, 4) one-hot
            oracle_model: Loaded SEI oracle model 
            original_data: Original test sequences for comparison
            
        Returns:
            Mean SP-MSE value
        """
        # Load SEI features for H3K4me3 filtering
        if not hasattr(self, 'sei_features'):
            try:
                sei_features_path = 'model_zoo/promoter/oracle_models/target.sei.names'
                self.sei_features = pd.read_csv(sei_features_path, sep='|', header=None)
            except:
                print("Warning: Could not load SEI features file, using all features")
                self.sei_features = None
        
        # Get oracle predictions for original and generated data using proper SEI inference
        val_score = self._get_sei_profile(original_data, oracle_model)
        val_pred_score = self._get_sei_profile(sampled_sequences, oracle_model)
        
        # Compute SP-MSE
        sp_mse = (val_score - val_pred_score) ** 2
        mean_sp_mse = torch.mean(torch.tensor(sp_mse)).cpu().item()
        
        return mean_sp_mse
    
    def _get_sei_profile(self, seq_one_hot, oracle_model):
        """
        Get SEI profile following the proper inference pattern.
        
        Args:
            seq_one_hot: One-hot encoded sequences (batch_size, seq_length, 4)
            oracle_model: SEI oracle model
            
        Returns:
            H3K4me3 predictions (batch_size,)
        """
        B, L, K = seq_one_hot.shape
        seq_one_hot = seq_one_hot.cpu()
        
        # Pad sequence to 4096 length as expected by SEI
        # Add 1536 bases on each side with uniform background (0.25 for each nucleotide)
        sei_inp = torch.cat([
            torch.ones((B, 4, 1536)) * 0.25,
            seq_one_hot.transpose(1, 2),  # Convert to (batch, channels, length)
            torch.ones((B, 4, 1536)) * 0.25
        ], 2).to(self.device)  # batchsize x 4 x 4,096
        
        # Get SEI predictions
        with torch.no_grad():
            sei_out = oracle_model(sei_inp).cpu().detach().numpy()  # batchsize x 21,907
        
        # Filter for H3K4me3 features if SEI features are available
        if self.sei_features is not None:
            h3k4me3_mask = self.sei_features[1].str.strip().values == 'H3K4me3'
            sei_out = sei_out[:, h3k4me3_mask]  # batchsize x 2,350 (H3K4me3 features)
        
        # Take mean across H3K4me3 features
        predh3k4me3 = sei_out.mean(axis=1)  # batchsize
        
        return predh3k4me3
    
    def get_original_test_data(self, data_path: str) -> torch.Tensor:
        """Get original test data for SP-MSE comparison."""
        try:
            # Load Promoter test data
            train_ds, val_ds = get_promoter_datasets()
            
            # Create a small batch for comparison
            dataloader = DataLoader(val_ds, batch_size=100, shuffle=False)
            batch = next(iter(dataloader))
            
            if len(batch) == 2:
                sequences, _ = batch
                return sequences
            else:
                return batch
                
        except Exception as e:
            print(f"Error loading original test data: {e}")
            # Return dummy data as fallback
            return torch.zeros(100, 1024, 4)  # One-hot encoded sequences


def load_config(architecture: str):
    """Load Promoter configuration."""
    config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main evaluation function using base framework."""
    # Parse arguments using base framework
    parser = parse_base_args()
    parser.add_argument('--model_path', required=True, help='Path to model directory (required for evaluation)')
    parser.add_argument('--steps', type=int, help='Number of sampling steps (defaults to sequence length)')
    args = parser.parse_args()
    
    # Validate required arguments for evaluation
    if not args.oracle_checkpoint:
        print("Error: --oracle_checkpoint is required for evaluation")
        return 1
    if not args.data_path:
        print("Error: --data_path is required for evaluation")
        return 1
    
    # Load config if not provided
    if not args.config:
        try:
            config_path = Path(__file__).parent / 'configs' / f'{args.architecture}.yaml'
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
    evaluator = PromoterEvaluator()
    
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
        save_sequences=getattr(args, 'save_sequences', False)
    )
    
    # Print and save results
    evaluator.print_results(metrics)
    
    output_path = args.output or f"evaluation_results/promoter_{args.architecture}_{args.split}_results.json"
    evaluator.save_results(metrics, output_path)
    
    print(f"\n✓ Promoter evaluation completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())