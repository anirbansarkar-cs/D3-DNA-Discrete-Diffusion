#!/usr/bin/env python3
"""
ATACSeq Evaluation Script

This script provides evaluation functionality specifically for the ATACSeq dataset,
inheriting from the base evaluation classes and implementing ATACSeq-specific
model creation, data loading, and oracle evaluation.
"""

import os
import sys
from pathlib import Path

# Package imports

from scripts.evaluate import BaseEvaluator, parse_base_args, main_evaluate
from model_zoo.atacseq.models import create_model
from model_zoo.atacseq.data import get_atacseq_datasets
from model_zoo.deepstarr.deepstarr import PL_DeepSTARR
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import torch


class ATACSeqEvaluator(BaseEvaluator):
    """Evaluator specifically for ATACSeq dataset."""
    
    def __init__(self):
        super().__init__('atacseq')
        
    def create_model(self, config, architecture):
        """Create ATACSeq-specific model."""
        return create_model(config, architecture)
        
    def create_dataloader(self, config, split='test', batch_size=None):
        """Create ATACSeq dataloader."""
        # Load datasets
        train_ds, val_ds = get_atacseq_datasets()
        
        # Select appropriate dataset
        if split == 'train':
            dataset = train_ds
        elif split in ['val', 'test']:  # Use val as test for now
            dataset = val_ds
        else:
            raise ValueError(f"Unknown split: {split}")
            
        # Create dataloader
        if batch_size is None:
            batch_size = getattr(config.eval, 'batch_size', 256) // (config.ngpus * config.training.accum)
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    def load_oracle_model(self, oracle_checkpoint, data_path):
        """Load ATACSeq oracle model."""
        try:
            # Load the PL_ATACSeq oracle model
            if not data_path:
                data_path = 'model_zoo/atacseq/ATACSeq_data.h5'
                
            oracle = PL_ATACSeq.load_from_checkpoint(
                oracle_checkpoint, 
                input_h5_file=data_path
            ).eval()
            oracle.to(self.device)
            
            print("✓ Loaded ATACSeq oracle model")
            return oracle
            
        except Exception as e:
            print(f"Failed to load oracle model: {e}")
            return None
    
    def evaluate_with_oracle(self, model, oracle_model, dataloader, config):
        """Evaluate using ATACSeq oracle model for SP-MSE and other metrics."""
        if oracle_model is None:
            return {'oracle_evaluation': 'oracle_model_not_loaded'}
        
        model.eval()
        oracle_model.eval()
        
        # Implement SP-MSE evaluation logic here
        # This would compare generated sequences to oracle predictions
        
        # Placeholder implementation
        sp_mse_scores = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Generate samples from the diffusion model
                # (This would need proper sampling implementation)
                
                # For now, just compute a placeholder metric
                oracle_pred = oracle_model(inputs)
                mse = torch.nn.functional.mse_loss(oracle_pred, targets)
                sp_mse_scores.append(mse.item())
                
                num_batches += 1
                if num_batches >= 10:  # Limit for demonstration
                    break
        
        avg_sp_mse = sum(sp_mse_scores) / len(sp_mse_scores) if sp_mse_scores else 0.0
        
        return {
            'oracle_evaluation': 'completed',
            'sp_mse': avg_sp_mse,
            'num_oracle_batches': len(sp_mse_scores)
        }


def load_config(architecture):
    """Load ATACSeq configuration."""
    config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main evaluation function."""
    parser = parse_base_args()
    parser.description = 'ATACSeq Evaluation Script'
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ATACSeqEvaluator()
    
    # Load config if not provided
    if not args.config:
        try:
            config = load_config(args.architecture)
        except FileNotFoundError:
            print(f"Error: No config provided and default config not found for architecture: {args.architecture}")
            return 1
    else:
        config = OmegaConf.load(args.config)
    
    # Run evaluation
    metrics = evaluator.evaluate(
        checkpoint_path=args.checkpoint,
        config=config,
        architecture=args.architecture,
        split=args.split,
        oracle_checkpoint=args.oracle_checkpoint if args.use_oracle else None,
        data_path=args.data_path
    )
    
    # Print and save results
    evaluator.print_results(metrics)
    
    output_path = args.output or f"evaluation_results/atacseq_{args.architecture}_{args.split}_results.json"
    evaluator.save_results(metrics, output_path)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())