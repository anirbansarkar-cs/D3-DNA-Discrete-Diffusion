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
        self._dataset_indices = None  # Store indices for matching original data
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """Load DeepSTARR model using dataset-specific model loading."""
        from model_zoo.deepstarr.models import load_trained_model
        
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None, max_samples: Optional[int] = None):
        """Create DeepSTARR dataloader with optional sample limiting."""
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
        
        # Limit dataset size if max_samples is specified
        if max_samples is not None and len(dataset) > max_samples:
            # Create random subset
            import torch.utils.data as data_utils
            indices = torch.randperm(len(dataset))[:max_samples]
            dataset = data_utils.Subset(dataset, indices)
            # Store the indices for matching original data later
            self._dataset_indices = indices
            print(f"  ↳ DeepSTARR dataset limited to {len(dataset)} samples from {split} split")
        else:
            # Full dataset - no indices needed
            self._dataset_indices = None
            
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
        """Load DeepSTARR oracle model (EvoAug or standard)."""
        try:
            import os
            # Check if data_path is empty, a directory, or doesn't exist
            if not data_path or os.path.isdir(data_path) or not os.path.exists(data_path):
                data_path = 'model_zoo/deepstarr/DeepSTARR_data.h5'
                print(f"Using default data path: {data_path}")
            
            if use_evoaug_oracle:
                # Load EvoAug oracle model
                from model_zoo.deepstarr.deepstarr import load_evoaug_oracle_model
                oracle = load_evoaug_oracle_model(oracle_checkpoint, device=self.device)
                print("✓ Loaded EvoAug DeepSTARR oracle model")
                return oracle
            else:
                # Load standard Lightning oracle model
                oracle = PL_DeepSTARR.load_from_checkpoint(
                    oracle_checkpoint, 
                    input_h5_file=data_path
                ).eval()
                oracle.to(self.device)
                
                print("✓ Loaded standard DeepSTARR oracle model")
                return oracle
            
        except Exception as e:
            print(f"Failed to load DeepSTARR oracle model: {e}")
            return None
    
    def get_original_test_data(self, data_path: str) -> torch.Tensor:
        """Get original test data for SP-MSE comparison, matching the limited dataset if applicable."""
        try:            
            # Load DeepSTARR test data h5  
            print(f"Loading original test data from: {data_path}")
            with h5py.File(data_path, 'r') as data_file:
                X = torch.tensor(np.array(data_file['X_test']))
            
            # If we limited the dataset, apply the same indices to original data
            if self._dataset_indices is not None:
                print(f"  ↳ Applying same subset indices to original data ({len(self._dataset_indices)} samples)")
                X = X[self._dataset_indices]
                
            return X
        except Exception as e:
            print(f"Error loading original test data: {e}")
            return torch.zeros(100, 4, 249)
    
    def evaluate_with_sampling(self, checkpoint_path: str, config: OmegaConf, 
                              oracle_checkpoint: str, data_path: str,
                              split: str = 'test', steps: Optional[int] = None, 
                              batch_size: Optional[int] = None, architecture: str = 'transformer',
                              show_progress: bool = False, save_sequences: bool = False,
                              save_visualization_data: bool = False, viz_output_path: Optional[str] = None,
                              viz_format: str = 'hdf5', max_samples: Optional[int] = None):
        """
        Override base method to pass EvoAug oracle flag from config and handle visualization.
        """
        print(f"Evaluating {self.dataset_name} on {split} split with sampling...")
        
        # Set default steps to sequence length if not provided
        if steps is None:
            steps = self.get_sequence_length(config)
            print(f"Using default steps: {steps} (sequence length)")
        
        # Create dataloader with optional sample limiting
        dataloader = self.create_dataloader(config, split, batch_size, max_samples)
        
        # Load oracle model with EvoAug flag from config
        print("Loading oracle model for SP-MSE evaluation...")
        use_evoaug_oracle = getattr(config.eval, 'use_evoaug_oracle', False)
        oracle_model = self.load_oracle_model(oracle_checkpoint, data_path, use_evoaug_oracle)
        
        if oracle_model is None:
            return {
                'error': 'oracle_model_not_loaded',
                'sampling_steps': steps
            }
        
        # Get original test data for comparison and visualization
        original_data = self.get_original_test_data(data_path)
        
        # Create visualization logger if requested
        viz_logger = None
        if save_visualization_data:
            from utils.visualization_logger import create_visualization_logger
            sequence_length = self.get_sequence_length(config)
            
            # Convert original samples to token indices for visualization storage
            # Keep original_data in one-hot format for SP-MSE computation
            original_samples_indices = None
            if original_data is not None:
                # Convert from (batch_size, 4, seq_length) to (batch_size, seq_length)
                original_samples_indices = torch.argmax(original_data, dim=1)
            
            viz_logger = create_visualization_logger(
                num_samples=len(dataloader.dataset),
                sequence_length=sequence_length,
                num_steps=steps,
                dataset_name=self.dataset_name,
                architecture=architecture,
                split=split,
                save_oracle_mse=True,  # Enable oracle MSE for evaluation
                device=self.device,
                original_samples=original_samples_indices  # Add original samples as token indices
            )
            print("  ↳ Visualization data logging enabled with oracle MSE and original samples")
        
        # Sample sequences using PC sampler
        print(f"Sampling sequences with PC sampler ({steps} steps)...")
        sampled_sequences, target_labels = self.sample_sequences_for_evaluation(
            checkpoint_path, config, dataloader, steps, architecture, show_progress, viz_logger, oracle_model
        )
        
        # Save sequences as NPZ if requested
        if save_sequences:
            # Create output path based on checkpoint directory
            checkpoint_dir = os.path.dirname(checkpoint_path)
            npz_path = os.path.join(checkpoint_dir, "sample.npz")
            self.save_sequences_as_npz(sampled_sequences, npz_path)
        
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
            'oracle_evaluation': 'completed',
            'use_evoaug_oracle': use_evoaug_oracle
        }
        
        # Save visualization data if requested
        if save_visualization_data and viz_logger is not None:
            if viz_output_path is None:
                # Auto-generate visualization output path
                checkpoint_dir = os.path.dirname(checkpoint_path)
                viz_output_path = os.path.join(checkpoint_dir, f"deepstarr_evaluation_visualization_data.{viz_format}")
            
            viz_logger.save(viz_output_path, viz_format)
            results['visualization_output_path'] = viz_output_path
        
        print(f"SP-MSE: {sp_mse:.6f}")
        
        return results
    
    def get_oracle_predictions_for_viz(self, sequences: torch.Tensor, oracle_model) -> torch.Tensor:
        """
        DeepSTARR-specific oracle predictions for visualization.
        
        Args:
            sequences: One-hot encoded sequences (batch_size, seq_length, 4)
            oracle_model: DeepSTARR oracle model
            
        Returns:
            Oracle predictions tensor (batch_size, 2) for Dev and Hk activities
        """
        if hasattr(oracle_model, 'predict_custom'):
            # Convert from (batch, length, channels) to (batch, channels, length)
            sequences_input = sequences.permute(0, 2, 1).to(self.device)
            return oracle_model.predict_custom(sequences_input)
        else:
            # Fallback
            return torch.zeros(sequences.shape[0], 2, device=self.device)


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
        oracle_checkpoint=args.oracle_checkpoint or config.paths.get('oracle_model'),
        data_path=args.data_path or config.paths.get('data_file'),
        split=args.split,
        steps=args.steps,
        batch_size=args.batch_size,
        architecture=args.architecture,
        show_progress=getattr(args, 'show_progress', False),  # DeepSTARR default: False
        save_sequences=args.save_sequences,
        save_visualization_data=getattr(args, 'save_viz_data', False),
        viz_output_path=getattr(args, 'viz_output', None),
        viz_format=getattr(args, 'viz_format', 'hdf5'),
        max_samples=getattr(args, 'max_samples', None)
    )
    
    # Print and save results
    evaluator.print_results(metrics)
    
    output_path = args.output or f"evaluation_results/deepstarr_{args.architecture}_{args.split}_results.json"
    evaluator.save_results(metrics, output_path)
    
    print(f"\n✓ DeepSTARR evaluation completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())