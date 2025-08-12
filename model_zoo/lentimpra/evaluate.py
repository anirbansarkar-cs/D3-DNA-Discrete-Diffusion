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
from typing import Optional, Tuple
from tqdm import tqdm
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
        self._dataset_indices = None  # Store indices for matching original data
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """Load LentIMPRA model using dataset-specific model loading."""
        from model_zoo.lentimpra.models import load_trained_model
        
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        """Get LentIMPRA sequence length."""
        return 230  # LentIMPRA fixed sequence length
    
    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None, max_samples: Optional[int] = None, specific_indices: Optional[str] = None):
        """Create LentIMPRA dataloader with optional sample limiting."""
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
        
        # Handle dataset subsetting with specific indices or random sampling
        if max_samples is not None and len(dataset) > max_samples:
            import torch.utils.data as data_utils
            
            # Parse specific indices if provided
            selected_indices = []
            if specific_indices:
                try:
                    selected_indices = [int(idx.strip()) for idx in specific_indices.split(',')]
                    # Validate indices are within dataset bounds
                    selected_indices = [idx for idx in selected_indices if 0 <= idx < len(dataset)]
                    print(f"  ↳ Guaranteed selection of indices: {selected_indices}")
                except ValueError:
                    print(f"  ↳ Warning: Invalid specific_indices format '{specific_indices}', ignoring")
                    selected_indices = []
            
            # Fill remaining slots with random indices if needed
            num_selected = len(selected_indices)
            if num_selected < max_samples:
                # Get remaining indices to sample from
                all_indices = set(range(len(dataset)))
                remaining_indices = list(all_indices - set(selected_indices))
                
                if remaining_indices:
                    num_random = min(max_samples - num_selected, len(remaining_indices))
                    random_indices = torch.randperm(len(remaining_indices))[:num_random].tolist()
                    selected_indices.extend([remaining_indices[i] for i in random_indices])
            
            # Create subset with selected indices
            indices = torch.tensor(selected_indices[:max_samples])
            dataset = data_utils.Subset(dataset, indices)
            # Store the indices for matching original data later
            self._dataset_indices = indices
            print(f"  ↳ LentIMPRA dataset limited to {len(dataset)} samples from {split} split")
        else:
            # Full dataset - no indices needed
            self._dataset_indices = None
            
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
        """Get original test data for SP-MSE comparison, matching the limited dataset if applicable."""
        try:
            # Load LentIMPRA test data from H5
            print(f"Loading original test data from: {data_path}")
            with h5py.File(data_path, 'r') as data_file:
                # Load one-hot data: (N, 230, 4)
                X = torch.tensor(np.array(data_file['onehot_test']))
            
            # If we limited the dataset, apply the same indices to original data
            if self._dataset_indices is not None:
                print(f"  ↳ Applying same subset indices to original data ({len(self._dataset_indices)} samples)")
                X = X[self._dataset_indices]
                
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
    
    def evaluate_with_sampling(self, checkpoint_path: str, config: OmegaConf, 
                              oracle_checkpoint: str, data_path: str,
                              split: str = 'test', steps: Optional[int] = None, 
                              batch_size: Optional[int] = None, architecture: str = 'transformer',
                              show_progress: bool = False, save_sequences: bool = False,
                              save_visualization_data: bool = False, viz_output_path: Optional[str] = None,
                              viz_format: str = 'hdf5', max_samples: Optional[int] = None,
                              specific_indices: Optional[str] = None):
        """
        Override base method to handle LentIMPRA-specific visualization and evaluation.
        """
        print(f"Evaluating {self.dataset_name} on {split} split with sampling...")
        
        # Set default steps to sequence length if not provided
        if steps is None:
            steps = self.get_sequence_length(config)
            print(f"Using default steps: {steps} (sequence length)")
        
        # Create dataloader with optional sample limiting
        dataloader = self.create_dataloader(config, split, batch_size, max_samples, specific_indices)
        
        # Load oracle model
        print("Loading oracle model for SP-MSE evaluation...")
        oracle_model = self.load_oracle_model(oracle_checkpoint, data_path)
        
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
            actual_samples = len(dataloader.dataset)
            
            # Convert original samples to token indices for visualization storage
            # Keep original_data in one-hot format for SP-MSE computation
            original_samples_indices = None
            if original_data is not None:
                # Convert from (batch_size, seq_length, 4) to (batch_size, seq_length)
                original_samples_indices = torch.argmax(original_data, dim=2)
            
            viz_logger = create_visualization_logger(
                num_samples=actual_samples,
                sequence_length=sequence_length,
                num_steps=steps,
                dataset_name=self.dataset_name,
                architecture=architecture,
                split=split,
                save_oracle_mse=True,  # Enable oracle MSE for evaluation
                device=self.device,
                original_samples=original_samples_indices,  # Add original samples as token indices
                dataset_indices=self._dataset_indices  # Add dataset indices for reproducibility
            )
            print(f"  ↳ Visualization data logging enabled with oracle MSE and original samples ({actual_samples} samples)")
        
        # Sample sequences using PC sampler
        print(f"Sampling sequences with PC sampler ({steps} steps)...")
        sampled_sequences, target_labels = self.sample_sequences_for_evaluation(
            checkpoint_path, config, dataloader, steps, architecture, show_progress, viz_logger, oracle_model, data_path
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
            'oracle_evaluation': 'completed'
        }
        
        # Save visualization data if requested
        if save_visualization_data and viz_logger is not None:
            if viz_output_path is None:
                # Auto-generate visualization output path
                checkpoint_dir = os.path.dirname(checkpoint_path)
                viz_output_path = os.path.join(checkpoint_dir, f"lentimpra_evaluation_visualization_data.{viz_format}")
            
            viz_logger.save(viz_output_path, viz_format)
            results['visualization_output_path'] = viz_output_path
        
        print(f"SP-MSE: {sp_mse:.6f}")
        
        return results
    
    def get_oracle_predictions_for_viz(self, sequences: torch.Tensor, oracle_model) -> torch.Tensor:
        """
        LentIMPRA-specific oracle predictions for visualization.
        
        Args:
            sequences: One-hot encoded sequences (batch_size, seq_length, 4)
            oracle_model: LentIMPRA oracle model with predict() method
            
        Returns:
            Oracle predictions tensor
        """
        try:
            # Convert from (batch, length, channels) to (batch, channels, length) for LegNet
            sequences_input = sequences.permute(0, 2, 1).to(self.device)
            predictions = oracle_model.predict(sequences_input)
            
            # If predictions is 1D (batch_size,), reshape to (batch_size, 1) for proper MSE computation
            # This ensures oracle_predictions.pow(2).mean(dim=-1) gives per-sample MSE values
            if len(predictions.shape) == 1:
                predictions = predictions.unsqueeze(-1)
                
            return predictions
        except Exception as e:
            print(f"Warning: Could not get oracle predictions for visualization: {e}")
            return torch.zeros(sequences.shape[0], 1, device=self.device)
    


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
    
    # Use the dataset-specific evaluate_with_sampling method
    metrics = evaluator.evaluate_with_sampling(
        checkpoint_path=args.checkpoint,
        config=config,
        oracle_checkpoint=args.oracle_checkpoint,
        data_path=args.data_path,
        split=args.split,
        steps=args.steps,
        batch_size=args.batch_size,
        architecture=args.architecture,
        show_progress=getattr(args, 'show_progress', False),
        save_sequences=args.save_sequences,
        save_visualization_data=getattr(args, 'save_viz_data', False),
        viz_output_path=getattr(args, 'viz_output', None),
        viz_format=getattr(args, 'viz_format', 'hdf5'),
        max_samples=getattr(args, 'max_samples', None),
        specific_indices=getattr(args, 'specific_indices', None)
    )
    
    # Print and save results
    evaluator.print_results(metrics)
    
    output_path = args.output or f"evaluation_results/lentimpra_{args.architecture}_{args.split}_results.json"
    evaluator.save_results(metrics, output_path)
    
    print(f"\n✓ LentIMPRA evaluation completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())