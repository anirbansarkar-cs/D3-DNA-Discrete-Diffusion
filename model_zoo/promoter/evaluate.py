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
        self._dataset_indices = None  # Store indices for matching original data
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """Load Promoter model using dataset-specific model loading."""
        from model_zoo.promoter.models import load_trained_model
        
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        """Get Promoter sequence length."""
        if hasattr(config, 'model') and hasattr(config.model, 'length'):
            return config.model.length
        return 1024  # Promoter default sequence length
    
    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None, max_samples: Optional[int] = None):
        """Create Promoter dataloader with optional sample limiting."""
        # Load datasets 
        train_ds, val_ds, test_ds = get_promoter_datasets(config.paths.data_file)
        
        # Select appropriate dataset
        if split == 'train':
            dataset = train_ds
        elif split == 'val':
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
            print(f"  ↳ Promoter dataset limited to {len(dataset)} samples from {split} split")
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
    
    def get_original_test_data(self, data_path: str) -> torch.Tensor:
        """Get original test data for SP-MSE comparison, matching the limited dataset if applicable."""
        try:
            # Load Promoter test data  
            print(f"Loading original test data from: {data_path}")
            train_ds, val_ds, test_ds = get_promoter_datasets(data_path)
            
            # Use test dataset for comparison
            # Create a dataloader to get all test data
            full_dataloader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
            batch = next(iter(full_dataloader))
            
            if len(batch) == 2:
                sequences, _ = batch
            else:
                sequences = batch
            
            # If we limited the dataset, apply the same indices to original data
            if self._dataset_indices is not None:
                print(f"  ↳ Applying same subset indices to original data ({len(self._dataset_indices)} samples)")
                sequences = sequences[self._dataset_indices]
                
            return sequences
                
        except Exception as e:
            print(f"Error loading original test data: {e}")
            # Return dummy data as fallback
            return torch.zeros(100, 1024, 4)  # Promoter one-hot encoded sequences
    
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
            seq_one_hot: One-hot encoded sequences (batch_size, seq_length, 4) or token indices (batch_size, seq_length)
            oracle_model: SEI oracle model
            
        Returns:
            H3K4me3 predictions (batch_size,)
        """
        # Convert to one-hot if needed
        if seq_one_hot.dim() == 2:  # Token indices (batch_size, seq_length)
            import torch.nn.functional as F
            seq_one_hot = F.one_hot(seq_one_hot.long(), num_classes=4).float()
        
        B, L, K = seq_one_hot.shape
        seq_one_hot = seq_one_hot.cpu()
        
        # Process in batches to avoid OOM
        batch_size = 256  # Adjust based on available memory
        all_predictions = []
        
        from tqdm import tqdm
        for i in tqdm(range(0, B, batch_size), desc="Processing SEI batches"):
            end_idx = min(i + batch_size, B)
            batch_seq = seq_one_hot[i:end_idx]
            batch_B = batch_seq.shape[0]
            
            # Pad sequence to 4096 length as expected by SEI
            # Add 1536 bases on each side with uniform background (0.25 for each nucleotide)
            sei_inp = torch.cat([
                torch.ones((batch_B, 4, 1536)) * 0.25,
                batch_seq.transpose(1, 2),  # Convert to (batch, channels, length)
                torch.ones((batch_B, 4, 1536)) * 0.25
            ], 2).to(self.device)  # batch_B x 4 x 4,096
            
            # Get SEI predictions for this batch
            with torch.no_grad():
                sei_out = oracle_model(sei_inp).cpu().detach().numpy()  # batch_B x 21,907
            
            # Filter for H3K4me3 features if SEI features are available
            if self.sei_features is not None:
                h3k4me3_mask = self.sei_features[1].str.strip().values == 'H3K4me3'
                sei_out = sei_out[:, h3k4me3_mask]  # batch_B x 2,350 (H3K4me3 features)
            
            # Take mean across H3K4me3 features for this batch
            batch_pred = sei_out.mean(axis=1)  # batch_B
            all_predictions.append(batch_pred)
        
        # Concatenate all batch predictions
        import numpy as np
        predh3k4me3 = np.concatenate(all_predictions, axis=0)  # B
        
        return predh3k4me3
    
    def evaluate_with_sampling(self, checkpoint_path: str, config: OmegaConf, 
                              oracle_checkpoint: str, data_path: str,
                              split: str = 'test', steps: Optional[int] = None, 
                              batch_size: Optional[int] = None, architecture: str = 'transformer',
                              show_progress: bool = False, save_sequences: bool = False,
                              save_visualization_data: bool = False, viz_output_path: Optional[str] = None,
                              viz_format: str = 'hdf5', max_samples: Optional[int] = None):
        """
        Override base method to handle Promoter-specific visualization and evaluation.
        """
        print(f"Evaluating {self.dataset_name} on {split} split with sampling...")
        
        # Set default steps to sequence length if not provided
        if steps is None:
            steps = self.get_sequence_length(config)
            print(f"Using default steps: {steps} (sequence length)")
        
        # Create dataloader with optional sample limiting
        dataloader = self.create_dataloader(config, split, batch_size, max_samples)
        
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
                original_samples_indices = torch.argmax(original_data, dim=-1)
            
            viz_logger = create_visualization_logger(
                num_samples=actual_samples,
                sequence_length=sequence_length,
                num_steps=steps,
                dataset_name=self.dataset_name,
                architecture=architecture,
                split=split,
                save_oracle_mse=True,  # Enable oracle MSE for evaluation
                device=self.device,
                original_samples=original_samples_indices  # Add original samples as token indices
            )
            print(f"  ↳ Visualization data logging enabled with oracle MSE and original samples ({actual_samples} samples)")
        
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
            'oracle_evaluation': 'completed'
        }
        
        # Save visualization data if requested
        if save_visualization_data and viz_logger is not None:
            if viz_output_path is None:
                # Auto-generate visualization output path
                checkpoint_dir = os.path.dirname(checkpoint_path)
                viz_output_path = os.path.join(checkpoint_dir, f"promoter_evaluation_visualization_data.{viz_format}")
            
            viz_logger.save(viz_output_path, viz_format)
            results['visualization_output_path'] = viz_output_path
        
        print(f"SP-MSE: {sp_mse:.6f}")
        
        return results
    
    def get_oracle_predictions_for_viz(self, sequences: torch.Tensor, oracle_model) -> torch.Tensor:
        """
        Promoter-specific oracle predictions for visualization using SEI model.
        
        Args:
            sequences: One-hot encoded sequences (batch_size, seq_length, 4)
            oracle_model: SEI oracle model
            
        Returns:
            Oracle H3K4me3 predictions tensor
        """
        # Use the same SEI profile method as in compute_sp_mse
        try:
            predictions = self._get_sei_profile(sequences, oracle_model)
            return torch.tensor(predictions, device=self.device)
        except Exception as e:
            print(f"Warning: Could not get oracle predictions for visualization: {e}")
            return torch.zeros(sequences.shape[0], device=self.device)


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
        show_progress=getattr(args, 'show_progress', False),
        save_sequences=args.save_sequences,
        save_visualization_data=getattr(args, 'save_viz_data', False),
        viz_output_path=getattr(args, 'viz_output', None),
        viz_format=getattr(args, 'viz_format', 'hdf5'),
        max_samples=getattr(args, 'max_samples', None)
    )
    
    # Print and save results
    evaluator.print_results(metrics)
    
    output_path = args.output or f"evaluation_results/promoter_{args.architecture}_{args.split}_results.json"
    evaluator.save_results(metrics, output_path)
    
    print(f"\n✓ Promoter evaluation completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())