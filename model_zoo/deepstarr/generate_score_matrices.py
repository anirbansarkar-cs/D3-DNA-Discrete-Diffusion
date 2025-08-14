#!/usr/bin/env python3
"""
DeepSTARR Score Matrix Generation Script

This script generates score matrices for DeepSTARR test sequences using D3 diffusion models.
It processes sequences at different noise levels and saves the results in HDF5 format
following the visualization data format specification.

Usage:
    python model_zoo/deepstarr/generate_score_matrices.py \
        --checkpoint path/to/checkpoint.ckpt \
        --config model_zoo/deepstarr/configs/transformer.yaml \
        --data_path model_zoo/deepstarr/DeepSTARR_data.h5 \
        --output deepstarr_score_matrices.h5
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import h5py
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import D3 components
from model_zoo.deepstarr.models import load_trained_model
from model_zoo.deepstarr.data import get_deepstarr_datasets
from utils.utils import get_score_fn


class DeepSTARRScoreMatrixGenerator:
    """DeepSTARR Score Matrix Generator using D3 diffusion models."""
    
    def __init__(self, checkpoint_path: str, config_path: str, device: str = 'cuda'):
        """
        Initialize the score matrix generator.
        
        Args:
            checkpoint_path: Path to trained D3 model checkpoint
            config_path: Path to model configuration file
            device: Device for computation ('cuda' or 'cpu')
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {self.device}")
        
        # Load configuration
        self.config = OmegaConf.load(config_path)
        
        # Initialize components
        self.model = None
        self.graph = None
        self.noise = None
        
        # Data storage
        self.sequences = None
        self.labels = None
        self.dataset_indices = None
        
    def load_model(self, architecture: str = 'transformer'):
        """Load the trained D3 model and associated components."""
        print(f"Loading D3 model from {self.checkpoint_path}")
        self.model, self.graph, self.noise = load_trained_model(
            self.checkpoint_path, self.config, architecture, self.device
        )
        self.model.eval()
        print("✓ Model loaded successfully")
        
    def load_data(self, data_path: str, split: str = 'test', max_samples: Optional[int] = None, 
                  specific_indices: Optional[str] = None):
        """
        Load DeepSTARR sequence data.
        
        Args:
            data_path: Path to DeepSTARR H5 file
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum number of samples to load
            specific_indices: Comma-separated string of specific indices to guarantee
        """
        print(f"Loading DeepSTARR data from {data_path}")
        
        # Get datasets
        train_ds, val_ds, test_ds = get_deepstarr_datasets(data_path)
        
        # Select appropriate dataset
        if split == 'train':
            dataset = train_ds
        elif split == 'val':
            dataset = val_ds
        elif split == 'test':
            dataset = test_ds
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Handle dataset subsetting with specific indices or random sampling
        selected_indices = None
        if max_samples is not None and len(dataset) > max_samples:
            # Parse specific indices if provided
            guaranteed_indices = []
            if specific_indices:
                try:
                    guaranteed_indices = [int(idx.strip()) for idx in specific_indices.split(',')]
                    # Validate indices are within dataset bounds
                    guaranteed_indices = [idx for idx in guaranteed_indices if 0 <= idx < len(dataset)]
                    print(f"  ↳ Guaranteed selection of indices: {guaranteed_indices}")
                except ValueError:
                    print(f"  ↳ Warning: Invalid specific_indices format '{specific_indices}', ignoring")
                    guaranteed_indices = []
            
            # Fill remaining slots with random indices if needed
            num_guaranteed = len(guaranteed_indices)
            if num_guaranteed < max_samples:
                # Get remaining indices to sample from
                all_indices = set(range(len(dataset)))
                remaining_indices = list(all_indices - set(guaranteed_indices))
                
                if remaining_indices:
                    num_random = min(max_samples - num_guaranteed, len(remaining_indices))
                    random_indices = torch.randperm(len(remaining_indices))[:num_random].tolist()
                    guaranteed_indices.extend([remaining_indices[i] for i in random_indices])
            
            # Create selected indices
            selected_indices = guaranteed_indices[:max_samples]
            print(f"  ↳ DeepSTARR dataset limited to {len(selected_indices)} samples from {split} split")
        
        # Load sequences and labels
        if selected_indices is not None:
            self.sequences = torch.stack([dataset[i][0] for i in selected_indices])
            self.labels = torch.stack([dataset[i][1] for i in selected_indices])
            self.dataset_indices = torch.tensor(selected_indices, dtype=torch.long)
        else:
            # Load all data
            self.sequences = torch.stack([dataset[i][0] for i in range(len(dataset))])
            self.labels = torch.stack([dataset[i][1] for i in range(len(dataset))])
            self.dataset_indices = None
            
        print(f"✓ Loaded {len(self.sequences)} sequences")
        print(f"  - Sequence shape: {self.sequences.shape}")
        print(f"  - Labels shape: {self.labels.shape}")
        
    def generate_noise_schedule(self, num_steps: int, eps: float = 1e-5) -> Tuple[torch.Tensor, int]:
        """
        Generate noise schedule for sampling.
        
        Args:
            num_steps: Number of noise steps
            eps: Minimum noise level
            
        Returns:
            Tuple of (sigma_values, default_sigma_index)
        """
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        sigma_values = []
        
        for t in timesteps:
            sigma, _ = self.noise(t.unsqueeze(0))
            sigma_values.append(sigma.item())
            
        sigma_values = torch.tensor(sigma_values, device=self.device)
        
        # Use 5th to last sigma as default (or last if less than 5 steps)
        default_idx = max(0, len(sigma_values) - 5)
        
        return sigma_values, default_idx
        
    def generate_score_matrices(self, sigma_values: torch.Tensor, default_sigma_idx: int, 
                               batch_size: int, save_intermediates: bool = False) -> Dict[str, Any]:
        """
        Generate score matrices for sequences at different noise levels.
        
        Args:
            sigma_values: Noise schedule values
            default_sigma_idx: Index of default sigma to use
            batch_size: Batch size for processing
            save_intermediates: Whether to save results for all noise steps
            
        Returns:
            Dictionary containing score matrix results
        """
        print("Generating score matrices...")
        
        n_sequences = len(self.sequences)
        sampling_score_fn = get_score_fn(self.model, train=False, sampling=True)
        
        results = {
            'default_step': {},
            'all_steps': {} if save_intermediates else None
        }
        
        # Determine which steps to process
        if save_intermediates:
            step_indices = list(range(len(sigma_values)))
        else:
            step_indices = [default_sigma_idx]
        
        for step_counter, step_idx in enumerate(tqdm(step_indices, desc="Processing noise steps")):
            sigma = sigma_values[step_idx]
            
            # Calculate timestep (inverse of sigma generation)
            timestep = 1.0 - (step_idx / (len(sigma_values) - 1)) * (1.0 - 1e-5) if len(sigma_values) > 1 else 1.0
            
            step_results = {
                'step': step_idx,
                'timestep': timestep,
                'noise_level': sigma.item(),
                'sequences': [],
                'score_matrices': []
            }
            
            # Process sequences in batches
            for batch_start in range(0, n_sequences, batch_size):
                batch_end = min(batch_start + batch_size, n_sequences)
                
                # Get batch sequences
                seq_batch = self.sequences[batch_start:batch_end].to(self.device)
                label_batch = self.labels[batch_start:batch_end].to(self.device)
                
                # Create sigma tensor for batch
                batch_sigma = sigma.repeat(seq_batch.shape[0]).to(self.device)
                
                # Generate score matrices
                with torch.no_grad():
                    score_matrix = sampling_score_fn(seq_batch, batch_sigma, label_batch)  # (batch, seq_len, 4)
                    
                step_results['sequences'].append(seq_batch.cpu())
                step_results['score_matrices'].append(score_matrix.cpu().to(torch.float16))
            
            # Concatenate batch results
            step_results['sequences'] = torch.cat(step_results['sequences'], dim=0)
            step_results['score_matrices'] = torch.cat(step_results['score_matrices'], dim=0)
            
            # Store results
            if step_idx == default_sigma_idx:
                results['default_step'] = step_results
            if save_intermediates:
                results['all_steps'][f'step_{step_counter:04d}'] = step_results
        
        return results
    
    def save_to_hdf5(self, output_path: str, results: Dict[str, Any], sigma_values: torch.Tensor, 
                     default_sigma_idx: int, architecture: str, split: str):
        """
        Save results to HDF5 file using visualization data format.
        
        Args:
            output_path: Path to output H5 file
            results: Score matrix results
            sigma_values: Noise schedule values
            default_sigma_idx: Default sigma index
            architecture: Model architecture
            split: Dataset split used
        """
        print(f"Saving results to {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Metadata group
            metadata_group = f.create_group('metadata')
            metadata_group.attrs['dataset'] = 'deepstarr'
            metadata_group.attrs['num_samples'] = len(self.sequences)
            metadata_group.attrs['sequence_length'] = self.sequences.shape[1]
            metadata_group.attrs['total_steps'] = len(sigma_values)
            metadata_group.attrs['architecture'] = architecture
            metadata_group.attrs['split'] = split
            metadata_group.attrs['save_oracle_mse'] = False  # Not applicable for score matrix generation
            
            # Add original samples to metadata (sequences as token indices)
            metadata_group.create_dataset('original_samples', data=self.sequences.numpy())
            
            # Add dataset indices to metadata if available
            if self.dataset_indices is not None:
                metadata_group.create_dataset('dataset_indices', data=self.dataset_indices.numpy())
            
            # Additional metadata
            metadata_group.attrs['checkpoint_path'] = self.checkpoint_path
            metadata_group.attrs['config_path'] = self.config_path
            metadata_group.attrs['timestamp'] = datetime.now().isoformat()
            metadata_group.attrs['default_sigma_idx'] = default_sigma_idx
            
            # Noise schedule metadata
            noise_group = metadata_group.create_group('noise_schedule')
            noise_group.attrs['type'] = 'geometric'
            noise_group.attrs['sigma_min'] = self.config.noise.sigma_min
            noise_group.attrs['sigma_max'] = self.config.noise.sigma_max
            noise_group.create_dataset('sigma_values', data=sigma_values.cpu().numpy())
            
            # Steps group
            steps_group = f.create_group('steps')
            
            # Save default step
            default_data = results['default_step']
            step_name = f"step_{default_data['step']:04d}"
            step_group = steps_group.create_group(step_name)
            
            step_group.attrs['step'] = default_data['step']
            step_group.attrs['timestep'] = default_data['timestep']
            step_group.attrs['noise_level'] = default_data['noise_level']
            step_group.create_dataset('sequence', data=default_data['sequences'].numpy())
            step_group.create_dataset('score_matrix', data=default_data['score_matrices'].numpy())
            
            # Save all steps if available
            if results['all_steps'] is not None:
                for step_key, step_data in results['all_steps'].items():
                    if step_key != step_name:  # Avoid duplicating default step
                        step_group = steps_group.create_group(step_key)
                        
                        step_group.attrs['step'] = step_data['step']
                        step_group.attrs['timestep'] = step_data['timestep']
                        step_group.attrs['noise_level'] = step_data['noise_level']
                        step_group.create_dataset('sequence', data=step_data['sequences'].numpy())
                        step_group.create_dataset('score_matrix', data=step_data['score_matrices'].numpy())
        
        print(f"✓ Results saved to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate DeepSTARR Score Matrices with D3')
    
    # Required arguments
    parser.add_argument('--checkpoint', required=True, help='Path to D3 model checkpoint')
    parser.add_argument('--config', required=True, help='Path to model configuration file')
    parser.add_argument('--data_path', required=True, help='Path to DeepSTARR H5 data file')
    
    # Optional arguments
    parser.add_argument('--architecture', choices=['transformer', 'convolutional'], default='transformer',
                       help='Model architecture to use')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test',
                       help='Dataset split to process')
    parser.add_argument('--steps', type=int, default=249,
                       help='Number of noise steps (default: sequence length)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (default: all)')
    parser.add_argument('--specific_indices', type=str, default=None,
                       help='Comma-separated specific indices to guarantee')
    parser.add_argument('--save_intermediates', action='store_true',
                       help='Save results for all noise steps')
    parser.add_argument('--output', default='deepstarr_score_matrices.h5',
                       help='Output HDF5 file path')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                       help='Device for computation')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Validate inputs
    for path in [args.checkpoint, args.config, args.data_path]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            return 1
    
    # Initialize generator
    print("Initializing DeepSTARR Score Matrix Generator...")
    generator = DeepSTARRScoreMatrixGenerator(args.checkpoint, args.config, args.device)
    
    # Load model and data
    generator.load_model(args.architecture)
    generator.load_data(args.data_path, args.split, args.max_samples, args.specific_indices)
    
    print(f"\n🚀 Starting score matrix generation...")
    print(f"   Architecture: {args.architecture}")
    print(f"   Split: {args.split}")
    print(f"   Steps: {args.steps}")
    print(f"   Save intermediates: {args.save_intermediates}")
    print(f"   Output: {args.output}")
    
    # Generate noise schedule
    print("\n📊 Generating noise schedule...")
    sigma_values, default_sigma_idx = generator.generate_noise_schedule(args.steps)
    print(f"Generated {len(sigma_values)} noise levels, using step {default_sigma_idx} as default (σ={sigma_values[default_sigma_idx]:.4f})")
    
    # Generate score matrices
    print(f"\n📈 Generating score matrices...")
    results = generator.generate_score_matrices(
        sigma_values, default_sigma_idx, args.batch_size, args.save_intermediates
    )
    print(f"✓ Score matrix generation completed")
    
    # Save results to H5 file
    print(f"\n💾 Saving results...")
    generator.save_to_hdf5(
        args.output, results, sigma_values, default_sigma_idx, args.architecture, args.split
    )
    
    print(f"\n✅ DeepSTARR Score Matrix Generation completed successfully!")
    print(f"   Results saved to: {args.output}")
    
    # Print final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Processed {len(generator.sequences)} sequences")
    print(f"Architecture: {args.architecture}")
    print(f"Split: {args.split}")
    print(f"Noise steps: {args.steps}")
    if args.save_intermediates:
        print(f"Saved results for all {len(sigma_values)} noise steps")
    else:
        print(f"Saved results for default step {default_sigma_idx} (σ={sigma_values[default_sigma_idx]:.4f})")
    print(f"Results file: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())