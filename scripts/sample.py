#!/usr/bin/env python3
"""
Base Sampling Framework for D3-DNA Discrete Diffusion

This module provides the base sampling framework that dataset-specific
sampling scripts should inherit from. It uses the proper PC sampler
and provides common functionality while allowing datasets to implement
their own model and data loading.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import the proper sampling functionality
from scripts import sampling


class BaseSampler:
    """
    Base sampler class that provides common sampling functionality.
    
    Dataset-specific sampling scripts should inherit from this class and
    implement the abstract methods for their specific needs.
    """
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Default token to nucleotide mapping (can be overridden by subclasses)
        self.token_to_nucleotide = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """
        Load model for the dataset. Must be implemented by subclasses.
        Each dataset implements its own simple model loading logic.
        
        Args:
            checkpoint_path: Path to specific checkpoint file
            config: Configuration object
            architecture: Architecture type ('transformer', 'convolutional')
            
        Returns:
            Tuple of (model, graph, noise) needed for sampling
        """
        raise NotImplementedError("Subclasses must implement load_model()")

    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None):
        """
        Create dataloader for sampling. Must be implemented by subclasses.
        
        Args:
            config: Configuration object
            split: Dataset split ('train', 'val', 'test')
            batch_size: Batch size (if None, uses config default)
            
        Returns:
            DataLoader instance
        """
        raise NotImplementedError("Subclasses must implement create_dataloader()")
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        """
        Get sequence length for the dataset. Can be overridden by subclasses.
        
        Args:
            config: Configuration object
            
        Returns:
            Sequence length
        """
        # Try common config locations
        if hasattr(config, 'dataset') and hasattr(config.dataset, 'sequence_length'):
            return config.dataset.sequence_length
        elif hasattr(config, 'model') and hasattr(config.model, 'length'):
            return config.model.length
        elif hasattr(config, 'data') and hasattr(config.data, 'sequence_length'):
            return config.data.sequence_length
        else:
            # Dataset-specific defaults - should be overridden by subclasses
            defaults = {
                'deepstarr': 249,
                'mpra': 200,
                'promoter': 1024
            }
            return defaults.get(self.dataset_name.lower(), 249)
    
    def generate_conditioning_labels(self, num_samples: int, config: OmegaConf) -> torch.Tensor:
        """
        Generate conditioning labels for sampling. Can be overridden by subclasses.
        
        Args:
            num_samples: Number of samples to generate labels for
            config: Configuration object
            
        Returns:
            Conditioning labels tensor
        """
        # Default: random conditioning (can be overridden by subclasses)
        if hasattr(config, 'model') and hasattr(config.model, 'num_classes'):
            num_classes = config.model.num_classes
            return torch.randn(num_samples, num_classes, device=self.device)
        else:
            # Fallback for datasets like DeepSTARR with 2 activities
            return torch.randn(num_samples, 2, device=self.device)
    
    def sample_sequences_with_pc_sampler(self, checkpoint_path: str, config: OmegaConf, 
                                       num_samples: int, steps: int, architecture: str = 'transformer',
                                       conditioning_labels: Optional[torch.Tensor] = None,
                                       viz_logger=None) -> torch.Tensor:
        """
        Sample sequences using the proper PC sampler.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Configuration object
            num_samples: Number of sequences to sample
            steps: Number of sampling steps
            architecture: Architecture type
            conditioning_labels: Optional conditioning labels (if None, generates random)
            viz_logger: Optional visualization data logger
            
        Returns:
            Sampled sequences tensor
        """
        # Load model using dataset-specific method
        model, graph, noise = self.load_model(checkpoint_path, config, architecture)
        model.eval()
        
        sequence_length = self.get_sequence_length(config)
        
        # Generate conditioning labels if not provided
        if conditioning_labels is None:
            conditioning_labels = self.generate_conditioning_labels(num_samples, config)
        
        # Update visualization logger with noise schedule metadata
        if viz_logger is not None:
            noise_config = {
                'type': getattr(config.noise, 'type', 'geometric'),
                'sigma_min': getattr(config.noise, 'sigma_min', 1e-3),
                'sigma_max': getattr(config.noise, 'sigma_max', 1.0)
            }
            viz_logger.update_noise_schedule_metadata(noise_config)
        
        # Create PC sampler with visualization support
        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (num_samples, sequence_length), 'analytic', steps, 
            device=self.device, viz_logger=viz_logger
        )
        
        # Sample sequences
        sampled_sequences = sampling_fn(model, conditioning_labels.to(self.device))
        
        return sampled_sequences
    
    
    def sequences_to_strings(self, sequences: torch.Tensor) -> List[str]:
        """
        Convert token sequences to nucleotide strings.
        
        Args:
            sequences: Token sequences (num_samples, seq_length)
            
        Returns:
            List of nucleotide strings
        """
        sequences_str = []
        for seq in sequences:
            seq_str = ''.join([self.token_to_nucleotide.get(token.item(), 'N') for token in seq])
            sequences_str.append(seq_str)
        return sequences_str
    
    def save_sequences(self, sequences: torch.Tensor, output_path: str, format: str = 'npz'):
        """
        Save generated sequences to file.
        
        Args:
            sequences: Generated sequences
            output_path: Output file path
            format: Output format ('npz', 'fasta', or 'csv')
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format.lower() == 'npz':
            # Save as numpy array (matches original implementation)
            np.savez(output_path, sequences.cpu().numpy())
            print(f"Sequences saved to: {output_path}")
            
        elif format.lower() == 'fasta':
            # Convert to strings and save as FASTA
            sequences_str = self.sequences_to_strings(sequences)
            with open(output_path, 'w') as f:
                for i, seq_str in enumerate(sequences_str):
                    f.write(f">{self.dataset_name}_sequence_{i}\n")
                    f.write(f"{seq_str}\n")
            print(f"Sequences saved to: {output_path}")
            
        elif format.lower() == 'csv':
            # Convert to strings and save as CSV
            sequences_str = self.sequences_to_strings(sequences)
            with open(output_path, 'w') as f:
                f.write("sequence_id,sequence\n")
                for i, seq_str in enumerate(sequences_str):
                    f.write(f"{self.dataset_name}_sequence_{i},{seq_str}\n")
            print(f"Sequences saved to: {output_path}")

        elif format == "h5" or format == "hdf5":
            import h5py
            with h5py.File(output_path, "w") as f:
                # Convert float8 to float16 for HDF5 compatibility
                if sequences.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    print("Converting float8 to float16 for HDF5 compatibility")
                    representations_data = sequences.to(torch.float16).numpy()
                else:
                    representations_data = sequences.numpy()
                f.create_dataset("representations", data=representations_data)
            print(f"Representations saved as HDF5 to: {output_path}")
        elif format == "pt":
            # PyTorch native format - supports float8 natively (if available)
            torch.save(sequences, output_path)
            print(f"Representations saved as PyTorch tensor (dtype: {sequences.dtype}) to: {output_path}")
            
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def sample_and_save(self, checkpoint_path: str, config: OmegaConf, num_samples: int, steps: int,
                       architecture: str = 'transformer', conditioning_labels: Optional[torch.Tensor] = None,
                       output_path: Optional[str] = None, format: str = 'npz', 
                       save_visualization_data: bool = False, viz_output_path: Optional[str] = None,
                       viz_format: str = 'hdf5') -> Dict[str, Any]:
        """
        Main sampling method - just samples and saves (no evaluation).
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Configuration object
            num_samples: Number of sequences to sample
            steps: Number of sampling steps
            architecture: Architecture type
            conditioning_labels: Optional conditioning labels
            output_path: Output file path (optional, auto-generated if None)
            format: Output format ('npz', 'fasta', 'csv')
            save_visualization_data: Whether to save intermediate sampling data
            viz_output_path: Output path for visualization data
            viz_format: Format for visualization data ('hdf5', 'npz')
            
        Returns:
            Dictionary of sampling results
        """
        print(f"Sampling {num_samples} {self.dataset_name} sequences using PC sampler with {steps} steps...")
        
        # Create visualization logger if requested
        viz_logger = None
        if save_visualization_data:
            from utils.visualization_logger import create_visualization_logger
            sequence_length = self.get_sequence_length(config)
            viz_logger = create_visualization_logger(
                num_samples=num_samples,
                sequence_length=sequence_length,
                num_steps=steps,
                dataset_name=self.dataset_name,
                architecture=architecture,
                save_oracle_mse=False,  # No oracle MSE for sampling
                device=self.device
            )
            print("  ↳ Visualization data logging enabled")
        
        # Sample sequences
        sampled_sequences = self.sample_sequences_with_pc_sampler(
            checkpoint_path, config, num_samples, steps, architecture, conditioning_labels, viz_logger
        )
        
        results = {
            'num_samples': sampled_sequences.shape[0],
            'sequence_length': sampled_sequences.shape[1],
            'sampling_steps': steps,
            'dataset': self.dataset_name
        }
        
        # Save sequences
        if output_path is None:
            # Extract directory from checkpoint path for output
            checkpoint_dir = os.path.dirname(checkpoint_path)
            output_path = os.path.join(checkpoint_dir, f"sample.{format}")
        
        self.save_sequences(sampled_sequences, output_path, format)
        results['output_path'] = output_path
        
        # Save visualization data if requested
        if save_visualization_data and viz_logger is not None:
            if viz_output_path is None:
                # Auto-generate visualization output path
                checkpoint_dir = os.path.dirname(checkpoint_path)
                viz_output_path = os.path.join(checkpoint_dir, f"visualization_data.{viz_format}")
            
            viz_logger.save(viz_output_path, viz_format)
            results['visualization_output_path'] = viz_output_path
        
        return results


    def save_representation(self, checkpoint_path: str, config: OmegaConf, 
                              split: str = 'test', save_rep_timestamp: Optional[int] = None, 
                              batch_size: Optional[int] = None, architecture: str = 'transformer',
                              output_path: Optional[str] = None, format: str = 'npz') -> Dict[str, Any]:
        """
        Save the representation of the model at a given timestamp.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Configuration object
            data_path: Path to data file needed by oracle
            split: Dataset split to evaluate on
            save_rep_timestamp: Timestamp to save representation (defaults to 200)
            batch_size: Batch size for evaluation (optional)
            architecture: Architecture type
            output_path: Output file path (optional, auto-generated if None)
            format: Output format ('npz', 'fasta', 'csv')
        Returns:
            Dictionary of evaluation results including SP-MSE
        """
        print(f"Saving representation of the model for {self.dataset_name} on {split} split...")
        
        # Set default steps to sequence length if not provided
        if save_rep_timestamp is None:
            print(f"Using default timestamp: {save_rep_timestamp}")
        
        # Create dataloader
        dataloader = self.create_dataloader(config, split, batch_size)

        # Load model using dataset-specific method  
        model, graph, noise = self.load_model(checkpoint_path, config, architecture)
        model.eval()

        # Get sequence length
        sequence_length = self.get_sequence_length(config)
        sampling_eps=1e-3

        timesteps = torch.linspace(1, sampling_eps, sequence_length + 1, device=self.device)
        t = timesteps[save_rep_timestamp] * torch.ones(batch_size, device=self.device)

        sigma, _ = noise(t)

        all_representations = []
        
        # # Wrap dataloader with tqdm if show_progress is True
        # if show_progress:
        #     dataloader = tqdm(dataloader, desc="Sampling sequences")
        
        for batch_idx, (batch, targets) in enumerate(dataloader):
            current_batch_size = batch.shape[0]
            if current_batch_size != batch_size:
                t = timesteps[save_rep_timestamp] * torch.ones(current_batch_size, device=self.device)
                sigma, _ = noise(t)

            batch = batch.to(self.device)
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])
            targets = None #for unconditional generation
            _, rep = model(perturbed_batch, targets, train=False, sigma=sigma, layer_idx=11)
            rep_cpu = rep.detach().cpu()
            # Convert to float8 for maximum memory efficiency
            try:
                all_representations.append(rep_cpu.to(torch.float16))
            except (RuntimeError, AttributeError):
                # Fallback to float16 if float8 not supported
                print("Warning: Float8 not supported, falling back to float16")
                all_representations.append(rep_cpu.to(torch.float16))
                        
            del rep
            torch.cuda.empty_cache()
        
        all_representations = torch.cat(all_representations, dim=0)
        print(all_representations.shape)
        
        results = {
            'dataset': self.dataset_name,
            'split': split,
            'sequence_length': sequence_length,
            'save_rep_timestamp': save_rep_timestamp,
            'batch_size': batch_size,
            'architecture': architecture
        }

        # Save representation
        if output_path is None:
            # Extract directory from checkpoint path for output
            checkpoint_dir = os.path.dirname(checkpoint_path)
            output_path = os.path.join(checkpoint_dir, f"rep_{save_rep_timestamp}_{split}.{format}")
        # if format == "h5" or format == "hdf5":
        #     import h5py
        #     with h5py.File(output_path, "w") as f:
        #         # Convert float8 to float16 for HDF5 compatibility
        #         if all_representations.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        #             print("Converting float8 to float16 for HDF5 compatibility")
        #             representations_data = all_representations.to(torch.float16).numpy()
        #         else:
        #             representations_data = all_representations.numpy()
        #         f.create_dataset("representations", data=representations_data)
        #     print(f"Representations saved as HDF5 to: {output_path}")
        # elif format == "pt":
        #     # PyTorch native format - supports float8 natively (if available)
        #     torch.save(all_representations, output_path)
        #     print(f"Representations saved as PyTorch tensor (dtype: {all_representations.dtype}) to: {output_path}")
        # else:
        self.save_sequences(all_representations, output_path, format)
        results['output_path'] = output_path
        
        print(f"Representation saved to: {output_path}")
        
        return results


def parse_base_args():
    """Parse common command line arguments for sampling scripts."""
    parser = argparse.ArgumentParser(description='D3 Sampling Script')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint file')
    parser.add_argument('--architecture', required=True, choices=['transformer', 'convolutional'], help='Model architecture')
    parser.add_argument('--save_rep', required=False, help='Saving representation of the model')
    parser.add_argument('--save_rep_timestamp', type=int, default=200, help='Saving representation at this timestamp')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test', help='Dataset split to save representation on')
    parser.add_argument('--data_path', required=False, help='Path to data file (required for representation)')
    parser.add_argument('--config', help='Path to config file (optional, dataset may provide default)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--steps', type=int, help='Number of sampling steps (defaults to sequence length)')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for sampling')
    parser.add_argument('--format', choices=['npz', 'fasta', 'csv', 'h5', 'hdf5', 'pt'], default='h5', help='Output format')
    
    # Visualization data arguments
    parser.add_argument('--save_viz_data', action='store_true', help='Save visualization data (sequences, scores, noise levels)')
    parser.add_argument('--viz_output', help='Output path for visualization data')
    parser.add_argument('--viz_format', choices=['hdf5', 'h5', 'npz'], default='hdf5', help='Visualization data format')
    
    return parser


def main_sample(sampler: BaseSampler, args):
    """
    Common main sampling function that can be used by dataset-specific scripts.
    
    Args:
        sampler: Dataset-specific sampler instance
        args: Parsed command line arguments
    """
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1
    
    # Load configuration (required)
    if not args.config:
        print("Error: Config file is required. Please provide --config path/to/config.yaml")
        return 1
    
    config = OmegaConf.load(args.config)

    if args.save_rep:
        if not args.data_path:
            print("Error: --data_path is required for saving representation")
            return 1
        else:
            print(f"Saving representation of the model to {args.data_path}")
            results = sampler.save_representation(
                checkpoint_path=args.checkpoint,
                config=config,
                split=args.split,
                save_rep_timestamp=args.save_rep_timestamp,
                batch_size=args.batch_size,
                architecture=args.architecture,
                output_path=args.output,
                format=args.format
            )

            # Print results
            print(f"\n{sampler.dataset_name} Representation Results:")
            print("=" * 40)
            for key, value in results.items():
                print(f"{key}: {value}")
            
            print(f"\n✓ {sampler.dataset_name} saving representation completed successfully!")
    
    # Set default steps to sequence length if not provided
    steps = args.steps
    if steps is None:
        steps = sampler.get_sequence_length(config)
        print(f"Using default steps: {steps} (sequence length)")
    
    # Run sampling only (no evaluation)
    results = sampler.sample_and_save(
        checkpoint_path=args.checkpoint,
        config=config,
        num_samples=args.num_samples,
        steps=steps,
        architecture=args.architecture,
        output_path=args.output,
        format=args.format,
        save_visualization_data=args.save_viz_data,
        viz_output_path=args.viz_output,
        viz_format=args.viz_format
    )
    
    # Print results
    print(f"\n{sampler.dataset_name} Sampling Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")
    
    print(f"\n✓ {sampler.dataset_name} sampling completed successfully!")
    return 0