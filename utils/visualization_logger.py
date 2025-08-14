"""
Visualization Data Logger for D3-DNA Discrete Diffusion

This module provides functionality to log intermediate data during the sampling process
for visualization purposes. It captures sequences, score matrices, noise levels, and
optionally oracle MSE predictions at each sampling step.
"""

import os
import h5py
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


class VisualizationDataLogger:
    """
    Logger to capture intermediate diffusion sampling data for visualization.
    
    Captures at each sampling step:
    - Sequences (token indices)
    - Score matrices (model predictions)
    - Noise levels (sigma, dsigma)  
    - Oracle MSE predictions (evaluation only)
    """
    
    def __init__(self, 
                 num_samples: int,
                 sequence_length: int, 
                 num_steps: int,
                 dataset_name: str = "unknown",
                 architecture: str = "transformer",
                 split: Optional[str] = None,
                 save_oracle_mse: bool = False,
                 device: torch.device = None,
                 original_samples: Optional[torch.Tensor] = None,
                 ground_truth_labels: Optional[torch.Tensor] = None,
                 ground_truth_predictions: Optional[torch.Tensor] = None,
                 dataset_indices: Optional[torch.Tensor] = None):
        """
        Initialize the visualization data logger.
        
        Args:
            num_samples: Number of samples being generated
            sequence_length: Length of each sequence
            num_steps: Total number of sampling steps
            dataset_name: Name of the dataset
            architecture: Model architecture type
            split: Dataset split (for evaluation)
            save_oracle_mse: Whether to save oracle MSE predictions
            device: Device to store tensors on
            original_samples: Original samples for MSE comparison (evaluation only)
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.num_steps = num_steps
        self.dataset_name = dataset_name
        self.architecture = architecture
        self.split = split
        self.save_oracle_mse = save_oracle_mse
        self.device = device or torch.device('cpu')
        self.original_samples = original_samples.detach().cpu() if original_samples is not None else None
        
        # Handle ground truth labels with shape normalization
        if ground_truth_labels is not None:
            gt_labels = ground_truth_labels.detach().cpu()
            # Convert (num_seq,) to (num_seq, 1) for consistent processing
            if len(gt_labels.shape) == 1:
                gt_labels = gt_labels.unsqueeze(-1)
            self.ground_truth_labels = gt_labels
        else:
            self.ground_truth_labels = None
            
        # Handle ground truth oracle predictions
        self.ground_truth_predictions = ground_truth_predictions.detach().cpu() if ground_truth_predictions is not None else None
        
        # Initialize storage for step data
        self.step_data = []
        self.metadata = {
            'dataset': dataset_name,
            'num_samples': num_samples,
            'sequence_length': sequence_length,
            'total_steps': num_steps,
            'architecture': architecture,
            'split': split,
            'save_oracle_mse': save_oracle_mse
        }
        
        # Add original samples to metadata if provided
        if self.original_samples is not None:
            self.metadata['original_samples'] = self.original_samples
            
        # Add ground truth data to metadata if provided
        if self.ground_truth_labels is not None:
            self.metadata['ground_truth_labels'] = self.ground_truth_labels
            
        if self.ground_truth_predictions is not None:
            self.metadata['ground_truth_predictions'] = self.ground_truth_predictions
            
        # Add dataset indices to metadata if provided
        if dataset_indices is not None:
            self.dataset_indices = dataset_indices.detach().cpu()
            self.metadata['dataset_indices'] = self.dataset_indices
        else:
            self.dataset_indices = None
        
        print(f"✓ Visualization logger initialized for {num_samples} samples, {num_steps} steps")
        if save_oracle_mse:
            print("  ↳ Oracle MSE logging enabled")
    
    def log_step(self,
                 step: int,
                 timestep: float,
                 sequences: torch.Tensor,
                 score_matrix: torch.Tensor,
                 prob_matrix: torch.Tensor,
                 noise_level: float,
                 noise_rate: Optional[float] = None,
                 oracle_mse: Optional[torch.Tensor] = None,
                 oracle_predictions: Optional[torch.Tensor] = None):
        """
        Log data for a single sampling step.
        
        Args:
            step: Current sampling step (0 to num_steps-1)
            timestep: Current diffusion timestep (1.0 to eps)
            sequences: Current sequences (batch_size, seq_length)
            score_matrix: Score matrix from model (batch_size, seq_length, 4)
            prob_matrix: Probability matrix from staggered score (batch_size, seq_length, 4)
            noise_level: Current noise level (sigma)
            noise_rate: Rate of noise change (dsigma), optional
            oracle_mse: Oracle MSE predictions (batch_size,), optional
            oracle_predictions: Oracle predictions for current sequences (batch_size, num_outputs), optional
        """
        # Convert tensors to CPU and detach for storage
        sequences_cpu = sequences.detach().cpu()
        score_matrix_cpu = score_matrix.detach().cpu()
        
        # Convert to float16 for memory efficiency
        try:
            score_matrix_cpu = score_matrix_cpu.to(torch.float16)
        except (RuntimeError, AttributeError):
            # Fallback if float16 conversion fails
            pass
        
        # Handle prob_matrix
        prob_matrix_cpu = prob_matrix.detach().cpu()
        try:
            prob_matrix_cpu = prob_matrix_cpu.to(torch.float16)
        except (RuntimeError, AttributeError):
            # Fallback if float16 conversion fails
            pass
        
        step_entry = {
            'step': step,
            'timestep': timestep,
            'sequence': sequences_cpu,
            'score_matrix': score_matrix_cpu,
            'prob_matrix': prob_matrix_cpu,
            'noise_level': noise_level,
        }
        
        # Add optional data
        if noise_rate is not None:
            step_entry['noise_rate'] = noise_rate
            
        if oracle_mse is not None and self.save_oracle_mse:
            step_entry['oracle_mse'] = oracle_mse.detach().cpu()
            
        if oracle_predictions is not None and self.save_oracle_mse:
            step_entry['oracle_predictions'] = oracle_predictions.detach().cpu()
        
        self.step_data.append(step_entry)
        
        if step % 50 == 0:  # Print progress every 50 steps
            print(f"  Logged step {step}/{self.num_steps}")
    
    def update_noise_schedule_metadata(self, noise_config: Dict[str, Any]):
        """Update metadata with noise schedule information."""
        self.metadata['noise_schedule'] = noise_config
    
    def save_to_hdf5(self, filepath: str):
        """
        Save collected visualization data to HDF5 file.
        
        Args:
            filepath: Path to save the HDF5 file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Save metadata
            metadata_group = f.create_group('metadata')
            for key, value in self.metadata.items():
                if isinstance(value, dict):
                    # Handle nested dictionaries (like noise_schedule)
                    subgroup = metadata_group.create_group(key)
                    for subkey, subvalue in value.items():
                        if subvalue is not None:
                            subgroup.attrs[subkey] = subvalue
                elif key in ['original_samples', 'ground_truth_labels', 'ground_truth_predictions', 'dataset_indices'] and value is not None:
                    # Save tensor data as dataset in metadata
                    metadata_group.create_dataset(key, data=value.numpy())
                elif value is not None:
                    metadata_group.attrs[key] = value
            
            # Save step data
            steps_group = f.create_group('steps')
            
            for i, step_entry in enumerate(self.step_data):
                step_group = steps_group.create_group(f'step_{step_entry["step"]:04d}')
                
                # Save scalar values as attributes
                step_group.attrs['step'] = step_entry['step']
                step_group.attrs['timestep'] = step_entry['timestep'] 
                step_group.attrs['noise_level'] = step_entry['noise_level']
                
                if 'noise_rate' in step_entry:
                    step_group.attrs['noise_rate'] = step_entry['noise_rate']
                
                # Save tensor data as datasets
                step_group.create_dataset('sequence', data=step_entry['sequence'].numpy())
                
                # Handle score matrix with potential dtype conversion
                score_data = step_entry['score_matrix']
                if hasattr(score_data, 'dtype') and score_data.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    score_data = score_data.to(torch.float16)
                step_group.create_dataset('score_matrix', data=score_data.numpy())
                
                # Handle prob matrix
                prob_data = step_entry['prob_matrix']
                if hasattr(prob_data, 'dtype') and prob_data.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    prob_data = prob_data.to(torch.float16)
                step_group.create_dataset('prob_matrix', data=prob_data.numpy())
                
                # Save oracle data if available
                if 'oracle_mse' in step_entry:
                    step_group.create_dataset('oracle_mse', data=step_entry['oracle_mse'].numpy())
                    
                if 'oracle_predictions' in step_entry:
                    step_group.create_dataset('oracle_predictions', data=step_entry['oracle_predictions'].numpy())
        
        print(f"✓ Visualization data saved to: {filepath}")
        print(f"  ↳ {len(self.step_data)} steps, {self.num_samples} samples")
    
    def save_to_npz(self, filepath: str):
        """
        Save collected visualization data to compressed NPZ file.
        
        Args:
            filepath: Path to save the NPZ file  
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data for NPZ
        save_dict = {}
        
        # Add metadata
        for key, value in self.metadata.items():
            if isinstance(value, dict):
                # Flatten nested dictionaries
                for subkey, subvalue in value.items():
                    save_dict[f'{key}_{subkey}'] = subvalue
            elif key in ['original_samples', 'ground_truth_labels', 'ground_truth_predictions', 'dataset_indices'] and value is not None:
                # Save tensor data directly (not as meta_ prefix)
                save_dict[key] = value.numpy()
            else:
                save_dict[f'meta_{key}'] = value
        
        # Stack step data into arrays
        if self.step_data:
            save_dict['steps'] = np.array([entry['step'] for entry in self.step_data])
            save_dict['timesteps'] = np.array([entry['timestep'] for entry in self.step_data])
            save_dict['noise_levels'] = np.array([entry['noise_level'] for entry in self.step_data])
            
            if 'noise_rate' in self.step_data[0]:
                save_dict['noise_rates'] = np.array([entry['noise_rate'] for entry in self.step_data])
            
            # Stack sequences and score matrices
            sequences = torch.stack([entry['sequence'] for entry in self.step_data], dim=0)
            score_matrices = torch.stack([entry['score_matrix'] for entry in self.step_data], dim=0)
            
            save_dict['sequences'] = sequences.numpy()  # Shape: (num_steps, batch_size, seq_length)
            
            # Handle score matrix dtype conversion for NPZ
            if score_matrices.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                score_matrices = score_matrices.to(torch.float16)
            save_dict['score_matrices'] = score_matrices.numpy()  # Shape: (num_steps, batch_size, seq_length, 4)
            
            # Stack prob matrices
            prob_matrices = torch.stack([entry['prob_matrix'] for entry in self.step_data], dim=0)
            if prob_matrices.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                prob_matrices = prob_matrices.to(torch.float16)
            save_dict['prob_matrices'] = prob_matrices.numpy()  # Shape: (num_steps, batch_size, seq_length, 4)
            
            # Stack oracle data if available
            if self.save_oracle_mse and 'oracle_mse' in self.step_data[0]:
                oracle_mses = torch.stack([entry['oracle_mse'] for entry in self.step_data], dim=0)
                save_dict['oracle_mses'] = oracle_mses.numpy()  # Shape: (num_steps, batch_size)
                
            if self.save_oracle_mse and 'oracle_predictions' in self.step_data[0]:
                oracle_predictions = torch.stack([entry['oracle_predictions'] for entry in self.step_data], dim=0)
                save_dict['oracle_predictions'] = oracle_predictions.numpy()  # Shape: (num_steps, batch_size, num_outputs)
            
            # Original samples already added in metadata section above
        
        np.savez_compressed(filepath, **save_dict)
        
        print(f"✓ Visualization data saved to: {filepath}")
        print(f"  ↳ {len(self.step_data)} steps, {self.num_samples} samples")
    
    def save(self, filepath: str, format: str = 'hdf5'):
        """
        Save visualization data in the specified format.
        
        Args:
            filepath: Path to save the file (extension will be added if missing)
            format: Save format ('hdf5', 'h5', 'npz')
        """
        format = format.lower()
        
        if format in ['hdf5', 'h5']:
            if not filepath.endswith(('.h5', '.hdf5')):
                filepath += '.h5'
            self.save_to_hdf5(filepath)
        elif format == 'npz':
            if not filepath.endswith('.npz'):
                filepath += '.npz'
            self.save_to_npz(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'hdf5', 'h5', or 'npz'")


def create_visualization_logger(num_samples: int,
                               sequence_length: int,
                               num_steps: int,
                               dataset_name: str = "unknown",
                               architecture: str = "transformer", 
                               split: Optional[str] = None,
                               save_oracle_mse: bool = False,
                               device: torch.device = None,
                               original_samples: Optional[torch.Tensor] = None,
                               ground_truth_labels: Optional[torch.Tensor] = None,
                               ground_truth_predictions: Optional[torch.Tensor] = None,
                               dataset_indices: Optional[torch.Tensor] = None) -> VisualizationDataLogger:
    """
    Factory function to create a visualization data logger.
    
    Args:
        num_samples: Number of samples being generated
        sequence_length: Length of each sequence
        num_steps: Total number of sampling steps
        dataset_name: Name of the dataset
        architecture: Model architecture type
        split: Dataset split (for evaluation)
        save_oracle_mse: Whether to save oracle MSE predictions
        device: Device to store tensors on
        original_samples: Original samples for MSE comparison (evaluation only)
        ground_truth_labels: Ground truth labels (target labels) for visualization
        ground_truth_predictions: Ground truth oracle predictions for proper MSE calculation
        dataset_indices: Dataset indices used for sample selection (for reproducibility)
    
    Returns:
        VisualizationDataLogger instance
    """
    return VisualizationDataLogger(
        num_samples=num_samples,
        sequence_length=sequence_length,
        num_steps=num_steps,
        dataset_name=dataset_name,
        architecture=architecture,
        split=split,
        save_oracle_mse=save_oracle_mse,
        device=device,
        original_samples=original_samples,
        ground_truth_labels=ground_truth_labels,
        ground_truth_predictions=ground_truth_predictions,
        dataset_indices=dataset_indices
    )