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
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import h5py

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

        # Wandb logging state
        self.wandb_run = None
        self.wandb_enabled = False

    def setup_wandb(self, args, config: Optional[OmegaConf] = None):
        """
        Initialize wandb logging for sampling if enabled.

        Args:
            args: Parsed command line arguments containing wandb settings
            config: Optional config object to log as wandb.config

        Returns:
            None (sets self.wandb_run and self.wandb_enabled)
        """
        if not args.use_wandb:
            self.wandb_enabled = False
            return

        try:
            import wandb
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb")
            print("Continuing without wandb logging...")
            self.wandb_enabled = False
            return

        # Set default project name if not provided
        project = args.wandb_project or f"{self.dataset_name.lower()}-sampling"

        # Set default run name if not provided
        if args.wandb_name:
            name = args.wandb_name
        else:
            # Auto-generate name: {architecture}_{steps}steps_{num_samples}samples
            name = f"{args.architecture}_{args.steps}steps_{args.num_samples}samples"

        # Prepare config dict
        config_dict = {
            'dataset': self.dataset_name,
            'architecture': args.architecture,
            'num_samples': args.num_samples,
            'steps': args.steps,
            'checkpoint': args.checkpoint,
            'batch_size': args.batch_size,
            'format': args.format,
            'sequence_encoding': args.sequence_encoding,
            'save_elements': args.save_elements,
            'start_at_timestep': args.start_at_timestep,
        }

        # Add config object if provided
        if config is not None:
            config_dict['model_config'] = OmegaConf.to_container(config, resolve=True)

        # Initialize wandb
        try:
            self.wandb_run = wandb.init(
                project=project,
                name=name,
                entity=args.wandb_entity,
                tags=args.wandb_tags,
                config=config_dict,
                job_type='sampling'
            )
            self.wandb_enabled = True
            print(f"Wandb logging enabled: {project}/{name}")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            print("Continuing without wandb logging...")
            self.wandb_enabled = False

    def log_to_wandb(self, sequences: Optional[torch.Tensor] = None,
                     conditioning_labels: Optional[torch.Tensor] = None,
                     saved_elements: Optional[Dict[str, torch.Tensor]] = None,
                     representations: Optional[torch.Tensor] = None):
        """
        Log all sampling results to wandb at end of sampling.

        Args:
            sequences: Generated sequences (N, L) as indices
            conditioning_labels: Conditioning labels used (N, signal_dim)
            saved_elements: Dict of saved elements from PC sampler (N, L, T, 4)
            representations: Hidden representations if saved (N, hidden_dim)

        Returns:
            None (logs to wandb)
        """
        if not self.wandb_enabled or self.wandb_run is None:
            return

        import wandb
        import tempfile

        print("Logging results to wandb...")

        # 1. Log sequences as wandb.Table
        if sequences is not None:
            sequences_str = self.sequences_to_strings(sequences)

            # Build table columns: [id, sequence, label_1, label_2, ...]
            columns = ['id', 'sequence']
            data = []

            # Add conditioning label columns if available
            num_signals = 0
            if conditioning_labels is not None:
                num_signals = conditioning_labels.shape[1]
                for i in range(num_signals):
                    columns.append(f'activity_label_{i}')

            # Build table rows
            for i, seq_str in enumerate(sequences_str):
                row = [i, seq_str]
                if conditioning_labels is not None:
                    # Add label values for this sequence
                    for j in range(num_signals):
                        row.append(conditioning_labels[i, j].item())
                data.append(row)

            sequences_table = wandb.Table(columns=columns, data=data)
            wandb.log({'sequences': sequences_table})
            print(f"  Logged {len(sequences_str)} sequences to wandb.Table")

            # 2. Log conditioning labels summary if available
            if conditioning_labels is not None:
                # Log as histogram for each signal dimension
                for i in range(conditioning_labels.shape[1]):
                    wandb.log({
                        f'activity_label_{i}_distribution': wandb.Histogram(
                            conditioning_labels[:, i].cpu().numpy()
                        )
                    })
                print(f"  Logged activity label distributions ({conditioning_labels.shape[1]} signals)")

            # 3. Log sequence statistics
            self._log_sequence_statistics(sequences)

        # 4. Log saved_elements as artifacts
        if saved_elements:
            for elem_name, elem_tensor in saved_elements.items():
                # Create artifact for this element
                artifact = wandb.Artifact(
                    name=f'{elem_name}_{wandb.run.id}',
                    type=f'sampling_{elem_name}',
                    description=f'{elem_name} from PC sampling (shape: {elem_tensor.shape})'
                )

                # Save to temporary file and add to artifact
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.npy', delete=False) as f:
                    np.save(f.name, elem_tensor.numpy())
                    artifact.add_file(f.name, name=f'{elem_name}.npy')
                    temp_path = f.name

                # Log artifact
                wandb.log_artifact(artifact)

                # Clean up temp file
                os.remove(temp_path)

                print(f"  Logged {elem_name} as artifact (shape: {elem_tensor.shape})")

        # 5. Log representations as artifact if available
        if representations is not None:
            artifact = wandb.Artifact(
                name=f'representations_{wandb.run.id}',
                type='representations',
                description=f'Hidden representations (shape: {representations.shape})'
            )

            # Save to temporary file and add to artifact
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.npy', delete=False) as f:
                # Handle float8 conversion for HDF5 compatibility
                if representations.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    reps_numpy = representations.to(torch.float16).cpu().numpy()
                else:
                    reps_numpy = representations.cpu().numpy()
                np.save(f.name, reps_numpy)
                artifact.add_file(f.name, name='representations.npy')
                temp_path = f.name

            wandb.log_artifact(artifact)
            os.remove(temp_path)

            print(f"  Logged representations as artifact (shape: {representations.shape})")

        print("Wandb logging completed!")

    def _log_sequence_statistics(self, sequences: torch.Tensor):
        """
        Calculate and log sequence statistics (GC content, nucleotide distribution).

        Args:
            sequences: Generated sequences (N, L) as indices
        """
        if not self.wandb_enabled:
            return

        import wandb

        # Convert to numpy for easier processing
        seqs_np = sequences.cpu().numpy()

        # Calculate nucleotide counts (0=A, 1=C, 2=G, 3=T)
        nucleotide_counts = {}
        for nuc_idx, nuc_name in self.token_to_nucleotide.items():
            count = (seqs_np == nuc_idx).sum()
            nucleotide_counts[f'nucleotide_{nuc_name}'] = int(count)

        # Calculate GC content
        total_bases = seqs_np.size
        gc_count = nucleotide_counts.get('nucleotide_G', 0) + nucleotide_counts.get('nucleotide_C', 0)
        gc_content = gc_count / total_bases if total_bases > 0 else 0

        # Log statistics
        wandb.log({
            'gc_content': gc_content,
            **nucleotide_counts,
            'total_sequences': len(sequences),
            'sequence_length': sequences.shape[1]
        })

        print(f"  Logged sequence statistics (GC content: {gc_content:.3f})")

    def cleanup_wandb(self):
        """Finish wandb run and cleanup."""
        if self.wandb_enabled and self.wandb_run is not None:
            import wandb
            wandb.finish()
            print("Wandb run finished")
            self.wandb_run = None
            self.wandb_enabled = False

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
                                       sampling_batch_size: Optional[int] = None,
                                       save_elements_list: Optional[list] = None,
                                       initial_x: Optional[torch.Tensor] = None,
                                       start_at_timestep: int = 0,
                                       proj_fun=None) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Sample sequences using the proper PC sampler with optional batching.

        Args:
            checkpoint_path: Path to checkpoint file
            config: Configuration object
            num_samples: Number of sequences to sample
            steps: Number of sampling steps
            architecture: Architecture type
            conditioning_labels: Optional conditioning labels (if None, generates random)
            sampling_batch_size: Batch size for sampling (None = smart default: 256 for >512 samples)
            save_elements_list: List of elements to save during sampling ('sequence', 'score', etc.)
            initial_x: Optional initial condition sequences (if None, uses graph.sample_limit())
            start_at_timestep: Start sampling at this timestep (delayed sampling, default 0)
            proj_fun: Projection function for inpainting (defaults to identity if None)

        Returns:
            Sampled sequences tensor, or tuple of (sequences, saved_elements) if save_elements_list provided
        """
        # Default to identity function if proj_fun not provided
        proj_fun = proj_fun or (lambda x: x)
        # Load model using dataset-specific method
        model, graph, noise = self.load_model(checkpoint_path, config, architecture)
        model.eval()

        sequence_length = self.get_sequence_length(config)

        # Generate conditioning labels if not provided
        if conditioning_labels is None:
            conditioning_labels = self.generate_conditioning_labels(num_samples, config)

        # Determine batch size using smart defaults
        if sampling_batch_size is None:
            # Smart default: batch large jobs to avoid memory issues
            sampling_batch_size = 256 if num_samples > 512 else num_samples

        # If batch size equals num_samples, do single-batch sampling (original behavior)
        if sampling_batch_size >= num_samples:
            sampling_fn = sampling.get_pc_sampler(
                graph, noise, (num_samples, sequence_length), 'analytic', steps,
                device=self.device, save_elements_list=save_elements_list, initial_x=initial_x,
                start_at_timestep=start_at_timestep, proj_fun=proj_fun
            )
            result = sampling_fn(model, conditioning_labels.to(self.device))
            if isinstance(result, tuple):
                sampled_sequences, saved_elements = result
                # Convert saved_elements to proper format (N, L, T, 4) and move to CPU
                processed_elements = self._process_saved_elements(saved_elements, num_samples, sequence_length, steps)
                # Move sequences to CPU to free GPU memory
                sampled_sequences = sampled_sequences.cpu()
                del saved_elements
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if save_elements_list:
                    return sampled_sequences, processed_elements
            else:
                # Result is just the sequences (no saved elements)
                sampled_sequences = result
            # Move sequences to CPU even if no saved elements
            sampled_sequences = sampled_sequences.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if save_elements_list:
                # No saved elements returned but save_elements_list was requested - return None
                return sampled_sequences, {}
            return sampled_sequences

        # Batched sampling to avoid flash attention memory issues
        num_batches = (num_samples + sampling_batch_size - 1) // sampling_batch_size
        all_sequences = []  # Will store CPU tensors
        all_saved_elements_processed = {} if save_elements_list else None  # Will store processed CPU tensors
        
        # Initialize structure for accumulating saved elements
        if save_elements_list:
            for elem_name in save_elements_list:
                all_saved_elements_processed[elem_name] = []

        print(f"Sampling {num_samples} sequences in {num_batches} batches of {sampling_batch_size}")

        for i in range(num_batches):
            start_idx = i * sampling_batch_size
            end_idx = min((i + 1) * sampling_batch_size, num_samples)
            current_batch_size = end_idx - start_idx

            # Get labels for this batch
            batch_labels = None
            if conditioning_labels is not None:
                batch_labels = conditioning_labels[start_idx:end_idx]

            # Get initial condition for this batch
            batch_initial_x = None
            if initial_x is not None:
                batch_initial_x = initial_x[start_idx:end_idx]

            # Create sampling function for this batch
            sampling_fn = sampling.get_pc_sampler(
                graph, noise, (current_batch_size, sequence_length),
                'analytic', steps, device=self.device, save_elements_list=save_elements_list,
                initial_x=batch_initial_x, start_at_timestep=start_at_timestep, proj_fun=proj_fun
            )

            # Generate sequences for this batch
            with torch.no_grad():
                result = sampling_fn(model, batch_labels)
                if isinstance(result, tuple):
                    batch_sequences, batch_saved_elements = result
                    
                    # Process and move saved elements to CPU immediately
                    batch_processed = self._process_saved_elements(
                        batch_saved_elements, current_batch_size, sequence_length, steps
                    )
                    # Accumulate processed elements (already on CPU)
                    for elem_name in save_elements_list:
                        if elem_name in batch_processed:
                            all_saved_elements_processed[elem_name].append(batch_processed[elem_name])
                    
                    # Clear GPU references
                    del batch_saved_elements, batch_processed
                else:
                    batch_sequences = result

            # Move sequences to CPU immediately to free GPU memory
            batch_sequences_cpu = batch_sequences.cpu()
            all_sequences.append(batch_sequences_cpu)
            
            # Clear GPU references
            del batch_sequences
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"Completed batch {i+1}/{num_batches} ({end_idx}/{num_samples} sequences)")

        # Concatenate all batches on CPU
        sampled_sequences = torch.cat(all_sequences, dim=0)
        
        # Concatenate processed saved elements if present (already on CPU)
        if all_saved_elements_processed and save_elements_list:
            final_processed_elements = {}
            for elem_name in save_elements_list:
                if elem_name in all_saved_elements_processed and all_saved_elements_processed[elem_name]:
                    # Concatenate along batch dimension (dim 0)
                    final_processed_elements[elem_name] = torch.cat(all_saved_elements_processed[elem_name], dim=0)
            return sampled_sequences, final_processed_elements

        if save_elements_list:
            return sampled_sequences, {}
        return sampled_sequences
    
    def _process_saved_elements(self, saved_elements: Dict[str, List[torch.Tensor]], 
                                num_samples: int, sequence_length: int, steps: int) -> Dict[str, torch.Tensor]:
        """
        Process saved elements into (N, L, T, 4) format.
        
        Args:
            saved_elements: Dict mapping element names to lists of tensors (one per timestep)
            num_samples: Number of sequences (N)
            sequence_length: Sequence length (L)
            steps: Number of timesteps (T)
            
        Returns:
            Dict mapping element names to tensors of shape (N, L, T, 4)
        """
        processed = {}
        # Number of timesteps: steps (predictor steps) + 1 (denoiser step if enabled, or final state)
        # The saved elements should have steps+1 items
        num_timesteps = len(list(saved_elements.values())[0]) if saved_elements else steps + 1
        
        for elem_name, elem_list in saved_elements.items():
            if not elem_list:
                continue
                
            # Stack along timestep dimension
            # elem_list contains T tensors, each of shape (N, L) or (N, L, 4)
            stacked = torch.stack(elem_list, dim=2)  # Stack along dim 2 -> (N, L, T) or (N, L, T, 4)
            
            if elem_name == 'sequence':
                # Convert indices to one-hot: (N, L, T) -> (N, L, T, 4)
                if stacked.dim() == 3:  # (N, L, T) - indices
                    stacked = F.one_hot(stacked.long(), num_classes=4).float()  # (N, L, T, 4)
                elif stacked.dim() == 4 and stacked.shape[-1] != 4:
                    # Already stacked but need to convert to one-hot
                    # If it's (N, L, T, 1) or something, reshape and convert
                    stacked = F.one_hot(stacked.squeeze(-1).long(), num_classes=4).float()
                # If already (N, L, T, 4), use as is
            elif elem_name in ['score', 'stag_score', 'prob']:
                # These should already be (N, L, 4) per timestep, so stacked -> (N, L, T, 4)
                if stacked.dim() == 3:
                    # If somehow (N, L, T), we need to expand to (N, L, T, 4)
                    # This shouldn't happen, but handle it
                    stacked = stacked.unsqueeze(-1).expand(-1, -1, -1, 4)
                # If already (N, L, T, 4), use as is
            
            # Ensure correct shape
            actual_timesteps = stacked.shape[2]
            if stacked.shape[:2] != (num_samples, sequence_length) or stacked.shape[3] != 4:
                # Try to reshape if possible
                if stacked.numel() == num_samples * sequence_length * actual_timesteps * 4:
                    stacked = stacked.view(num_samples, sequence_length, actual_timesteps, 4)
                else:
                    print(f"Warning: {elem_name} shape {stacked.shape} doesn't match expected pattern")
            
            processed[elem_name] = stacked.cpu()
        
        return processed
    
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
    
    def save_sequences(self, sequences: torch.Tensor, output_path: str, format: str = 'npz',
                      encoding: str = 'index'):
        """
        Save generated sequences to file.

        Args:
            sequences: Generated sequences (as indices)
            output_path: Output file path
            format: Output format ('npz', 'fasta', 'csv', 'h5', 'pt')
            encoding: Sequence encoding ('index' or 'onehot')
                     - 'index': Save as integer indices (N, L) where values are 0,1,2,3 for A,C,G,T
                     - 'onehot': Save as one-hot encoding (N, L, 4)
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert to one-hot if requested
        if encoding == 'onehot':
            # Check if already one-hot (has 3 dimensions)
            if sequences.dim() == 2:
                # Convert from indices (N, L) to one-hot (N, L, 4)
                sequences = F.one_hot(sequences.long(), num_classes=4).float()
                print(f"Converting sequences to one-hot encoding: {sequences.shape}")
        elif encoding == 'index':
            # Ensure sequences are indices (2D)
            if sequences.dim() == 3:
                # Convert from one-hot (N, L, 4) to indices (N, L)
                sequences = torch.argmax(sequences, dim=-1)
                print(f"Converting sequences to index encoding: {sequences.shape}")
        else:
            raise ValueError(f"Invalid encoding: {encoding}. Must be 'index' or 'onehot'")
        
        if format.lower() == 'npz':
            # Save as numpy array (matches original implementation)
            np.savez(output_path, sequences.cpu().numpy())
            print(f"Sequences saved to: {output_path} (encoding: {encoding})")

        elif format.lower() == 'fasta':
            # Convert to strings and save as FASTA
            # For FASTA, we need indices for string conversion
            seq_for_strings = sequences if sequences.dim() == 2 else torch.argmax(sequences, dim=-1)
            sequences_str = self.sequences_to_strings(seq_for_strings)
            with open(output_path, 'w') as f:
                for i, seq_str in enumerate(sequences_str):
                    f.write(f">{self.dataset_name}_sequence_{i}\n")
                    f.write(f"{seq_str}\n")
            print(f"Sequences saved to: {output_path}")

        elif format.lower() == 'csv':
            # Convert to strings and save as CSV
            # For CSV, we need indices for string conversion
            seq_for_strings = sequences if sequences.dim() == 2 else torch.argmax(sequences, dim=-1)
            sequences_str = self.sequences_to_strings(seq_for_strings)
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
                    sequences_data = sequences.to(torch.float16).cpu().numpy()
                else:
                    sequences_data = sequences.cpu().numpy()
                f.create_dataset("sequences", data=sequences_data)
            print(f"Sequences saved as HDF5 to: {output_path} (encoding: {encoding}, shape: {sequences.shape})")
        elif format == "pt":
            # PyTorch native format - supports float8 natively (if available)
            torch.save(sequences, output_path)
            print(f"Sequences saved as PyTorch tensor to: {output_path} (dtype: {sequences.dtype}, encoding: {encoding}, shape: {sequences.shape})")
            
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def sample_and_save(self, checkpoint_path: str, config: OmegaConf, num_samples: int, steps: int,
                       architecture: str = 'transformer', conditioning_labels: Optional[torch.Tensor] = None,
                       output_path: Optional[str] = None, format: str = 'npz',
                       sampling_batch_size: Optional[int] = None, encoding: str = 'index',
                       wandb_args=None) -> Dict[str, Any]:
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
            format: Output format ('npz', 'fasta', 'csv', 'h5', 'pt')
            sampling_batch_size: Batch size for sampling (None = smart default)
            encoding: Sequence encoding ('index' or 'onehot')

        Returns:
            Dictionary of sampling results
        """
        # Setup wandb if args provided
        if wandb_args is not None:
            self.setup_wandb(wandb_args, config)

        print(f"Sampling {num_samples} {self.dataset_name} sequences using PC sampler with {steps} steps...")

        # Sample sequences
        sampled_sequences = self.sample_sequences_with_pc_sampler(
            checkpoint_path, config, num_samples, steps, architecture, conditioning_labels, sampling_batch_size
        )

        results = {
            'num_samples': sampled_sequences.shape[0],
            'sequence_length': sampled_sequences.shape[1],
            'sampling_steps': steps,
            'dataset': self.dataset_name,
            'encoding': encoding
        }

        # Save sequences
        if output_path is None:
            # Extract directory from checkpoint path for output
            checkpoint_dir = os.path.dirname(checkpoint_path)
            output_path = os.path.join(checkpoint_dir, f"sample.{format}")

        self.save_sequences(sampled_sequences, output_path, format, encoding)
        results['output_path'] = output_path

        # Log to wandb if enabled
        if self.wandb_enabled:
            try:
                self.log_to_wandb(
                    sequences=sampled_sequences,
                    conditioning_labels=conditioning_labels
                )
            except Exception as e:
                print(f"Warning: Error logging to wandb: {e}")
            finally:
                self.cleanup_wandb()

        return results


    def save_representation(self, checkpoint_path: str, config: OmegaConf,
                              split: str = 'test', save_rep_timestamp: Optional[int] = None,
                              batch_size: Optional[int] = None, architecture: str = 'transformer',
                              output_path: Optional[str] = None, format: str = 'npz',
                              wandb_args=None) -> Dict[str, Any]:
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
        # Setup wandb if args provided
        if wandb_args is not None:
            self.setup_wandb(wandb_args, config)

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

        # Generate output path if not provided
        if output_path is None:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            output_path = os.path.join(checkpoint_dir, f"rep_{save_rep_timestamp}_{split}.{format}")

        # Use handle_sample_result to save with consistent logic
        _, _, save_results = self.handle_sample_result(
            all_representations,
            output_path=output_path,
            format=format,
            encoding='index'
        )
        results.update(save_results)

        print(f"Representation saved to: {output_path}")

        # Log to wandb if enabled
        if self.wandb_enabled:
            try:
                self.log_to_wandb(
                    sequences=None,
                    representations=all_representations
                )
            except Exception as e:
                print(f"Warning: Error logging to wandb: {e}")
            finally:
                self.cleanup_wandb()

        return results

    def save_sampling_elements(self, saved_elements: Dict[str, torch.Tensor],
                              output_path: Optional[str] = None, base_name: str = 'samples') -> str:
        """
        Save sampling elements (score, stag_score, prob, sequence) to HDF5 file.

        This method consolidates the duplicated element-saving logic across dataset samplers.
        Elements are saved with shape (N, L, T, 4) where:
        - N = number of samples
        - L = sequence length
        - T = number of timesteps
        - 4 = number of classes (A, C, G, T)

        Args:
            saved_elements: Dict mapping element names to tensors of shape (N, L, T, 4)
            output_path: Optional output directory path (if None, uses current directory)
            base_name: Base name for output file (default: 'samples')

        Returns:
            Path to saved HDF5 file
        """
        import h5py

        # Determine output directory
        if output_path:
            output_dir = Path(output_path).parent if Path(output_path).suffix else Path(output_path)
            if Path(output_path).suffix:
                # If output_path is a file, use its stem as base_name
                base_name = Path(output_path).stem
        else:
            output_dir = Path('.')

        # Create output file path
        output_file = output_dir / f"{base_name}_elements.h5"

        print(f"\nSaving sampling elements to {output_file}...")
        with h5py.File(output_file, 'w') as f:
            for elem_name, elem_tensor in saved_elements.items():
                # elem_tensor shape: (N, L, T, 4)
                f.create_dataset(elem_name, data=elem_tensor.numpy(), compression='gzip')
                print(f"  Saved dataset '{elem_name}': shape {elem_tensor.shape}")

            # Save metadata as attributes
            if saved_elements:
                first_elem = list(saved_elements.values())[0]
                f.attrs['num_samples'] = first_elem.shape[0]
                f.attrs['sequence_length'] = first_elem.shape[1]
                f.attrs['num_timesteps'] = first_elem.shape[2]
                f.attrs['num_classes'] = first_elem.shape[3]
                f.attrs['saved_elements'] = list(saved_elements.keys())
                f.attrs['dataset'] = self.dataset_name

        print(f"  All elements saved to: {output_file}")
        return str(output_file)

    def handle_sample_result(self, result: Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
                            output_path: Optional[str] = None, format: str = 'npz',
                            encoding: str = 'indices') -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Dict[str, Any]]:
        """
        Handle sampling result (sequences or tuple) and optionally save sequences.

        This method consolidates the duplicated result-handling logic across dataset samplers.

        Args:
            result: Either sequences tensor or (sequences, saved_elements) tuple
            output_path: Optional path to save sequences
            format: Output format for sequences
            encoding: Sequence encoding type

        Returns:
            Tuple of (sequences, saved_elements, results_dict)
        """
        # Unpack result
        if isinstance(result, tuple):
            sequences, saved_elements = result
        else:
            sequences = result
            saved_elements = None

        # Create results dictionary
        results = {
            'num_sequences': len(sequences),
            'sequence_length': self.get_sequence_length(OmegaConf.create()) if hasattr(self, 'get_sequence_length') else sequences.shape[1]
        }

        # Save sequences if output path provided
        if output_path:
            self.save_sequences(sequences, output_path, format, encoding)
            results['output_file'] = output_path
            results['encoding'] = encoding

        return sequences, saved_elements, results

    def handle_saved_elements(self, saved_elements: Optional[Dict[str, torch.Tensor]],
                             output_path: Optional[str], base_name: str) -> Dict[str, Any]:
        """
        Handle saved elements by saving them to HDF5 and returning metadata.

        This method consolidates the duplicated saved_elements handling logic
        across all dataset samplers.

        Args:
            saved_elements: Optional dict of saved elements (or None if no elements)
            output_path: Optional output path for sequences
            base_name: Base name for the output file (e.g., 'deepstarr_samples')

        Returns:
            Dictionary with saved_elements_file and saved_elements keys (empty if no elements)
        """
        results = {}

        if saved_elements:
            elements_file = self.save_sampling_elements(saved_elements, output_path, base_name)
            results['saved_elements_file'] = elements_file
            results['saved_elements'] = list(saved_elements.keys())

        return results

    @staticmethod
    def load_config_with_fallback(config_path: Optional[str], dataset_dir: Path,
                                 default_name: str = 'transformer.yaml') -> Tuple[OmegaConf, str]:
        """
        Load config with standard fallback logic.

        This method consolidates the duplicated config-loading logic across dataset samplers.

        Args:
            config_path: Optional path to config file
            dataset_dir: Path to dataset directory (e.g., model_zoo/promoter)
            default_name: Default config filename to use if config_path not provided

        Returns:
            Tuple of (config, config_path_used)
        """
        if not config_path:
            # Try to use default config from dataset directory
            try:
                config_path = dataset_dir / 'configs' / default_name
                if config_path.exists():
                    print(f"Using default config: {config_path}")
                else:
                    raise FileNotFoundError(
                        f"No config provided and default config not found: {config_path}\n"
                        f"Please provide a config file with --config"
                    )
            except Exception as e:
                raise RuntimeError(f"Error loading default config: {e}")

        config = OmegaConf.load(config_path)
        return config, str(config_path)

    def load_and_generate_labels(self, config: OmegaConf, num_samples: int,
                                use_test_set: bool, data_path: Optional[str],
                                dataset_class, specific_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
        """
        Load labels from test set or generate random labels.

        This method consolidates the duplicated test-set label loading logic across dataset samplers.

        Args:
            config: Configuration object
            num_samples: Number of samples to generate labels for
            use_test_set: Whether to use test set labels
            data_path: Path to data file (required if use_test_set is True)
            dataset_class: Dataset class to use for loading test data
            specific_labels: Optional pre-computed labels to use

        Returns:
            Tuple of (labels, actual_num_samples)
        """
        if specific_labels is not None:
            # Use user-provided labels
            return specific_labels.to(self.device), num_samples

        if use_test_set:
            # Load test set labels
            if not data_path:
                raise ValueError("--data_path is required when using --use_test_set")

            # Load test dataset to get labels
            test_dataset = dataset_class(data_path, split='test')
            labels = test_dataset.y.to(self.device)
            num_samples = len(test_dataset)
            print(f"Using test set labels: {num_samples} samples with shape {labels.shape}")
            return labels, num_samples
        else:
            # Generate random labels
            labels = self.generate_conditioning_labels(num_samples, config)
            print(f"Using random labels with shape {labels.shape}")
            return labels, num_samples


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
    parser.add_argument('--sequence_encoding', choices=['index', 'onehot'], default='index',
                       help='Sequence encoding: "index" for integer indices (0-3 for A,C,G,T) or "onehot" for one-hot encoding')
    parser.add_argument('--save_elements', type=str, nargs='+', default=None,
                       choices=['sequence', 'score', 'activity_label'],
                       help='List of elements to save during sampling: sequence, score, activity_label. '
                            'sequence and score are saved as (N, L, T, 4) tensors. '
                            'activity_label saves the conditioning labels used for generation.')
    parser.add_argument('--use_test_set', action='store_true', default=False,
                       help='Use test set labels from dataset as conditioning labels')
    parser.add_argument('--initial_condition', type=str, default='random',
                       choices=['random', 'test', 'dinuc', 'custom'],
                       help='Initial condition for sampling: random (default), test (use onehot_test sequences), '
                            'dinuc (use pre-computed onehot_test_dinuc sequences from H5 file), '
                            'or custom (provide custom sequences via model-specific arguments). '
                            'Requires --data_path when using test or dinuc.')
    parser.add_argument('--start_at_timestep', type=int, default=0,
                       help='Start sampling at this timestep (delayed sampling). Default is 0 (start from beginning).')

    # Inpainting arguments
    parser.add_argument('--inpainting_mode', type=str, default='none',
                       choices=['none', 'inpaint_motifs', 'inpaint_not_motifs'],
                       help='Inpainting mode for constrained generation: '
                            'inpaint_motifs (fix outside, generate inside motif regions), '
                            'inpaint_not_motifs (fix inside, generate outside motif regions), '
                            'or none (no inpainting)')
    parser.add_argument('--inpainting_data', type=str, default=None,
                       help='Path to all_hits_combined.h5 (required when inpainting_mode != none)')
    parser.add_argument('--inpainting_seed', type=int, default=None,
                       help='Random seed for choosing dev vs hk positions when both are available')
    parser.add_argument('--inpainting_iterations', type=int, default=1,
                       help='Number of sampling iterations per sample (repeats sampling N times for each motif)')

    # Wandb logging arguments
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Enable Weights & Biases logging for sampling run')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='Wandb project name (default: {dataset_name}-sampling)')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Wandb run name (default: auto-generated)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity/team name (optional)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None,
                       help='Wandb tags for the run (optional)')

    return parser

    # TODO: unify save elements and save rep


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
                format=args.format,
                wandb_args=args
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
        encoding=args.sequence_encoding,
        wandb_args=args
    )
    
    # Print results
    print(f"\n{sampler.dataset_name} Sampling Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")
    
    print(f"\n✓ {sampler.dataset_name} sampling completed successfully!")
    return 0