#!/usr/bin/env python3
"""
Base Evaluation Framework for D3-DNA Discrete Diffusion

This module provides the base evaluation framework that dataset-specific
evaluation scripts should inherit from. It provides common functionality
while allowing datasets to implement their own model and data loading.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm


class BaseEvaluator:
    """
    Base evaluator class that provides common evaluation functionality.
    
    Dataset-specific evaluation scripts should inherit from this class and
    implement the abstract methods for their specific needs.
    """
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """
        Load model for the dataset. Must be implemented by subclasses.
        Each dataset implements its own simple model loading logic.
        
        Args:
            checkpoint_path: Path to specific checkpoint file
            config: Configuration object
            architecture: Architecture type ('transformer', 'convolutional')
            
        Returns:
            Tuple of (model, graph, noise) needed for sampling/evaluation
        """
        raise NotImplementedError("Subclasses must implement load_model()")
    
    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None, max_samples: Optional[int] = None):
        """
        Create dataloader for evaluation. Must be implemented by subclasses.
        
        Args:
            config: Configuration object
            split: Dataset split ('train', 'val', 'test')
            batch_size: Batch size (if None, uses config default)
            max_samples: Maximum number of samples to evaluate (if None, uses entire dataset)
            
        Returns:
            DataLoader instance
        """
        raise NotImplementedError("Subclasses must implement create_dataloader()")
    
    
    def load_oracle_model(self, oracle_checkpoint: str, data_path: str):
        """
        Load oracle model for specialized evaluation metrics.
        Can be overridden by subclasses for dataset-specific oracle loading.
        
        Args:
            oracle_checkpoint: Path to oracle model checkpoint
            data_path: Path to data file needed by oracle
            
        Returns:
            Oracle model instance or None if not supported
        """
        print(f"Warning: Oracle evaluation not implemented for {self.dataset_name}")
        return None
    
    def sample_sequences_for_evaluation(self, checkpoint_path: str, config: OmegaConf, 
                                       dataloader, num_steps: int, architecture: str = 'transformer', 
                                       show_progress: bool = False, viz_logger=None, oracle_model=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample sequences for evaluation using PC sampler.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Configuration object  
            dataloader: DataLoader for getting target labels
            num_steps: Number of sampling steps
            architecture: Architecture type
            show_progress: Whether to show progress bar during sampling
            viz_logger: Optional visualization data logger
            oracle_model: Optional oracle model for MSE computation during sampling
            
        Returns:
            Tuple of (sampled_sequences, target_labels)
        """
        from scripts import sampling
        
        # Load model using dataset-specific method  
        model, graph, noise = self.load_model(checkpoint_path, config, architecture)
        model.eval()
        
        # Get sequence length
        sequence_length = self.get_sequence_length(config)
        
        # Get batch size from dataloader
        batch_size = dataloader.batch_size
        
        # Update visualization logger with noise schedule metadata
        if viz_logger is not None:
            noise_config = {
                'type': getattr(config.noise, 'type', 'geometric'),
                'sigma_min': getattr(config.noise, 'sigma_min', 1e-3),
                'sigma_max': getattr(config.noise, 'sigma_max', 1.0)
            }
            viz_logger.update_noise_schedule_metadata(noise_config)
        
        # Create special PC sampler for evaluation with visualization and oracle MSE
        if viz_logger is not None and oracle_model is not None:
            # Use a custom sampling function that captures oracle MSE at each step
            sampling_fn = self._get_evaluation_pc_sampler_with_viz(
                graph, noise, (batch_size, sequence_length), num_steps, 
                viz_logger, oracle_model
            )
        else:
            # Regular PC sampler with optional visualization
            sampling_fn = sampling.get_pc_sampler(
                graph, noise, (batch_size, sequence_length), 'analytic', num_steps, 
                device=self.device, viz_logger=viz_logger
            )
        
        sampled_sequences = []
        all_targets = []
        
        # Wrap dataloader with tqdm if show_progress is True
        if show_progress:
            dataloader = tqdm(dataloader, desc="Sampling sequences")
        
        for batch_idx, (batch, targets) in enumerate(dataloader):
            current_batch_size = batch.shape[0]
            
            # If last batch has different size, create new sampling function
            if current_batch_size != batch_size:
                if viz_logger is not None and oracle_model is not None:
                    sampling_fn = self._get_evaluation_pc_sampler_with_viz(
                        graph, noise, (current_batch_size, sequence_length), num_steps,
                        viz_logger, oracle_model
                    )
                else:
                    sampling_fn = sampling.get_pc_sampler(
                        graph, noise, (current_batch_size, sequence_length), 'analytic', num_steps, 
                        device=self.device, viz_logger=viz_logger
                    )
            
            # Sample sequences conditioned on targets
            sample = sampling_fn(model, targets.to(self.device))
            seq_pred_one_hot = F.one_hot(sample, num_classes=4).float()
            sampled_sequences.append(seq_pred_one_hot)
            all_targets.append(targets)
        
        # Concatenate all samples
        all_samples = torch.cat(sampled_sequences, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        return all_samples, all_targets
    
    def _get_evaluation_pc_sampler_with_viz(self, graph, noise, batch_dims, steps, viz_logger, oracle_model):
        """
        Create a PC sampler that captures oracle MSE at each step during evaluation.
        
        This is a specialized version for evaluation that computes oracle MSE predictions
        at each step and logs them with the visualization data.
        """
        from scripts.sampling import get_predictor, Denoiser
        from utils.utils import get_score_fn
        import torch.nn.functional as F
        
        predictor = get_predictor('analytic')(graph, noise)
        denoiser = Denoiser(graph, noise)
        eps = 1e-5
        
        @torch.no_grad()
        def evaluation_pc_sampler_with_viz(model, labels):
            sampling_score_fn = get_score_fn(model, train=False, sampling=True)
            x = graph.sample_limit(*batch_dims).to(self.device)
            timesteps = torch.linspace(1, eps, steps + 1, device=self.device)
            dt = (1 - eps) / steps

            # Get ground truth oracle predictions once (for SP-MSE computation)
            ground_truth_oracle_predictions = None
            if oracle_model is not None:
                try:
                    # Convert labels to one-hot format for oracle prediction
                    # labels should be token indices, convert to LongTensor first, then to one-hot
                    labels_long = labels.long()
                    labels_one_hot = F.one_hot(labels_long, num_classes=4).float()
                    ground_truth_oracle_predictions = self.get_oracle_predictions_for_viz(labels_one_hot, oracle_model)
                except Exception as e:
                    print(f"Warning: Could not compute ground truth oracle predictions: {e}")
                    ground_truth_oracle_predictions = None

            for i in range(steps):
                t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
                
                # Get current noise level and score matrix
                sigma, dsigma = noise(t.squeeze())
                score_matrix = sampling_score_fn(x, sigma, labels)
                
                # Normalize score matrix
                # score_matrix = score_matrix / score_matrix.sum(dim=-1, keepdim=True)
                
                # Calculate prob_matrix following the same pattern as AnalyticPredictor
                curr_sigma = noise(t)[0]
                next_sigma = noise(t - dt)[0]
                dsigma_step = curr_sigma - next_sigma
                stag_score = graph.staggered_score(score_matrix, dsigma_step)
                prob_matrix = stag_score * graph.transp_transition(x, dsigma_step)
                
                # Compute oracle predictions and SP-MSE for current sequences
                oracle_mse = None
                oracle_predictions = None
                if oracle_model is not None and ground_truth_oracle_predictions is not None:
                    try:
                        # Convert sequences to one-hot for oracle prediction
                        x_long = x.long()
                        x_one_hot = F.one_hot(x_long, num_classes=4).float()
                        oracle_predictions = self.get_oracle_predictions_for_viz(x_one_hot, oracle_model)
                        
                        # Compute proper SP-MSE: (ground_truth_oracle - current_oracle)^2
                        sp_mse = (ground_truth_oracle_predictions - oracle_predictions) ** 2
                        oracle_mse = sp_mse.mean(dim=-1)  # Average across output dimensions
                    except Exception as e:
                        print(f"Warning: Could not compute oracle MSE at step {i}: {e}")
                        oracle_mse = None
                        oracle_predictions = None
                
                # Log the step data
                viz_logger.log_step(
                    step=i,
                    timestep=timesteps[i].item(),
                    sequences=x,
                    score_matrix=score_matrix,
                    prob_matrix=prob_matrix,
                    noise_level=sigma.mean().item() if sigma.numel() > 1 else sigma.item(),
                    noise_rate=dsigma.mean().item() if dsigma.numel() > 1 else dsigma.item(),
                    oracle_mse=oracle_mse,
                    ground_truth_labels=labels,
                    oracle_predictions=oracle_predictions
                )
                
                # Store ground truth oracle predictions in metadata (only once, at step 0)
                if ground_truth_oracle_predictions is not None:
                    viz_logger.metadata['ground_truth_oracle_predictions'] = ground_truth_oracle_predictions.detach().cpu()
                
                x = predictor.update_fn(sampling_score_fn, x, labels, t, dt)

            # Final denoising step
            x = denoiser.update_fn(sampling_score_fn, x, labels, timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device))
            
            return x
        
        return evaluation_pc_sampler_with_viz
    
    def get_oracle_predictions_for_viz(self, sequences: torch.Tensor, oracle_model) -> torch.Tensor:
        """
        Get oracle predictions for sequences during visualization.
        Can be overridden by subclasses for dataset-specific oracle handling.
        
        Args:
            sequences: One-hot encoded sequences (batch_size, seq_length, 4)
            oracle_model: Oracle model
            
        Returns:
            Oracle predictions tensor
        """
        # Default implementation - subclasses should override for specific datasets
        if hasattr(oracle_model, 'predict_custom'):
            # Convert from (batch, length, channels) to (batch, channels, length)
            sequences_input = sequences.permute(0, 2, 1).to(self.device)
            return oracle_model.predict_custom(sequences_input)
        else:
            # Fallback
            return torch.zeros(sequences.shape[0], device=self.device)
    
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
                'promoter': 1024,
                'atacseq': 1001
            }
            return defaults.get(self.dataset_name.lower(), 249)
    
    def compute_sp_mse(self, sampled_sequences: torch.Tensor, oracle_model, 
                      original_data: torch.Tensor) -> float:
        """
        Compute SP-MSE using oracle model.
        
        Args:
            sampled_sequences: Generated sequences (token indices)
            oracle_model: Oracle model for evaluation
            original_data: Original test data for comparison
            
        Returns:
            SP-MSE score
        """
        
        # Safety check: ensure original data matches the number of sampled sequences
        num_samples = sampled_sequences.shape[0]
        if original_data.shape[0] != num_samples:
            print(f"  ⚠️  Warning: Original data has {original_data.shape[0]} samples but sampled {num_samples}. Using first {num_samples} for SP-MSE.")
            original_data = original_data[:num_samples]
        
        # Get oracle predictions for original and generated data using dataset-specific method
        # Convert original_data to one-hot format if it's not already
        if len(original_data.shape) == 3 and original_data.shape[1] == 4:
            # Already in one-hot format
            original_one_hot = original_data
        else:
            # Convert from token indices to one-hot
            original_long = original_data.long()
            original_one_hot = F.one_hot(original_long, num_classes=4).float()
        
        # Convert sampled_sequences to one-hot format
        sampled_long = sampled_sequences.long()
        sampled_one_hot = F.one_hot(sampled_long, num_classes=4).float()
        
        # Get predictions using dataset-specific method
        val_score = self.get_oracle_predictions_for_viz(original_one_hot, oracle_model)
        val_pred_score = self.get_oracle_predictions_for_viz(sampled_one_hot, oracle_model)
        
        # Compute SP-MSE
        sp_mse = (val_score - val_pred_score) ** 2
        mean_sp_mse = torch.mean(sp_mse).cpu().item()
        
        return mean_sp_mse
    
    def get_original_test_data(self, data_path: str) -> torch.Tensor:
        """
        Get original test data for SP-MSE comparison. Must be implemented by subclasses.
        
        Args:
            data_path: Path to data file
            
        Returns:
            Original test data tensor
        """
        raise NotImplementedError("Subclasses must implement get_original_test_data()")
    
    def evaluate_with_oracle(self, model, oracle_model, dataloader, config: OmegaConf) -> Dict[str, Any]:
        """
        Evaluate model using oracle model (for SP-MSE and similar metrics).
        Can be overridden by subclasses for dataset-specific oracle evaluation.
        
        Args:
            model: Trained diffusion model
            oracle_model: Oracle model for evaluation
            dataloader: Data loader
            config: Configuration object
            
        Returns:
            Dictionary of oracle-based evaluation metrics
        """
        if oracle_model is None:
            return {'oracle_evaluation': 'not_supported'}
        
        # Default implementation returns placeholder
        return {
            'oracle_evaluation': 'not_implemented',
            'sp_mse': 0.0
        }
    
    def evaluate_with_sampling(self, checkpoint_path: str, config: OmegaConf, 
                              oracle_checkpoint: str, data_path: str,
                              split: str = 'test', steps: Optional[int] = None, 
                              batch_size: Optional[int] = None, architecture: str = 'transformer',
                              show_progress: bool = False, save_sequences: bool = False,
                              save_visualization_data: bool = False, viz_output_path: Optional[str] = None,
                              viz_format: str = 'hdf5', max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate model by sampling sequences and computing SP-MSE with oracle.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Configuration object
            oracle_checkpoint: Path to oracle model checkpoint
            data_path: Path to data file needed by oracle
            split: Dataset split to evaluate on
            steps: Number of sampling steps (defaults to sequence length)
            batch_size: Batch size for evaluation (optional)
            architecture: Architecture type
            show_progress: Whether to show progress bar during sampling
            save_sequences: Whether to save sampled sequences as NPZ file
            save_visualization_data: Whether to save intermediate sampling data
            viz_output_path: Output path for visualization data
            viz_format: Format for visualization data ('hdf5', 'npz')
            max_samples: Maximum number of samples to evaluate (if None, uses entire dataset)
            
        Returns:
            Dictionary of evaluation results including SP-MSE
        """
        print(f"Evaluating {self.dataset_name} on {split} split with sampling...")
        
        # Set default steps to sequence length if not provided
        if steps is None:
            steps = self.get_sequence_length(config)
            print(f"Using default steps: {steps} (sequence length)")
        
        # Create dataloader with optional sample limiting
        dataloader = self.create_dataloader(config, split, batch_size, max_samples)
        
        # Print evaluation info
        if max_samples is not None:
            print(f"  ↳ Limiting evaluation to {max_samples} samples (randomly selected)")
        
        # Load oracle model first (needed for visualization if enabled)
        print("Loading oracle model for SP-MSE evaluation...")
        oracle_model = self.load_oracle_model(oracle_checkpoint, data_path)
        
        if oracle_model is None:
            return {
                'error': 'oracle_model_not_loaded',
                'sampling_steps': steps
            }
        
        # Create visualization logger if requested
        viz_logger = None
        if save_visualization_data:
            from utils.visualization_logger import create_visualization_logger
            sequence_length = self.get_sequence_length(config)
            actual_samples = len(dataloader.dataset)
            viz_logger = create_visualization_logger(
                num_samples=actual_samples,
                sequence_length=sequence_length,
                num_steps=steps,
                dataset_name=self.dataset_name,
                architecture=architecture,
                split=split,
                save_oracle_mse=True,  # Enable oracle MSE for evaluation
                device=self.device
            )
            print(f"  ↳ Visualization data logging enabled with oracle MSE ({actual_samples} samples)")
        
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
        
        # Get original test data for comparison
        original_data = self.get_original_test_data(data_path)
        
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
                viz_output_path = os.path.join(checkpoint_dir, f"evaluation_visualization_data.{viz_format}")
            
            viz_logger.save(viz_output_path, viz_format)
            results['visualization_output_path'] = viz_output_path
        
        print(f"SP-MSE: {sp_mse:.6f}")
        
        return results
    
    def save_sequences_as_npz(self, sequences: torch.Tensor, save_path: str):
        """
        Save sampled sequences as NPZ file.
        
        Args:
            sequences: Sampled sequences tensor (batch_size, seq_len, 4)
            save_path: Path to save the NPZ file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, sequences.cpu().numpy())
        print(f"✓ Saved {sequences.shape[0]} sequences to {save_path}")

    def evaluate(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer',
                 split: str = 'test', oracle_checkpoint: Optional[str] = None,
                 data_path: Optional[str] = None, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Main evaluation method - now deprecated. Use evaluate_with_sampling instead.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Configuration object
            architecture: Architecture type
            split: Dataset split to evaluate on
            oracle_checkpoint: Path to oracle model checkpoint (optional)
            data_path: Path to data file (optional, needed for oracle)
            batch_size: Batch size for evaluation (optional)
            
        Returns:
            Dictionary of evaluation results
        """
        print("Warning: evaluate() method is deprecated. Use evaluate_with_sampling() instead.")
        print("This method only performs oracle evaluation without sampling.")
        
        if not oracle_checkpoint:
            return {'error': 'oracle_checkpoint required for evaluation'}
        
        # Load model using dataset-specific method
        model, _, _ = self.load_model(checkpoint_path, config, architecture)
        
        # Create dataloader
        dataloader = self.create_dataloader(config, split, batch_size)
        
        print(f"Evaluating {self.dataset_name} on {split} split...")
        
        # Oracle evaluation only
        print("Loading oracle model for specialized evaluation...")
        oracle_model = self.load_oracle_model(oracle_checkpoint, data_path or "")
        metrics = self.evaluate_with_oracle(model, oracle_model, dataloader, config)
        
        return metrics
    
    def save_results(self, metrics: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"Results saved to: {output_path}")
    
    def print_results(self, metrics: Dict[str, Any]):
        """Print evaluation results in a formatted way."""
        print(f"\n{self.dataset_name} Evaluation Results:")
        print("=" * 40)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")


def parse_base_args():
    """Parse common command line arguments for evaluation scripts."""
    parser = argparse.ArgumentParser(description='D3 Evaluation Script - Sampling + SP-MSE')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint file')
    parser.add_argument('--architecture', required=False, choices=['transformer', 'convolutional'], default='transformer', help='Model architecture')
    parser.add_argument('--oracle_checkpoint', required=False, help='Path to oracle model checkpoint (required for SP-MSE)')
    parser.add_argument('--data_path', required=False, help='Path to data file (required for oracle models)')
    parser.add_argument('--config', help='Path to config file (optional, dataset may provide default)')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test', help='Dataset split to evaluate on')
    parser.add_argument('--steps', type=int, help='Number of sampling steps (defaults to sequence length)')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for evaluation')
    parser.add_argument('--show_progress', action='store_true', help='Show progress bar during sampling')
    parser.add_argument('--save_sequences', action='store_true', help='Save sampled sequences as NPZ file')
    
    # Visualization data arguments
    parser.add_argument('--save_viz_data', action='store_true', help='Save visualization data (sequences, scores, noise levels, oracle MSE)')
    parser.add_argument('--viz_output', help='Output path for visualization data')
    parser.add_argument('--viz_format', choices=['hdf5', 'h5', 'npz'], default='hdf5', help='Visualization data format')
    
    # Evaluation sample limiting
    parser.add_argument('--max_samples', type=int, help='Maximum number of samples to evaluate (randomly selected if less than dataset size)')
    
    return parser


def main_evaluate(evaluator: BaseEvaluator, args):
    """
    Common main evaluation function that can be used by dataset-specific scripts.
    Now focuses on sampling + SP-MSE evaluation only.
    
    Args:
        evaluator: Dataset-specific evaluator instance
        args: Parsed command line arguments
    """
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1
    
    if not args.oracle_checkpoint:
        print("Error: --oracle_checkpoint is required for SP-MSE evaluation")
        return 1
    
    if not args.data_path:
        print("Error: --data_path is required for SP-MSE evaluation")
        return 1
    
    # Load configuration (required)
    if not args.config:
        print("Error: Config file is required. Please provide --config path/to/config.yaml")
        return 1
    
    config = OmegaConf.load(args.config)
    
    # Handle progress bar settings
    show_progress = args.show_progress
    
    # Run evaluation with sampling (the main evaluation method now)
    metrics = evaluator.evaluate_with_sampling(
        checkpoint_path=args.checkpoint,
        config=config,
        oracle_checkpoint=args.oracle_checkpoint,
        data_path=args.data_path,
        split=args.split,
        steps=args.steps,
        batch_size=args.batch_size,
        architecture=args.architecture,
        show_progress=show_progress,
        save_sequences=args.save_sequences,
        save_visualization_data=args.save_viz_data,
        viz_output_path=args.viz_output,
        viz_format=args.viz_format,
        max_samples=args.max_samples
    )
    
    # Print results
    evaluator.print_results(metrics)
    
    # Save results
    dataset_name = evaluator.dataset_name
    output_path = args.output or f"evaluation_results/{dataset_name}_{args.architecture}_{args.split}_results.json"
    evaluator.save_results(metrics, output_path)
    
    return 0