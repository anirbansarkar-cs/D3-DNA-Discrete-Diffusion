#!/usr/bin/env python3
"""
cCRE Evaluation Script

Implements variant effect prediction using the diffusion model instead of oracle MSE.
This evaluator can process variant data and compute effect scores based on 
sequence embedding distances.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import h5py

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import base framework and cCRE-specific components
from scripts.evaluate import BaseEvaluator, parse_base_args, main_evaluate
from model_zoo.ccre.data import get_ccre_datasets


class cCREEvaluator(BaseEvaluator):
    """cCRE-specific evaluator for variant effect prediction."""
    
    def __init__(self):
        super().__init__("cCRE")
        self._dataset_indices = None  # Store indices for matching original data
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """Load cCRE model using dataset-specific model loading."""
        from model_zoo.ccre.models import load_trained_model
        
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None, max_samples: Optional[int] = None):
        """Create cCRE dataloader with optional sample limiting."""
        # Get split configuration from config
        train_ratio = getattr(config.data, 'train_ratio', 0.95)
        valid_ratio = getattr(config.data, 'valid_ratio', 0.05)
        split_seed = getattr(config.data, 'split_seed', 42)
        
        # Load datasets
        train_ds, val_ds = get_ccre_datasets(
            config.paths.data_file,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            seed=split_seed
        )
        
        # Select appropriate dataset
        if split == 'train':
            dataset = train_ds
        elif split in ['val', 'test']:  # Use val as test for now
            dataset = val_ds
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
            print(f"  ↳ cCRE dataset limited to {len(dataset)} samples from {split} split")
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
    
    def get_original_test_data(self, data_path: str) -> torch.Tensor:
        """Get original test data for evaluation, matching the limited dataset if applicable."""
        try:
            # Load cCRE data from the 'seqs' key
            print(f"Loading original test data from: {data_path}")
            with h5py.File(data_path, 'r') as data_file:
                X = torch.tensor(np.array(data_file['seqs']))
            
            # If we limited the dataset, apply the same indices to original data
            if self._dataset_indices is not None:
                print(f"  ↳ Applying same subset indices to original data ({len(self._dataset_indices)} samples)")
                X = X[self._dataset_indices]
                
            return X
        except Exception as e:
            print(f"Error loading original test data: {e}")
            return torch.zeros(100, 4, 512)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        """Get cCRE sequence length."""
        return 512
    
    def encode_sequence(self, model, sequence: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence using the diffusion model to get its representation.
        
        Args:
            model: Trained diffusion model
            sequence: One-hot encoded sequence tensor (batch_size, 4, seq_length)
            
        Returns:
            Sequence embedding/representation
        """
        model.eval()
        with torch.no_grad():
            # Convert one-hot to indices
            if sequence.dim() == 3 and sequence.shape[1] == 4:
                indices = torch.argmax(sequence, dim=1)  # (batch_size, seq_length)
            else:
                indices = sequence
            
            # Get model embedding - we'll use the model's internal representation
            # For this, we need to access the model's encoder layers
            if hasattr(model, 'transformer'):
                # For transformer models, get the final hidden states
                embedding = model.transformer.encode(indices)
            elif hasattr(model, 'conv_blocks'):
                # For convolutional models, use the feature representation
                x = F.one_hot(indices, num_classes=4).float().permute(0, 2, 1)
                embedding = model.linear(x)
                for block in model.conv_blocks[:3]:  # Use first few conv blocks
                    embedding = F.relu(block(embedding))
                embedding = F.adaptive_avg_pool1d(embedding, 1).squeeze(-1)
            else:
                # Fallback: use the full forward pass with dummy conditioning
                dummy_sigma = torch.ones(indices.shape[0], device=indices.device) * 0.1
                try:
                    output = model(indices, train=False, sigma=dummy_sigma)
                    # Use mean of output as embedding
                    embedding = output.mean(dim=1)
                except:
                    # Final fallback: simple average of one-hot encoding
                    x = F.one_hot(indices, num_classes=4).float()
                    embedding = x.mean(dim=1)
        
        return embedding
    
    def compute_variant_effect_score(self, model, ref_seq: torch.Tensor, alt_seq: torch.Tensor) -> float:
        """
        Compute variant effect score as the distance between reference and alternate embeddings.
        
        Args:
            model: Trained diffusion model
            ref_seq: Reference sequence (one-hot encoded)
            alt_seq: Alternate sequence (one-hot encoded)
            
        Returns:
            Variant effect score (higher = more effect)
        """
        # Get embeddings
        ref_embedding = self.encode_sequence(model, ref_seq.unsqueeze(0))
        alt_embedding = self.encode_sequence(model, alt_seq.unsqueeze(0))
        
        # Compute Euclidean distance
        score = F.pairwise_distance(ref_embedding, alt_embedding).item()
        
        return score
    
    def sequence_from_variant(self, chrom: str, pos: int, ref: str, alt: str, 
                            genome_fasta_path: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate reference and alternate sequences from variant information.
        
        Args:
            chrom: Chromosome (e.g., '1', '2', 'X')
            pos: Position (1-based)
            ref: Reference allele
            alt: Alternate allele
            genome_fasta_path: Path to genome FASTA file
            
        Returns:
            Tuple of (ref_sequence, alt_sequence) as one-hot tensors
        """
        window_size = 512
        
        # For now, create a dummy implementation
        # In practice, you would load from genome FASTA
        if genome_fasta_path is None:
            print("Warning: No genome FASTA provided, using dummy sequences")
            # Create random sequences for demonstration
            ref_seq = torch.randint(0, 4, (window_size,))
            alt_seq = ref_seq.clone()
            
            # Place the variant at the center
            center = window_size // 2
            ref_base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
            ref_seq[center] = ref_base_map.get(ref.upper(), 0)
            alt_seq[center] = ref_base_map.get(alt.upper(), 0)
        else:
            # TODO: Implement genome sequence loading
            # This would use pyfaidx or similar to load actual sequences
            raise NotImplementedError("Genome FASTA loading not implemented yet")
        
        # Convert to one-hot
        ref_onehot = F.one_hot(ref_seq, num_classes=4).float()
        alt_onehot = F.one_hot(alt_seq, num_classes=4).float()
        
        return ref_onehot, alt_onehot
    
    def evaluate_variants(self, checkpoint_path: str, config: OmegaConf, 
                         variants: List[Dict], architecture: str = 'transformer',
                         genome_fasta_path: str = None) -> Dict[str, Any]:
        """
        Evaluate variant effect prediction on a list of variants.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration object
            variants: List of variant dictionaries
            architecture: Model architecture
            genome_fasta_path: Path to genome FASTA file
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating {len(variants)} variants with cCRE model...")
        
        # Load model
        model, graph, noise = self.load_model(checkpoint_path, config, architecture)
        model.eval()
        
        scores = []
        failed_variants = 0
        
        for variant in tqdm(variants, desc="Processing variants"):
            try:
                # Extract variant information
                chrom = str(variant['chrom'])
                pos = int(variant['pos'])
                ref = str(variant['ref'])
                alt = str(variant['alt'])
                
                # Generate sequences
                ref_seq, alt_seq = self.sequence_from_variant(chrom, pos, ref, alt, genome_fasta_path)
                
                # Compute variant effect score
                score = self.compute_variant_effect_score(model, ref_seq, alt_seq)
                scores.append(score)
                
            except Exception as e:
                print(f"Failed to process variant {variant}: {e}")
                scores.append(0.0)
                failed_variants += 1
        
        results = {
            'dataset': 'cCRE',
            'num_variants': len(variants),
            'failed_variants': failed_variants,
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'model_checkpoint': checkpoint_path,
            'architecture': architecture
        }
        
        return results
    
    def evaluate_with_sampling(self, checkpoint_path: str, config: OmegaConf, 
                              oracle_checkpoint: str = None, data_path: str = None,
                              split: str = 'test', steps: Optional[int] = None, 
                              batch_size: Optional[int] = None, architecture: str = 'transformer',
                              show_progress: bool = False, save_sequences: bool = False,
                              save_visualization_data: bool = False, viz_output_path: Optional[str] = None,
                              viz_format: str = 'hdf5', max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Override base evaluation to provide variant effect prediction with optional visualization.
        
        Since cCRE has no oracle model, we skip oracle-based evaluation and focus on
        the model's ability to distinguish between sequences.
        """
        print(f"cCRE evaluation: Variant effect prediction mode")
        print("Note: Oracle-based evaluation not applicable for unlabeled cCRE data")
        
        # Load model for basic validation
        model, graph, noise = self.load_model(checkpoint_path, config, architecture)
        
        # Set default steps to sequence length if not provided
        if steps is None:
            steps = self.get_sequence_length(config)
            print(f"Using default steps: {steps} (sequence length)")
        
        # Create dataloader with optional sample limiting (for visualization)
        if save_visualization_data:
            dataloader = self.create_dataloader(config, split, batch_size, max_samples)
            
            # Get original test data for potential future use
            original_data = self.get_original_test_data(data_path)
            
            # Create visualization logger
            from utils.visualization_logger import create_visualization_logger
            sequence_length = self.get_sequence_length(config)
            actual_samples = len(dataloader.dataset)
            
            # Convert original samples to token indices for visualization storage
            original_samples_indices = None
            if original_data is not None:
                # Convert from (batch_size, 4, seq_length) to (batch_size, seq_length)
                original_samples_indices = torch.argmax(original_data, dim=1)
            
            viz_logger = create_visualization_logger(
                num_samples=actual_samples,
                sequence_length=sequence_length,
                num_steps=steps,
                dataset_name=self.dataset_name,
                architecture=architecture,
                split=split,
                save_oracle_mse=False,  # No oracle for cCRE
                device=self.device,
                original_samples=original_samples_indices  # Add original samples for completeness
            )
            print(f"  ↳ Visualization data logging enabled with original samples ({actual_samples} samples)")
            
            # Sample sequences using PC sampler for visualization
            print(f"Sampling sequences with PC sampler ({steps} steps) for visualization...")
            sampled_sequences, target_labels = self.sample_sequences_for_evaluation(
                checkpoint_path, config, dataloader, steps, architecture, show_progress, viz_logger, None
            )
            
            # Save sequences as NPZ if requested
            if save_sequences:
                checkpoint_dir = os.path.dirname(checkpoint_path)
                npz_path = os.path.join(checkpoint_dir, "sample.npz")
                self.save_sequences_as_npz(sampled_sequences, npz_path)
        
        # Create a simple test: generate some random variants and compute scores
        test_variants = [
            {'chrom': '1', 'pos': 1000000, 'ref': 'A', 'alt': 'T'},
            {'chrom': '1', 'pos': 1000001, 'ref': 'C', 'alt': 'G'},
            {'chrom': '2', 'pos': 2000000, 'ref': 'G', 'alt': 'A'},
            {'chrom': '2', 'pos': 2000001, 'ref': 'T', 'alt': 'C'},
        ]
        
        # Evaluate these test variants
        results = self.evaluate_variants(checkpoint_path, config, test_variants, architecture)
        
        # Add evaluation metadata
        results.update({
            'evaluation_type': 'variant_effect_prediction',
            'split': split,
            'sampling_steps': steps,
            'note': 'Test evaluation with dummy variants - replace with real TraitGym data'
        })
        
        # Save visualization data if requested
        if save_visualization_data and 'viz_logger' in locals():
            if viz_output_path is None:
                # Auto-generate visualization output path
                checkpoint_dir = os.path.dirname(checkpoint_path)
                viz_output_path = os.path.join(checkpoint_dir, f"ccre_evaluation_visualization_data.{viz_format}")
            
            viz_logger.save(viz_output_path, viz_format)
            results['visualization_output_path'] = viz_output_path
        
        return results
    
    def get_oracle_predictions_for_viz(self, sequences: torch.Tensor, oracle_model) -> torch.Tensor:
        """
        cCRE-specific oracle predictions for visualization (no oracle available).
        
        Args:
            sequences: One-hot encoded sequences (batch_size, seq_length, 4)
            oracle_model: Not used for cCRE
            
        Returns:
            Zero tensor (no oracle predictions available)
        """
        # cCRE has no oracle model, return zeros
        return torch.zeros(sequences.shape[0], device=self.device)


def load_default_config():
    """Load cCRE default configuration (transformer)."""
    config_file = Path(__file__).parent / 'configs' / 'transformer.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():
    """Main evaluation function using base framework."""
    # Parse arguments using base framework
    parser = parse_base_args()
    parser.add_argument('--variants', type=str, help='Path to variants file (JSON format)')
    parser.add_argument('--genome_fasta', type=str, help='Path to genome FASTA file')
    args = parser.parse_args()
    
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
    evaluator = cCREEvaluator()
    
    # Check if specific variants file is provided
    if args.variants:
        import json
        with open(args.variants, 'r') as f:
            variants = json.load(f)
        
        # Evaluate on provided variants
        metrics = evaluator.evaluate_variants(
            checkpoint_path=args.checkpoint,
            config=config,
            variants=variants,
            architecture=args.architecture,
            genome_fasta_path=args.genome_fasta
        )
    else:
        # Run default evaluation (with dummy variants)
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
            save_sequences=getattr(args, 'save_sequences', False),
            save_visualization_data=getattr(args, 'save_viz_data', False),
            viz_output_path=getattr(args, 'viz_output', None),
            viz_format=getattr(args, 'viz_format', 'hdf5'),
            max_samples=getattr(args, 'max_samples', None)
        )
    
    # Print and save results
    evaluator.print_results(metrics)
    
    output_path = args.output or f"evaluation_results/ccre_{args.architecture}_{args.split}_results.json"
    evaluator.save_results(metrics, output_path)
    
    print(f"\n✓ cCRE evaluation completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())