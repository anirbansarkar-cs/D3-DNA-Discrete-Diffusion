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
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """Load cCRE model using dataset-specific model loading."""
        from model_zoo.ccre.models import load_trained_model
        
        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None):
        """Create cCRE dataloader."""
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
        """Get original test data for evaluation."""
        try:
            # Load cCRE data from the 'seqs' key
            with h5py.File(data_path, 'r') as data_file:
                X = torch.tensor(np.array(data_file['seqs']))
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
                              show_progress: bool = False) -> Dict[str, Any]:
        """
        Override base evaluation to provide variant effect prediction instead of SP-MSE.
        
        Since cCRE has no oracle model, we skip oracle-based evaluation and focus on
        the model's ability to distinguish between sequences.
        """
        print(f"cCRE evaluation: Variant effect prediction mode")
        print("Note: Oracle-based evaluation not applicable for unlabeled cCRE data")
        
        # Load model for basic validation
        model, graph, noise = self.load_model(checkpoint_path, config, architecture)
        
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
            'sampling_steps': steps or self.get_sequence_length(config),
            'note': 'Test evaluation with dummy variants - replace with real TraitGym data'
        })
        
        return results


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
            show_progress=args.show_progress
        )
    
    # Print and save results
    evaluator.print_results(metrics)
    
    output_path = args.output or f"evaluation_results/ccre_{args.architecture}_{args.split}_results.json"
    evaluator.save_results(metrics, output_path)
    
    print(f"\n✓ cCRE evaluation completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())