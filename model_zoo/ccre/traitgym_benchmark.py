#!/usr/bin/env python3
"""
TraitGym Benchmark for cCRE D3 Model

This script implements the TraitGym benchmarking protocol for variant effect prediction
using the cCRE diffusion model. It follows the pattern from the TraitGym example but
uses our D3 model instead of GPN.
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model_zoo.ccre.models import load_trained_model


class Genome:
    """Genome sequence fetcher using s3fs and pyfaidx."""
    
    def __init__(self, path: str = "s3://broad-references/hg38/v0/Homo_sapiens_assembly38.fasta"):
        """
        Initialize genome with FASTA path.
        
        Args:
            path: Path to genome FASTA file (local or s3)
        """
        try:
            import fsspec
            from pyfaidx import Fasta
            self.data = Fasta(fsspec.open(path, anon=True))
            print(f"✓ Loaded genome from {path}")
        except ImportError:
            print("Warning: pyfaidx or fsspec not available. Install with:")
            print("pip install pyfaidx s3fs")
            self.data = None
        except Exception as e:
            print(f"Warning: Could not load genome from {path}: {e}")
            self.data = None

    def __call__(self, chrom: str, start: int, end: int, strand: str = "+") -> str:
        """
        Fetch genome sequence.
        
        Args:
            chrom: Chromosome name
            start: Start position (0-based)
            end: End position (0-based)
            strand: Strand ('+' or '-')
            
        Returns:
            DNA sequence string
        """
        if self.data is None:
            raise ValueError("Genome not initialized. Please initialize with a valid genome path.")
            
        try:
            res = self.data[chrom][start:end]
            if strand == "-":
                res = res.reverse.complement
            return str(res).upper()
        except Exception as e:
            raise ValueError(f"Could not fetch sequence {chrom}:{start}-{end}: {e}")


class cCREVEP(torch.nn.Module):
    """cCRE Variant Effect Predictor using D3 diffusion model."""
    
    def __init__(self, checkpoint_path: str, config_path: str, architecture: str = 'transformer'):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load config
        self.config = OmegaConf.load(config_path)
        
        # Load model
        self.model, self.graph, self.noise = load_trained_model(
            checkpoint_path, self.config, architecture, self.device
        )
        self.model.eval()
        
        print(f"✓ Loaded cCRE D3 model ({architecture}) from {checkpoint_path}")
    
    def tokenize_sequence(self, sequence: str) -> torch.Tensor:
        """
        Convert DNA sequence to token indices.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Token indices tensor
        """
        # DNA to index mapping
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        # Convert to indices
        indices = [base_to_idx.get(base.upper(), 0) for base in sequence]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        Encode DNA sequence using the diffusion model.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Sequence embedding tensor
        """
        # Tokenize sequence
        indices = self.tokenize_sequence(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get model embedding
            if hasattr(self.model, 'transformer'):
                # For transformer models
                try:
                    embedding = self.model.transformer.encode(indices)
                    embedding = embedding.mean(dim=1)  # Pool over sequence length
                except:
                    # Fallback: use forward pass with dummy inputs
                    dummy_sigma = torch.ones(1, device=self.device) * 0.1
                    output = self.model(indices, train=False, sigma=dummy_sigma)
                    embedding = output.mean(dim=1)
            else:
                # For convolutional models
                try:
                    x = F.one_hot(indices, num_classes=4).float().permute(0, 2, 1)
                    embedding = self.model.linear(x)
                    for block in self.model.conv_blocks[:3]:
                        embedding = F.relu(block(embedding))
                    embedding = F.adaptive_avg_pool1d(embedding, 1).squeeze(-1)
                except:
                    # Fallback
                    dummy_sigma = torch.ones(1, device=self.device) * 0.1
                    output = self.model(indices, train=False, sigma=dummy_sigma)
                    embedding = output.mean(dim=1)
        
        return embedding
    
    def forward(self, input_ids_ref: torch.Tensor, input_ids_alt: torch.Tensor) -> torch.Tensor:
        """
        Compute variant effect scores as embedding distance.
        
        Args:
            input_ids_ref: Reference sequence token indices
            input_ids_alt: Alternate sequence token indices
            
        Returns:
            Variant effect scores (Euclidean distance)
        """
        batch_size = input_ids_ref.shape[0]
        scores = []
        
        for i in range(batch_size):
            # Convert indices back to sequences
            ref_seq = ''.join(['ACGT'[idx.item()] for idx in input_ids_ref[i]])
            alt_seq = ''.join(['ACGT'[idx.item()] for idx in input_ids_alt[i]])
            
            # Get embeddings
            ref_embed = self.encode_sequence(ref_seq)
            alt_embed = self.encode_sequence(alt_seq)
            
            # Compute distance
            score = F.pairwise_distance(ref_embed, alt_embed).item()
            scores.append(score)
        
        return torch.tensor(scores, device=self.device)


def tokenize_sequences(sequences: List[str]) -> torch.Tensor:
    """
    Tokenize a list of DNA sequences.
    
    Args:
        sequences: List of DNA sequence strings
        
    Returns:
        Tensor of token indices
    """
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    tokenized = []
    for seq in sequences:
        indices = [base_to_idx.get(base.upper(), 0) for base in seq]
        tokenized.append(indices)
    
    return torch.tensor(tokenized, dtype=torch.long)


def get_tokenized_sequences(variants_df: pd.DataFrame, genome: Genome, window_size: int = 512):
    """
    Get tokenized reference and alternate sequences for variants.
    
    Args:
        variants_df: DataFrame with variant information
        genome: Genome sequence fetcher
        window_size: Sequence window size
        
    Returns:
        Dictionary with tokenized sequences
    """
    chrom = variants_df['chrom'].values
    pos = variants_df['pos'].values - 1  # Convert to 0-based
    ref = variants_df['ref'].values
    alt = variants_df['alt'].values
    
    n = len(chrom)
    start = pos - window_size // 2
    end = pos + window_size // 2
    
    # Fetch sequences
    sequences_ref = []
    sequences_alt = []
    
    for i in tqdm(range(n), desc="Fetching sequences"):
        # Get reference sequence
        seq = genome(f"chr{chrom[i]}", start[i], end[i])
        seq_array = list(seq.upper())
        
        # Verify reference allele matches
        center_pos = window_size // 2
        if center_pos < len(seq_array):
            if seq_array[center_pos] != ref[i].upper():
                print(f"Warning: Reference mismatch at {chrom[i]}:{pos[i]+1} "
                      f"(expected {ref[i]}, got {seq_array[center_pos]})")
        
        # Create reference and alternate sequences
        ref_seq = ''.join(seq_array)
        alt_seq_array = seq_array.copy()
        alt_seq_array[center_pos] = alt[i].upper()
        alt_seq = ''.join(alt_seq_array)
        
        sequences_ref.append(ref_seq)
        sequences_alt.append(alt_seq)
    
    # Tokenize sequences
    input_ids_ref = tokenize_sequences(sequences_ref)
    input_ids_alt = tokenize_sequences(sequences_alt)
    
    return {
        'input_ids_ref': input_ids_ref,
        'input_ids_alt': input_ids_alt
    }


def run_traitgym_benchmark(checkpoint_path: str, config_path: str, 
                          architecture: str = 'transformer',
                          dataset_config: str = "mendelian_traits",
                          output_dir: str = "traitgym_results",
                          batch_size: int = 128) -> Dict[str, Any]:
    """
    Run TraitGym benchmark for cCRE D3 model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to model config
        architecture: Model architecture
        dataset_config: TraitGym dataset configuration
        output_dir: Output directory for results
        batch_size: Batch size for inference
        
    Returns:
        Dictionary with benchmark results
    """
    print("=" * 60)
    print("Running TraitGym Benchmark for cCRE D3 Model")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load TraitGym dataset
    print(f"Loading TraitGym dataset: {dataset_config}")
    try:
        dataset = load_dataset("songlab/TraitGym", dataset_config, split="test")
        print(f"✓ Loaded {len(dataset)} test variants")
    except Exception as e:
        print(f"Error loading TraitGym dataset: {e}")
        print("Make sure you have the datasets library installed: pip install datasets")
        return {'error': 'dataset_loading_failed'}
    
    # Convert to DataFrame
    variants_df = dataset.to_pandas()
    print(f"Dataset shape: {variants_df.shape}")
    print(f"Columns: {list(variants_df.columns)}")
    
    # Initialize genome
    print("Initializing genome...")
    genome = Genome()
    
    # Get tokenized sequences
    print("Tokenizing sequences...")
    window_size = 512
    tokenized_data = get_tokenized_sequences(variants_df, genome, window_size)
    
    # Initialize model
    print("Loading cCRE D3 model...")
    model = cCREVEP(checkpoint_path, config_path, architecture)
    
    # Run inference
    print("Running variant effect prediction...")
    input_ids_ref = tokenized_data['input_ids_ref']
    input_ids_alt = tokenized_data['input_ids_alt']
    
    # Process in batches
    all_scores = []
    num_batches = (len(input_ids_ref) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(input_ids_ref))
        
        batch_ref = input_ids_ref[start_idx:end_idx]
        batch_alt = input_ids_alt[start_idx:end_idx]
        
        scores = model(batch_ref, batch_alt)
        all_scores.append(scores.cpu())
    
    # Concatenate all scores
    all_scores = torch.cat(all_scores, dim=0).numpy()
    
    # Add scores to DataFrame
    variants_df['score'] = all_scores
    
    # Compute metrics
    print("Computing metrics...")
    
    # Global AUPRC
    global_auprc = average_precision_score(variants_df['label'], variants_df['score'])
    print(f"Global AUPRC: {global_auprc:.4f}")
    
    # AUPRC by chromosome
    results_by_chrom = []
    for chrom in variants_df['chrom'].unique():
        chrom_data = variants_df[variants_df['chrom'] == chrom]
        if len(chrom_data) > 0 and len(chrom_data['label'].unique()) > 1:
            chrom_auprc = average_precision_score(chrom_data['label'], chrom_data['score'])
            results_by_chrom.append({
                'chrom': chrom,
                'n': len(chrom_data),
                'AUPRC': chrom_auprc
            })
    
    results_by_chrom_df = pd.DataFrame(results_by_chrom)
    
    # Weighted average AUPRC
    if len(results_by_chrom_df) > 0:
        weights = results_by_chrom_df['n'] / results_by_chrom_df['n'].sum()
        weighted_auprc = (results_by_chrom_df['AUPRC'] * weights).sum()
    else:
        weighted_auprc = 0.0
    
    print(f"Weighted AUPRC: {weighted_auprc:.4f}")
    
    # Save results
    results = {
        'model_checkpoint': checkpoint_path,
        'model_architecture': architecture,
        'dataset_config': dataset_config,
        'global_auprc': global_auprc,
        'weighted_auprc': weighted_auprc,
        'num_variants': len(variants_df),
        'results_by_chrom': results_by_chrom_df.to_dict('records') if len(results_by_chrom_df) > 0 else []
    }
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"ccre_d3_{architecture}_traitgym_results.json")
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save predictions
    predictions_file = os.path.join(output_dir, f"ccre_d3_{architecture}_predictions.csv")
    variants_df[['chrom', 'pos', 'ref', 'alt', 'label', 'score']].to_csv(predictions_file, index=False)
    
    # Create visualization
    try:
        plt.figure(figsize=(10, 6))
        
        # Score distribution by label
        plt.subplot(1, 2, 1)
        sns.histplot(data=variants_df, x="score", hue="label", bins=30, stat="density",
                    common_norm=False, common_bins=True)
        plt.title("Score Distribution by Label")
        plt.xlabel("cCRE D3 Variant Effect Score")
        
        # AUPRC by chromosome
        if len(results_by_chrom_df) > 0:
            plt.subplot(1, 2, 2)
            plt.bar(range(len(results_by_chrom_df)), results_by_chrom_df['AUPRC'])
            plt.xlabel("Chromosome")
            plt.ylabel("AUPRC")
            plt.title("AUPRC by Chromosome")
            plt.xticks(range(len(results_by_chrom_df)), results_by_chrom_df['chrom'], rotation=45)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"ccre_d3_{architecture}_results.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved plot to {plot_file}")
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")
    
    print(f"✓ Results saved to {output_dir}")
    print("=" * 60)
    print("TraitGym Benchmark Completed")
    print("=" * 60)
    
    return results


def main():
    """Main function for running TraitGym benchmark."""
    parser = argparse.ArgumentParser(description="TraitGym benchmark for cCRE D3 model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", required=True, help="Path to model config")
    parser.add_argument("--architecture", default="transformer", choices=["transformer", "convolutional"],
                       help="Model architecture")
    parser.add_argument("--dataset", default="mendelian_traits", help="TraitGym dataset configuration")
    parser.add_argument("--output_dir", default="traitgym_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_traitgym_benchmark(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        architecture=args.architecture,
        dataset_config=args.dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    if 'error' not in results:
        print(f"✓ Benchmark completed successfully!")
        print(f"Global AUPRC: {results['global_auprc']:.4f}")
        print(f"Weighted AUPRC: {results['weighted_auprc']:.4f}")
    else:
        print(f"✗ Benchmark failed: {results['error']}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())