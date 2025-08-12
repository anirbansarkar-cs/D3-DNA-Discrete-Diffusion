#!/usr/bin/env python3
"""
CAGI5 Variant Effect Prediction Script for LentIMPRA

This script implements variant effect prediction using D3 diffusion models on the CAGI5 dataset.
It supports two prediction methods:
1. Cosine similarity of sequence representations
2. Score matrix differences at mutation positions

The script processes H5 sequence data and CSV metadata, generates predictions, and saves
comprehensive results in H5 format with hierarchical evaluation metrics.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import h5py
from omegaconf import OmegaConf
from tqdm import tqdm
from scipy.stats import pearsonr

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import D3 components
from model_zoo.lentimpra.models import load_trained_model
from utils.utils import get_score_fn


class CAGI5VEPProcessor:
    """CAGI5 Variant Effect Prediction processor using D3 diffusion models."""
    
    def __init__(self, checkpoint_path: str, config_path: str, device: str = 'cuda'):
        """
        Initialize the VEP processor.
        
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
        self.ref_sequences = None
        self.alt_sequences = None
        self.metadata_df = None
        self.mutation_positions = None
        self.ref_nucleotides = None
        self.alt_nucleotides = None
        
        # Results storage
        self.results = {}
        
    def load_model(self):
        """Load the trained D3 model and associated components."""
        print(f"Loading D3 model from {self.checkpoint_path}")
        self.model, self.graph, self.noise = load_trained_model(
            self.checkpoint_path, self.config, 'transformer', self.device
        )
        self.model.eval()
        print("✓ Model loaded successfully")
        
    def load_cagi5_data(self, h5_path: str, csv_path: str):
        """
        Load CAGI5 sequence data from H5 and metadata from CSV.
        
        Args:
            h5_path: Path to CAGI5 H5 file with 'ref' and 'alt' keys
            csv_path: Path to CAGI5 metadata CSV file
        """
        print(f"Loading CAGI5 data from {h5_path} and {csv_path}")
        
        # Load H5 sequence data
        with h5py.File(h5_path, 'r') as f:
            self.ref_sequences = torch.tensor(np.array(f['ref']), dtype=torch.float32)  # (N, 230, 4)
            self.alt_sequences = torch.tensor(np.array(f['alt']), dtype=torch.float32)  # (N, 230, 4)
        
        # Load CSV metadata
        self.metadata_df = pd.read_csv(csv_path)
        
        # Verify data consistency
        if len(self.ref_sequences) != len(self.alt_sequences):
            raise ValueError("Reference and alternative sequences must have same length")
        if len(self.ref_sequences) != len(self.metadata_df):
            raise ValueError("Sequence data and metadata must have same length")
            
        # Extract mutation information
        self._extract_mutation_info()
        
        print(f"✓ Loaded {len(self.ref_sequences)} variant sequences")
        print(f"  - Genes: {sorted(self.metadata_df['gene'].unique())}")
        print(f"  - Cell lines: {sorted(self.metadata_df['cell_line'].unique())}")
        
    def _extract_mutation_info(self):
        """Extract mutation positions and nucleotide mappings from sequences."""
        n_sequences = len(self.ref_sequences)
        self.mutation_positions = []
        self.ref_nucleotides = []
        self.alt_nucleotides = []
        
        nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        for i in range(n_sequences):
            ref_seq = self.ref_sequences[i]  # (230, 4)
            alt_seq = self.alt_sequences[i]  # (230, 4)
            
            # Find mutation position by comparing sequences
            diff_mask = (ref_seq != alt_seq).any(dim=1)  # (230,)
            mutation_positions = torch.where(diff_mask)[0]
            
            if len(mutation_positions) == 0:
                # No mutation found - this shouldn't happen in CAGI5 data
                print(f"Warning: No mutation found for sequence {i}")
                self.mutation_positions.append(115)  # Use center position as fallback
                self.ref_nucleotides.append(0)
                self.alt_nucleotides.append(1)
            else:
                # Use first mutation position if multiple found
                mut_pos = mutation_positions[0].item()
                self.mutation_positions.append(mut_pos)
                
                # Get nucleotide indices at mutation position
                ref_nuc = torch.argmax(ref_seq[mut_pos]).item()
                alt_nuc = torch.argmax(alt_seq[mut_pos]).item()
                self.ref_nucleotides.append(ref_nuc)
                self.alt_nucleotides.append(alt_nuc)
        
        self.mutation_positions = torch.tensor(self.mutation_positions, dtype=torch.long)
        self.ref_nucleotides = torch.tensor(self.ref_nucleotides, dtype=torch.long)
        self.alt_nucleotides = torch.tensor(self.alt_nucleotides, dtype=torch.long)
        
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
        
        for i, t in enumerate(timesteps):
            sigma, _ = self.noise(t.unsqueeze(0))
            sigma_values.append(sigma.item())
            
        sigma_values = torch.tensor(sigma_values, device=self.device)
        
        # Use 5th to last sigma as default (or last if less than 5 steps)
        default_idx = max(0, len(sigma_values) - 5)
        
        return sigma_values, default_idx
        
    def predict_cosine_similarity(self, sigma_values: torch.Tensor, default_sigma_idx: int, 
                                batch_size: int, save_intermediates: bool = False) -> Dict[str, Any]:
        """
        Predict variant effects using cosine similarity of sequence representations.
        
        Args:
            sigma_values: Noise schedule values
            default_sigma_idx: Index of default sigma to use
            batch_size: Batch size for processing
            save_intermediates: Whether to save results for all noise steps
            
        Returns:
            Dictionary containing cosine similarity results
        """
        print("Computing cosine similarity predictions...")
        
        n_sequences = len(self.ref_sequences)
        score_fn = get_score_fn(self.model, train=False, sampling=False)
        
        results = {
            'default_step': {},
            'all_steps': {} if save_intermediates else None
        }
        
        # Determine which steps to process
        if save_intermediates:
            step_indices = list(range(len(sigma_values)))
        else:
            step_indices = [default_sigma_idx]
        
        for step_idx in tqdm(step_indices, desc="Processing noise steps"):
            sigma = sigma_values[step_idx]
            step_results = {
                'ref_representations': [],
                'alt_representations': [], 
                'cosine_scores': [],
                'noise_level': sigma.item()
            }
            
            # Process sequences in batches
            for batch_start in range(0, n_sequences, batch_size):
                batch_end = min(batch_start + batch_size, n_sequences)
                
                # Get batch sequences
                ref_batch = self.ref_sequences[batch_start:batch_end].to(self.device)
                alt_batch = self.alt_sequences[batch_start:batch_end].to(self.device)
                
                # Convert from one-hot to token indices for model input
                ref_tokens = torch.argmax(ref_batch, dim=-1)  # (batch, 230)
                alt_tokens = torch.argmax(alt_batch, dim=-1)  # (batch, 230)
                
                # Create sigma tensor for batch
                batch_sigma = sigma.repeat(ref_tokens.shape[0]).to(self.device)
                
                # Get representations by feeding through model with noise
                with torch.no_grad():
                    # Add noise to sequences using graph.sample_transition
                    # ref_perturbed = self.graph.sample_transition(ref_tokens, batch_sigma[:, None])
                    # alt_perturbed = self.graph.sample_transition(alt_tokens, batch_sigma[:, None])
                    
                    # Get representations from layer 11 (following sample.py pattern)
                    targets = None  # for unconditional generation
                    _, ref_repr = self.model(ref_tokens, targets, train=False, sigma=batch_sigma, layer_idx=11)
                    _, alt_repr = self.model(alt_tokens, targets, train=False, sigma=batch_sigma, layer_idx=11)
                    
                    # Aggregate representations (mean pooling across sequence length)
                    ref_repr_pooled = ref_repr.mean(dim=1)  # (batch, hidden_dim)
                    alt_repr_pooled = alt_repr.mean(dim=1)  # (batch, hidden_dim)
                    
                    # Compute cosine similarity
                    cosine_sim = F.cosine_similarity(ref_repr_pooled, alt_repr_pooled, dim=1)  # (batch,)
                    
                step_results['ref_representations'].append(ref_repr_pooled.cpu())
                step_results['alt_representations'].append(alt_repr_pooled.cpu())
                step_results['cosine_scores'].append(cosine_sim.cpu())
            
            # Concatenate batch results
            step_results['ref_representations'] = torch.cat(step_results['ref_representations'], dim=0)
            step_results['alt_representations'] = torch.cat(step_results['alt_representations'], dim=0)
            step_results['cosine_scores'] = torch.cat(step_results['cosine_scores'], dim=0)
            
            # Store results
            if step_idx == default_sigma_idx:
                results['default_step'] = step_results
            if save_intermediates:
                results['all_steps'][f'step_{step_idx}'] = step_results
        
        return results
    
    def predict_score_matrix(self, sigma_values: torch.Tensor, default_sigma_idx: int,
                           batch_size: int, save_intermediates: bool = False) -> Dict[str, Any]:
        """
        Predict variant effects using score matrix differences at mutation positions.
        
        Args:
            sigma_values: Noise schedule values
            default_sigma_idx: Index of default sigma to use
            batch_size: Batch size for processing
            save_intermediates: Whether to save results for all noise steps
            
        Returns:
            Dictionary containing score matrix results
        """
        print("Computing score matrix predictions...")
        
        n_sequences = len(self.ref_sequences)
        sampling_score_fn = get_score_fn(self.model, train=False, sampling=True)  # Use sampling=True for score matrices
        
        results = {
            'default_step': {},
            'all_steps': {} if save_intermediates else None
        }
        
        # Determine which steps to process
        if save_intermediates:
            step_indices = list(range(len(sigma_values)))
        else:
            step_indices = [default_sigma_idx]
        
        for step_idx in tqdm(step_indices, desc="Processing noise steps"):
            sigma = sigma_values[step_idx]
            step_results = {
                'ref_score_matrices': [],
                'alt_score_matrices': [],
                'ref_mutation_scores': [],
                'alt_mutation_scores': [],
                'score_differences': [],
                'noise_level': sigma.item()
            }
            
            # Process sequences in batches
            for batch_start in range(0, n_sequences, batch_size):
                batch_end = min(batch_start + batch_size, n_sequences)
                
                # Get batch data
                ref_batch = self.ref_sequences[batch_start:batch_end].to(self.device)
                alt_batch = self.alt_sequences[batch_start:batch_end].to(self.device)
                mut_pos_batch = self.mutation_positions[batch_start:batch_end]
                ref_nuc_batch = self.ref_nucleotides[batch_start:batch_end]
                alt_nuc_batch = self.alt_nucleotides[batch_start:batch_end]
                
                # Convert from one-hot to token indices for model input
                ref_tokens = torch.argmax(ref_batch, dim=-1)  # (batch, 230)
                alt_tokens = torch.argmax(alt_batch, dim=-1)  # (batch, 230)
                
                # Create sigma tensor for batch
                batch_sigma = sigma.repeat(ref_tokens.shape[0]).to(self.device)
                
                # Get score matrices using sampling score function
                with torch.no_grad():
                    # Use None for targets (unconditional generation)
                    targets = None
                    ref_scores = sampling_score_fn(ref_tokens, batch_sigma, targets)  # (batch, 230, 4)
                    alt_scores = sampling_score_fn(alt_tokens, batch_sigma, targets)  # (batch, 230, 4)
                    
                    # Extract scores at mutation positions
                    batch_ref_mut_scores = []
                    batch_alt_mut_scores = []
                    batch_score_diffs = []
                    
                    for i in range(len(ref_tokens)):
                        mut_pos = mut_pos_batch[i].item()
                        ref_nuc = ref_nuc_batch[i].item()
                        alt_nuc = alt_nuc_batch[i].item()
                        
                        # Get scores at mutation position
                        ref_mut_score = ref_scores[i, mut_pos, ref_nuc].item()
                        alt_mut_score = alt_scores[i, mut_pos, alt_nuc].item()
                        score_diff = alt_mut_score - ref_mut_score
                        
                        batch_ref_mut_scores.append(ref_mut_score)
                        batch_alt_mut_scores.append(alt_mut_score)
                        batch_score_diffs.append(score_diff)
                    
                step_results['ref_score_matrices'].append(ref_scores.cpu())
                step_results['alt_score_matrices'].append(alt_scores.cpu())
                step_results['ref_mutation_scores'].extend(batch_ref_mut_scores)
                step_results['alt_mutation_scores'].extend(batch_alt_mut_scores)
                step_results['score_differences'].extend(batch_score_diffs)
            
            # Concatenate and tensorize results
            step_results['ref_score_matrices'] = torch.cat(step_results['ref_score_matrices'], dim=0)
            step_results['alt_score_matrices'] = torch.cat(step_results['alt_score_matrices'], dim=0)
            step_results['ref_mutation_scores'] = torch.tensor(step_results['ref_mutation_scores'])
            step_results['alt_mutation_scores'] = torch.tensor(step_results['alt_mutation_scores'])
            step_results['score_differences'] = torch.tensor(step_results['score_differences'])
            
            # Store results
            if step_idx == default_sigma_idx:
                results['default_step'] = step_results
            if save_intermediates:
                results['all_steps'][f'step_{step_idx}'] = step_results
        
        return results
    
    def evaluate_predictions(self, cosine_results: Optional[Dict] = None, 
                           score_matrix_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth using Pearson correlation.
        
        Args:
            cosine_results: Cosine similarity prediction results
            score_matrix_results: Score matrix prediction results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print("Evaluating predictions...")
        
        evaluation_results = {
            'overall_metrics': {},
            'per_gene_results': {},
            'per_cell_line_results': {}
        }
        
        # Get ground truth scores
        ground_truth = torch.tensor(self.metadata_df['score'].values, dtype=torch.float32)
        genes = self.metadata_df['gene'].values
        cell_lines = self.metadata_df['cell_line'].values
        
        unique_genes = sorted(self.metadata_df['gene'].unique())
        unique_cell_lines = sorted(self.metadata_df['cell_line'].unique())
        
        # Evaluate each method
        for method_name, results in [('cosine_method', cosine_results), ('score_matrix_method', score_matrix_results)]:
            if results is None:
                continue
                
            # Get predictions from default step
            predictions = results['default_step']['cosine_scores'] if method_name == 'cosine_method' else results['default_step']['score_differences']
            
            # Overall correlation
            overall_r, overall_p = pearsonr(predictions.numpy(), ground_truth.numpy())
            evaluation_results['overall_metrics'][method_name] = {
                'pearson_r': overall_r,
                'p_value': overall_p
            }
            
            # Per-gene evaluation
            gene_results = {
                'gene_names': unique_genes,
                'pearson_correlations': [],
                'p_values': [],
                'sample_counts': []
            }
            
            for gene in unique_genes:
                gene_mask = genes == gene
                if gene_mask.sum() > 1:  # Need at least 2 samples for correlation
                    gene_predictions = predictions[gene_mask].numpy()
                    gene_ground_truth = ground_truth[gene_mask].numpy()
                    gene_r, gene_p = pearsonr(gene_predictions, gene_ground_truth)
                    gene_results['pearson_correlations'].append(gene_r)
                    gene_results['p_values'].append(gene_p)
                    gene_results['sample_counts'].append(len(gene_predictions))
                else:
                    gene_results['pearson_correlations'].append(np.nan)
                    gene_results['p_values'].append(np.nan)
                    gene_results['sample_counts'].append(gene_mask.sum().item())
            
            evaluation_results['per_gene_results'][method_name] = gene_results
            
            # Per-cell-line evaluation
            cell_line_results = {
                'cell_line_names': unique_cell_lines,
                'pearson_correlations': [],
                'p_values': [],
                'sample_counts': []
            }
            
            for cell_line in unique_cell_lines:
                cell_line_mask = cell_lines == cell_line
                if cell_line_mask.sum() > 1:
                    cl_predictions = predictions[cell_line_mask].numpy()
                    cl_ground_truth = ground_truth[cell_line_mask].numpy()
                    cl_r, cl_p = pearsonr(cl_predictions, cl_ground_truth)
                    cell_line_results['pearson_correlations'].append(cl_r)
                    cell_line_results['p_values'].append(cl_p)
                    cell_line_results['sample_counts'].append(len(cl_predictions))
                else:
                    cell_line_results['pearson_correlations'].append(np.nan)
                    cell_line_results['p_values'].append(np.nan)
                    cell_line_results['sample_counts'].append(cell_line_mask.sum().item())
            
            evaluation_results['per_cell_line_results'][method_name] = cell_line_results
        
        # Compute CAGI5-specific metrics (K562 vs HepG2 aggregation)
        self._compute_cagi5_metrics(evaluation_results)
        
        return evaluation_results
    
    def _compute_cagi5_metrics(self, evaluation_results: Dict):
        """
        Compute CAGI5-specific evaluation metrics.
        
        For K562: Direct Pearson r (PKLR gene only)
        For HepG2: Average Pearson r across LDLR, F9, SORT1 genes
        """
        for method_name in ['cosine_method', 'score_matrix_method']:
            if method_name not in evaluation_results['per_gene_results']:
                continue
                
            gene_data = evaluation_results['per_gene_results'][method_name]
            gene_names = gene_data['gene_names']
            gene_correlations = gene_data['pearson_correlations']
            
            # K562 metric (PKLR gene)
            try:
                pklr_idx = gene_names.index('PKLR')
                k562_pearson_r = gene_correlations[pklr_idx]
            except (ValueError, IndexError):
                k562_pearson_r = np.nan
            
            # HepG2 metric (average of LDLR, F9, SORT1)
            hepg2_genes = ['LDLR', 'F9', 'SORT1']
            hepg2_correlations = []
            for gene in hepg2_genes:
                try:
                    gene_idx = gene_names.index(gene)
                    if not np.isnan(gene_correlations[gene_idx]):
                        hepg2_correlations.append(gene_correlations[gene_idx])
                except (ValueError, IndexError):
                    continue
            
            hepg2_average_r = np.mean(hepg2_correlations) if hepg2_correlations else np.nan
            
            # Store CAGI5-specific metrics
            evaluation_results['overall_metrics'][method_name].update({
                'k562_pearson_r': k562_pearson_r,
                'hepg2_average_pearson_r': hepg2_average_r
            })
    
    def save_results_h5(self, output_path: str, cosine_results: Optional[Dict] = None,
                       score_matrix_results: Optional[Dict] = None, evaluation_results: Dict = None,
                       sigma_values: torch.Tensor = None, default_sigma_idx: int = None):
        """
        Save all results to H5 file using the planned format.
        
        Args:
            output_path: Path to output H5 file
            cosine_results: Cosine similarity results
            score_matrix_results: Score matrix results
            evaluation_results: Evaluation metrics
            sigma_values: Noise schedule values
            default_sigma_idx: Default sigma index
        """
        print(f"Saving results to {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Metadata group
            meta_group = f.create_group('metadata')
            meta_group.create_dataset('dataset_info', data='CAGI5 Variant Effect Prediction')
            meta_group.create_dataset('model_checkpoint', data=self.checkpoint_path)
            meta_group.create_dataset('config_used', data=OmegaConf.to_yaml(self.config))
            meta_group.create_dataset('num_sequences', data=len(self.ref_sequences))
            meta_group.create_dataset('sequence_length', data=230)
            meta_group.create_dataset('timestamp', data=datetime.now().isoformat())
            
            methods_used = []
            if cosine_results is not None:
                methods_used.append('cosine_similarity')
            if score_matrix_results is not None:
                methods_used.append('score_matrix')
            meta_group.create_dataset('methods_used', data=methods_used)
            
            unique_genes = sorted(self.metadata_df['gene'].unique())
            unique_cell_lines = sorted(self.metadata_df['cell_line'].unique())
            meta_group.create_dataset('genes', data=unique_genes)
            meta_group.create_dataset('cell_lines', data=unique_cell_lines)
            
            # Input data group
            input_group = f.create_group('input_data')
            input_group.create_dataset('ref_sequences', data=self.ref_sequences.numpy())
            input_group.create_dataset('alt_sequences', data=self.alt_sequences.numpy())
            input_group.create_dataset('mutation_positions', data=self.mutation_positions.numpy())
            input_group.create_dataset('ref_nucleotides', data=self.ref_nucleotides.numpy())
            input_group.create_dataset('alt_nucleotides', data=self.alt_nucleotides.numpy())
            input_group.create_dataset('ground_truth_scores', data=self.metadata_df['score'].values)
            input_group.create_dataset('genes', data=self.metadata_df['gene'].values.astype('S'))
            input_group.create_dataset('cell_lines', data=self.metadata_df['cell_line'].values.astype('S'))
            input_group.create_dataset('identifiers', data=self.metadata_df['identifier'].values.astype('S'))
            
            # Noise schedule group
            if sigma_values is not None:
                noise_group = f.create_group('noise_schedule')
                noise_group.create_dataset('sigma_values', data=sigma_values.cpu().numpy())
                noise_group.create_dataset('sigma_indices', data=np.arange(len(sigma_values)))
                if default_sigma_idx is not None:
                    noise_group.create_dataset('default_sigma_idx', data=default_sigma_idx)
            
            # Save cosine similarity results
            if cosine_results is not None:
                self._save_cosine_results_h5(f, cosine_results)
            
            # Save score matrix results
            if score_matrix_results is not None:
                self._save_score_matrix_results_h5(f, score_matrix_results)
            
            # Save evaluation results
            if evaluation_results is not None:
                self._save_evaluation_results_h5(f, evaluation_results)
        
        print(f"✓ Results saved to {output_path}")
    
    def _save_cosine_results_h5(self, f: h5py.File, cosine_results: Dict):
        """Save cosine similarity results to H5 file."""
        cosine_group = f.create_group('method_cosine_similarity')
        
        # Default step
        default_data = cosine_results['default_step']
        default_group = cosine_group.create_group('default_step')
        default_group.create_dataset('ref_representations', data=default_data['ref_representations'].float().numpy())
        default_group.create_dataset('alt_representations', data=default_data['alt_representations'].float().numpy())
        default_group.create_dataset('cosine_scores', data=default_data['cosine_scores'].float().numpy())
        default_group.create_dataset('noise_level', data=default_data['noise_level'])
        
        # All steps (if available)
        if cosine_results['all_steps'] is not None:
            all_steps_group = cosine_group.create_group('all_steps')
            for step_name, step_data in cosine_results['all_steps'].items():
                step_group = all_steps_group.create_group(step_name)
                step_group.create_dataset('ref_representations', data=step_data['ref_representations'].float().numpy())
                step_group.create_dataset('alt_representations', data=step_data['alt_representations'].float().numpy())
                step_group.create_dataset('cosine_scores', data=step_data['cosine_scores'].float().numpy())
                step_group.create_dataset('noise_level', data=step_data['noise_level'])
    
    def _save_score_matrix_results_h5(self, f: h5py.File, score_matrix_results: Dict):
        """Save score matrix results to H5 file."""
        score_group = f.create_group('method_score_matrix')
        
        # Default step
        default_data = score_matrix_results['default_step']
        default_group = score_group.create_group('default_step')
        default_group.create_dataset('ref_score_matrices', data=default_data['ref_score_matrices'].float().numpy())
        default_group.create_dataset('alt_score_matrices', data=default_data['alt_score_matrices'].float().numpy())
        default_group.create_dataset('ref_mutation_scores', data=default_data['ref_mutation_scores'].float().numpy())
        default_group.create_dataset('alt_mutation_scores', data=default_data['alt_mutation_scores'].float().numpy())
        default_group.create_dataset('score_differences', data=default_data['score_differences'].float().numpy())
        default_group.create_dataset('noise_level', data=default_data['noise_level'])
        
        # All steps (if available)
        if score_matrix_results['all_steps'] is not None:
            all_steps_group = score_group.create_group('all_steps')
            for step_name, step_data in score_matrix_results['all_steps'].items():
                step_group = all_steps_group.create_group(step_name)
                step_group.create_dataset('ref_score_matrices', data=step_data['ref_score_matrices'].float().numpy())
                step_group.create_dataset('alt_score_matrices', data=step_data['alt_score_matrices'].float().numpy())
                step_group.create_dataset('ref_mutation_scores', data=step_data['ref_mutation_scores'].float().numpy())
                step_group.create_dataset('alt_mutation_scores', data=step_data['alt_mutation_scores'].float().numpy())
                step_group.create_dataset('score_differences', data=step_data['score_differences'].float().numpy())
                step_group.create_dataset('noise_level', data=step_data['noise_level'])
    
    def _save_evaluation_results_h5(self, f: h5py.File, evaluation_results: Dict):
        """Save evaluation results to H5 file."""
        eval_group = f.create_group('evaluation_results')
        
        # Overall metrics
        overall_group = eval_group.create_group('overall_metrics')
        for method_name, metrics in evaluation_results['overall_metrics'].items():
            method_group = overall_group.create_group(method_name)
            for metric_name, value in metrics.items():
                method_group.create_dataset(metric_name, data=value if not np.isnan(value) else -999.0)
        
        # Per-gene results
        if 'per_gene_results' in evaluation_results:
            gene_group = eval_group.create_group('per_gene_results')
            for method_name, gene_data in evaluation_results['per_gene_results'].items():
                method_group = gene_group.create_group(method_name)
                method_group.create_dataset('gene_names', data=[g.encode() for g in gene_data['gene_names']])
                method_group.create_dataset('pearson_correlations', 
                                          data=[r if not np.isnan(r) else -999.0 for r in gene_data['pearson_correlations']])
                method_group.create_dataset('p_values',
                                          data=[p if not np.isnan(p) else -999.0 for p in gene_data['p_values']])
                method_group.create_dataset('sample_counts', data=gene_data['sample_counts'])
        
        # Per-cell-line results
        if 'per_cell_line_results' in evaluation_results:
            cell_line_group = eval_group.create_group('per_cell_line_results')
            for method_name, cl_data in evaluation_results['per_cell_line_results'].items():
                method_group = cell_line_group.create_group(method_name)
                method_group.create_dataset('cell_line_names', data=[c.encode() for c in cl_data['cell_line_names']])
                method_group.create_dataset('pearson_correlations',
                                          data=[r if not np.isnan(r) else -999.0 for r in cl_data['pearson_correlations']])
                method_group.create_dataset('p_values',
                                          data=[p if not np.isnan(p) else -999.0 for p in cl_data['p_values']])
                method_group.create_dataset('sample_counts', data=cl_data['sample_counts'])


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CAGI5 Variant Effect Prediction with D3')
    
    # Required arguments
    parser.add_argument('--checkpoint', required=True, help='Path to D3 model checkpoint')
    parser.add_argument('--config', required=True, help='Path to model configuration file')
    parser.add_argument('--cagi5_h5', required=True, help='Path to CAGI5 H5 sequence file')
    parser.add_argument('--cagi5_csv', required=True, help='Path to CAGI5 metadata CSV file')
    
    # Optional arguments
    parser.add_argument('--method', choices=['cosine', 'score_matrix', 'both'], default='both',
                       help='Prediction method to use')
    parser.add_argument('--steps', type=int, default=230,
                       help='Number of noise steps (default: sequence length)')
    parser.add_argument('--save_intermediates', action='store_true',
                       help='Save intermediate results for all noise steps')
    parser.add_argument('--output_h5', default='cagi5_vep_results.h5',
                       help='Output H5 file path')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                       help='Device for computation')
    
    return parser.parse_args()


def print_results_summary(evaluation_results: Dict):
    """Print a summary of evaluation results."""
    print("\n" + "="*60)
    print("CAGI5 VARIANT EFFECT PREDICTION RESULTS")
    print("="*60)
    
    for method_name, metrics in evaluation_results['overall_metrics'].items():
        print(f"\n{method_name.upper().replace('_', ' ')}:")
        print("-" * 40)
        print(f"Overall Pearson r: {metrics['pearson_r']:.4f} (p={metrics['p_value']:.2e})")
        print(f"K562 Pearson r (PKLR): {metrics['k562_pearson_r']:.4f}")
        print(f"HepG2 Average Pearson r: {metrics['hepg2_average_pearson_r']:.4f}")
        
        # Per-gene breakdown
        if method_name in evaluation_results['per_gene_results']:
            gene_data = evaluation_results['per_gene_results'][method_name]
            print("\nPer-gene results:")
            for i, gene in enumerate(gene_data['gene_names']):
                r = gene_data['pearson_correlations'][i]
                n = gene_data['sample_counts'][i]
                if not np.isnan(r):
                    print(f"  {gene}: r={r:.4f} (n={n})")
                else:
                    print(f"  {gene}: insufficient data (n={n})")


def load_results_from_h5(h5_path: str) -> Dict[str, Any]:
    """
    Utility function to load results from saved H5 file.
    
    Args:
        h5_path: Path to H5 results file
        
    Returns:
        Dictionary containing loaded results
    """
    results = {}
    
    with h5py.File(h5_path, 'r') as f:
        # Load metadata
        results['metadata'] = {}
        if 'metadata' in f:
            meta_group = f['metadata']
            for key in meta_group.keys():
                data = meta_group[key][()]
                if isinstance(data, bytes):
                    data = data.decode()
                results['metadata'][key] = data
        
        # Load evaluation results
        if 'evaluation_results' in f:
            eval_group = f['evaluation_results']
            results['evaluation_results'] = {}
            
            # Overall metrics
            if 'overall_metrics' in eval_group:
                results['evaluation_results']['overall_metrics'] = {}
                overall_group = eval_group['overall_metrics']
                for method_name in overall_group.keys():
                    method_group = overall_group[method_name]
                    results['evaluation_results']['overall_metrics'][method_name] = {}
                    for metric_name in method_group.keys():
                        value = method_group[metric_name][()]
                        if value == -999.0:  # Placeholder for NaN
                            value = np.nan
                        results['evaluation_results']['overall_metrics'][method_name][metric_name] = value
    
    return results


def main():
    """Main execution function."""
    args = parse_args()
    
    # Validate inputs
    for path in [args.checkpoint, args.config, args.cagi5_h5, args.cagi5_csv]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            return 1
    
    # Initialize processor
    print("Initializing CAGI5 VEP Processor...")
    processor = CAGI5VEPProcessor(args.checkpoint, args.config, args.device)
    
    # Load model and data
    processor.load_model()
    processor.load_cagi5_data(args.cagi5_h5, args.cagi5_csv)
    
    print(f"\n🚀 Starting variant effect prediction...")
    print(f"   Method: {args.method}")
    print(f"   Steps: {args.steps}")
    print(f"   Save intermediates: {args.save_intermediates}")
    print(f"   Output: {args.output_h5}")
    
    # Generate noise schedule
    print("\n📊 Generating noise schedule...")
    sigma_values, default_sigma_idx = processor.generate_noise_schedule(args.steps)
    print(f"Generated {len(sigma_values)} noise levels, using step {default_sigma_idx} as default (σ={sigma_values[default_sigma_idx]:.4f})")
    
    # Initialize results storage
    cosine_results = None
    score_matrix_results = None
    
    # Run prediction methods
    if args.method in ['cosine', 'both']:
        print(f"\n🧮 Running cosine similarity method...")
        cosine_results = processor.predict_cosine_similarity(
            sigma_values, default_sigma_idx, args.batch_size, args.save_intermediates
        )
        print(f"✓ Cosine similarity predictions completed")
    
    if args.method in ['score_matrix', 'both']:
        print(f"\n📈 Running score matrix method...")
        score_matrix_results = processor.predict_score_matrix(
            sigma_values, default_sigma_idx, args.batch_size, args.save_intermediates
        )
        print(f"✓ Score matrix predictions completed")
    
    # Evaluate predictions
    print(f"\n📏 Evaluating predictions...")
    evaluation_results = processor.evaluate_predictions(cosine_results, score_matrix_results)
    print(f"✓ Evaluation completed")
    
    # Print results summary
    print_results_summary(evaluation_results)
    
    # Save results to H5 file
    print(f"\n💾 Saving results...")
    processor.save_results_h5(
        args.output_h5, cosine_results, score_matrix_results, evaluation_results,
        sigma_values, default_sigma_idx
    )
    
    print(f"\n✅ CAGI5 Variant Effect Prediction completed successfully!")
    print(f"   Results saved to: {args.output_h5}")
    
    # Print final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Processed {len(processor.ref_sequences)} variant sequences")
    print(f"Genes: {', '.join(sorted(processor.metadata_df['gene'].unique()))}")
    print(f"Cell lines: {', '.join(sorted(processor.metadata_df['cell_line'].unique()))}")
    print(f"Methods: {args.method}")
    print(f"Noise steps: {args.steps}")
    if args.save_intermediates:
        print(f"Saved intermediate results for all {len(sigma_values)} noise steps")
    print(f"Results file: {args.output_h5}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())