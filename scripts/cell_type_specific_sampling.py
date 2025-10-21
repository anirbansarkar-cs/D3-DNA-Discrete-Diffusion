#!/usr/bin/env python3
"""
Cell-Type-Specific Activity Prediction Experiment

This script implements targeted sampling experiments for both DeepSTARR and LentiMPRA models.
For each experiment, we:
1. Generate 500 sequences conditioned on specific activity tuples
2. Use oracle models to predict activities for the generated sequences
3. Save sequences and oracle-predicted activities for analysis

DeepSTARR: Activity tuple format (dev, hk)
LentiMPRA: Activity tuple format (k562, hepg2, wtc11)
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass, asdict

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import dataset-specific modules
from scripts import sampling
from model_zoo.deepstarr.models import load_trained_model as load_deepstarr_model
from model_zoo.lentimpra.models import load_trained_model as load_lentimpra_model


# =============================================================================
# Data Classes for Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    experiment_name: str
    model: str  # 'deepstarr' or 'lentimpra'
    d3_checkpoint: str  # Path to D3 model checkpoint
    oracle_checkpoint: str  # Path to oracle model checkpoint
    config_path: str  # Path to D3 model config
    num_samples_per_tuple: int = 500
    num_steps: int = None  # Sampling steps (defaults to sequence length)
    batch_size: int = 128
    architecture: str = 'transformer'
    device: str = 'cuda'
    output_dir: str = './outputs/cell_type_specific'


# =============================================================================
# DeepSTARR Oracle Model (from sampling_small_data.py)
# =============================================================================

class DeepSTARR_Oracle(nn.Module):
    """DeepSTARR oracle model for activity prediction."""

    def __init__(self, output_dim=2, d=256):
        super().__init__()

        self.activation = nn.ReLU()
        self.dropout4 = nn.Dropout(0.4)
        self.flatten = nn.Flatten()

        # Layer 1 (convolutional)
        self.conv1_filters = nn.Parameter(torch.zeros(d, 4, 7))
        nn.init.kaiming_normal_(self.conv1_filters)
        self.batchnorm1 = nn.BatchNorm1d(d)
        self.activation1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(2)

        # Layer 2 (convolutional)
        self.conv2_filters = nn.Parameter(torch.zeros(60, d, 3))
        nn.init.kaiming_normal_(self.conv2_filters)
        self.batchnorm2 = nn.BatchNorm1d(60)
        self.maxpool2 = nn.MaxPool1d(2)

        # Layer 3 (convolutional)
        self.conv3_filters = nn.Parameter(torch.zeros(60, 60, 5))
        nn.init.kaiming_normal_(self.conv3_filters)
        self.batchnorm3 = nn.BatchNorm1d(60)
        self.maxpool3 = nn.MaxPool1d(2)

        # Layer 4 (convolutional)
        self.conv4_filters = nn.Parameter(torch.zeros(120, 60, 3))
        nn.init.kaiming_normal_(self.conv4_filters)
        self.batchnorm4 = nn.BatchNorm1d(120)
        self.maxpool4 = nn.MaxPool1d(2)

        # Layer 5 (fully connected)
        self.fc5 = nn.LazyLinear(256, bias=True)
        self.batchnorm5 = nn.BatchNorm1d(256)

        # Layer 6 (fully connected)
        self.fc6 = nn.Linear(256, 256, bias=True)
        self.batchnorm6 = nn.BatchNorm1d(256)

        # Output layer
        self.fc7 = nn.Linear(256, output_dim)

    def forward(self, x):
        # Layer 1
        cnn = torch.conv1d(x, self.conv1_filters, stride=1, padding="same")
        cnn = self.batchnorm1(cnn)
        cnn = self.activation1(cnn)
        cnn = self.maxpool1(cnn)

        # Layer 2
        cnn = torch.conv1d(cnn, self.conv2_filters, stride=1, padding="same")
        cnn = self.batchnorm2(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool2(cnn)

        # Layer 3
        cnn = torch.conv1d(cnn, self.conv3_filters, stride=1, padding="same")
        cnn = self.batchnorm3(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool3(cnn)

        # Layer 4
        cnn = torch.conv1d(cnn, self.conv4_filters, stride=1, padding="same")
        cnn = self.batchnorm4(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool4(cnn)

        # Layer 5
        cnn = self.flatten(cnn)
        cnn = self.fc5(cnn)
        cnn = self.batchnorm5(cnn)
        cnn = self.activation(cnn)
        cnn = self.dropout4(cnn)

        # Layer 6
        cnn = self.fc6(cnn)
        cnn = self.batchnorm6(cnn)
        cnn = self.activation(cnn)
        cnn = self.dropout4(cnn)

        # Output layer
        y_pred = self.fc7(cnn)

        return y_pred


def load_deepstarr_oracle(checkpoint_path: str, device: str = 'cuda') -> DeepSTARR_Oracle:
    """Load DeepSTARR oracle model from checkpoint."""
    print(f"Loading DeepSTARR oracle from: {checkpoint_path}")

    model = DeepSTARR_Oracle(output_dim=2)
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        # PyTorch Lightning format
        state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('model.'):
                new_key = key.replace('model.', '')
                state_dict[new_key] = value
            else:
                state_dict[key] = value
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    print("✓ DeepSTARR oracle loaded successfully")
    return model


# =============================================================================
# LentiMPRA Oracle Model Loading
# =============================================================================

def load_lentimpra_oracle(checkpoint_path: str, config_path: str, device: str = 'cuda'):
    """Load LentiMPRA oracle model from checkpoint.

    Note: LentiMPRA uses MPRALegNet as the oracle model.
    This requires the mpralegnet.py implementation.
    """
    print(f"Loading LentiMPRA oracle from: {checkpoint_path}")

    try:
        from model_zoo.lentimpra.mpralegnet import MPRALegNet

        # Load configuration to get signal_dim
        config = OmegaConf.load(config_path)
        signal_dim = config.dataset.get('signal_dim', 3)  # Default 3 for K562, HepG2, WTC11

        # Create model
        model = MPRALegNet(signal_dim=signal_dim, seq_length=230)
        model.to(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.eval()
        print("✓ LentiMPRA oracle loaded successfully")
        return model

    except ImportError as e:
        print(f"Error loading LentiMPRA oracle: {e}")
        print("Note: Ensure mpralegnet.py is available in model_zoo/lentimpra/")
        raise


# =============================================================================
# Sampling Functions
# =============================================================================

def sample_deepstarr_sequences(
    d3_model,
    graph,
    noise,
    activity_tuple: Tuple[float, float],
    num_samples: int,
    num_steps: int,
    batch_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample DeepSTARR sequences conditioned on (dev, hk) activity tuple.

    Returns:
        sequences_onehot: (N, L, 4) one-hot encoded sequences
        conditioning_labels: (N, 2) activity labels used for sampling
    """
    sequence_length = 249

    # Create conditioning labels (all samples use the same activity tuple)
    dev_activity, hk_activity = activity_tuple
    conditioning_labels = torch.tensor([[dev_activity, hk_activity]], dtype=torch.float32)
    conditioning_labels = conditioning_labels.repeat(num_samples, 1).to(device)

    # Sample in batches
    sampled_sequences = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"  Sampling {num_samples} sequences conditioned on Dev={dev_activity:.2f}, HK={hk_activity:.2f}")

    for i in tqdm(range(num_batches), desc="  Sampling batches"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx

        batch_labels = conditioning_labels[start_idx:end_idx]

        # Create PC sampler
        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (current_batch_size, sequence_length), 'analytic',
            num_steps, device=device
        )

        # Sample sequences (returns token indices)
        with torch.amp.autocast('cuda', enabled=True):
            sample_indices = sampling_fn(d3_model, batch_labels)

        # Convert to one-hot: (B, L) -> (B, L, 4)
        sample_onehot = F.one_hot(sample_indices, num_classes=4).float()
        sampled_sequences.append(sample_onehot.cpu())

        # Clear GPU cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()

    # Concatenate all batches
    sequences_onehot = torch.cat(sampled_sequences, dim=0)

    return sequences_onehot, conditioning_labels.cpu()


def sample_lentimpra_sequences(
    d3_model,
    graph,
    noise,
    activity_tuple: Tuple[float, float, float],
    num_samples: int,
    num_steps: int,
    batch_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample LentiMPRA sequences conditioned on (k562, hepg2, wtc11) activity tuple.

    Returns:
        sequences_onehot: (N, L, 4) one-hot encoded sequences
        conditioning_labels: (N, 3) activity labels used for sampling
    """
    sequence_length = 230

    # Create conditioning labels (all samples use the same activity tuple)
    k562_activity, hepg2_activity, wtc11_activity = activity_tuple
    conditioning_labels = torch.tensor([[k562_activity, hepg2_activity, wtc11_activity]], dtype=torch.float32)
    conditioning_labels = conditioning_labels.repeat(num_samples, 1).to(device)

    # Sample in batches
    sampled_sequences = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"  Sampling {num_samples} sequences conditioned on K562={k562_activity:.2f}, HepG2={hepg2_activity:.2f}, WTC11={wtc11_activity:.2f}")

    for i in tqdm(range(num_batches), desc="  Sampling batches"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx

        batch_labels = conditioning_labels[start_idx:end_idx]

        # Create PC sampler
        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (current_batch_size, sequence_length), 'analytic',
            num_steps, device=device
        )

        # Sample sequences (returns token indices)
        with torch.amp.autocast('cuda', enabled=True):
            sample_indices = sampling_fn(d3_model, batch_labels)

        # Convert to one-hot: (B, L) -> (B, L, 4)
        sample_onehot = F.one_hot(sample_indices, num_classes=4).float()
        sampled_sequences.append(sample_onehot.cpu())

        # Clear GPU cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()

    # Concatenate all batches
    sequences_onehot = torch.cat(sampled_sequences, dim=0)

    return sequences_onehot, conditioning_labels.cpu()


# =============================================================================
# Oracle Prediction Functions
# =============================================================================

def predict_with_deepstarr_oracle(
    oracle_model: nn.Module,
    sequences_onehot: torch.Tensor,
    batch_size: int,
    device: torch.device
) -> torch.Tensor:
    """
    Predict activities using DeepSTARR oracle model.

    Args:
        oracle_model: DeepSTARR oracle model
        sequences_onehot: (N, L, 4) one-hot encoded sequences
        batch_size: Batch size for prediction
        device: Device to use

    Returns:
        predictions: (N, 2) predicted activities [dev, hk]
    """
    oracle_model.eval()
    predictions = []

    num_batches = (len(sequences_onehot) + batch_size - 1) // batch_size

    print(f"  Predicting activities for {len(sequences_onehot)} sequences...")

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="  Oracle prediction"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(sequences_onehot))

            # Get batch and convert to (B, 4, L) format for DeepSTARR
            batch_sequences = sequences_onehot[start_idx:end_idx]
            batch_sequences = batch_sequences.permute(0, 2, 1).to(device)

            # Predict
            batch_predictions = oracle_model(batch_sequences)
            predictions.append(batch_predictions.cpu())

    return torch.cat(predictions, dim=0)


def predict_with_lentimpra_oracle(
    oracle_model: nn.Module,
    sequences_onehot: torch.Tensor,
    batch_size: int,
    device: torch.device
) -> torch.Tensor:
    """
    Predict activities using LentiMPRA oracle model.

    Args:
        oracle_model: LentiMPRA oracle model (MPRALegNet)
        sequences_onehot: (N, L, 4) one-hot encoded sequences
        batch_size: Batch size for prediction
        device: Device to use

    Returns:
        predictions: (N, 3) predicted activities [k562, hepg2, wtc11]
    """
    oracle_model.eval()
    predictions = []

    num_batches = (len(sequences_onehot) + batch_size - 1) // batch_size

    print(f"  Predicting activities for {len(sequences_onehot)} sequences...")

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="  Oracle prediction"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(sequences_onehot))

            # Get batch and convert to (B, 4, L) format for LentiMPRA oracle
            batch_sequences = sequences_onehot[start_idx:end_idx]
            batch_sequences = batch_sequences.permute(0, 2, 1).to(device)

            # Predict
            batch_predictions = oracle_model(batch_sequences)
            predictions.append(batch_predictions.cpu())

    return torch.cat(predictions, dim=0)


# =============================================================================
# Experiment Runner
# =============================================================================

def run_single_tuple_experiment(config: ExperimentConfig, activity_tuple: Tuple[float, ...],
                                tuple_idx: int, samples_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Run experiment for a single activity tuple.

    Args:
        config: Experiment configuration
        activity_tuple: Single activity tuple to condition on
        tuple_idx: Index of this tuple (for naming output files)

    Returns:
        Dictionary containing experiment results
    """
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print(f"CELL-TYPE-SPECIFIC EXPERIMENT: {config.experiment_name}")
    print(f"TUPLE {tuple_idx}: {activity_tuple}")
    print("=" * 80)
    print(f"Model: {config.model}")
    print(f"D3 Checkpoint: {config.d3_checkpoint}")
    print(f"Oracle Checkpoint: {config.oracle_checkpoint}")
    print(f"Samples: {config.num_samples_per_tuple}")
    print(f"Device: {device}")
    print("=" * 80)

    # Create output directory
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if pre-existing samples are provided
    if samples_file is not None:
        print(f"\n[1/2] Loading pre-existing samples from: {samples_file}")

        # Load sequences from H5 file
        try:
            with h5py.File(samples_file, 'r') as f:
                # Load sequences (prefer one-hot format)
                if 'sequences_onehot' in f:
                    sequences_onehot = torch.tensor(np.array(f['sequences_onehot']), dtype=torch.float32)
                    print(f"  Loaded sequences_onehot: {sequences_onehot.shape}")
                elif 'sequences_indices' in f:
                    # Convert indices to one-hot
                    sequences_indices = torch.tensor(np.array(f['sequences_indices']), dtype=torch.long)
                    sequences_onehot = F.one_hot(sequences_indices, num_classes=4).float()
                    print(f"  Loaded sequences_indices and converted to one-hot: {sequences_onehot.shape}")
                else:
                    raise ValueError("H5 file must contain either 'sequences_onehot' or 'sequences_indices'")

                # Load or create conditioning labels
                if 'conditioning_labels' in f:
                    conditioning_labels = torch.tensor(np.array(f['conditioning_labels']), dtype=torch.float32)
                    print(f"  Loaded conditioning_labels: {conditioning_labels.shape}")
                else:
                    # Create conditioning labels from activity tuple
                    num_samples = len(sequences_onehot)
                    conditioning_labels = torch.tensor([activity_tuple], dtype=torch.float32).repeat(num_samples, 1)
                    print(f"  Created conditioning_labels from activity tuple: {conditioning_labels.shape}")

                # Validate dimensions
                if config.model == 'deepstarr':
                    expected_length = 249
                    expected_signal_dim = 2
                elif config.model == 'lentimpra':
                    expected_length = 230
                    expected_signal_dim = 3

                if sequences_onehot.shape[1] != expected_length:
                    raise ValueError(f"Sequence length mismatch: expected {expected_length}, got {sequences_onehot.shape[1]}")

                if sequences_onehot.shape[2] != 4:
                    raise ValueError(f"Sequence encoding mismatch: expected 4 (one-hot), got {sequences_onehot.shape[2]}")

                if conditioning_labels.shape[1] != expected_signal_dim:
                    raise ValueError(f"Signal dimension mismatch: expected {expected_signal_dim}, got {conditioning_labels.shape[1]}")

                print(f"  ✓ Loaded {len(sequences_onehot)} sequences successfully")

        except Exception as e:
            print(f"  Error loading samples file: {e}")
            raise

        # Load oracle model only (skip D3 model)
        print(f"\n[2/2] Loading oracle model...")
        if config.model == 'deepstarr':
            oracle_model = load_deepstarr_oracle(config.oracle_checkpoint, str(device))
        elif config.model == 'lentimpra':
            oracle_model = load_lentimpra_oracle(
                config.oracle_checkpoint, config.config_path, str(device)
            )

    else:
        # Original workflow: Load D3 model and sample sequences
        print(f"\n[1/3] Loading D3 model and configuration...")
        d3_config = OmegaConf.load(config.config_path)

        # Set default num_steps based on model
        if config.num_steps is None:
            config.num_steps = 249 if config.model == 'deepstarr' else 230

        # Load D3 model
        if config.model == 'deepstarr':
            d3_model, graph, noise = load_deepstarr_model(
                config.d3_checkpoint, d3_config, config.architecture, str(device)
            )
        elif config.model == 'lentimpra':
            d3_model, graph, noise = load_lentimpra_model(
                config.d3_checkpoint, d3_config, config.architecture, str(device)
            )
        else:
            raise ValueError(f"Unknown model: {config.model}")

        d3_model.eval()

        # Load oracle model
        print("\n[2/3] Loading oracle model...")
        if config.model == 'deepstarr':
            oracle_model = load_deepstarr_oracle(config.oracle_checkpoint, str(device))
        elif config.model == 'lentimpra':
            oracle_model = load_lentimpra_oracle(
                config.oracle_checkpoint, config.config_path, str(device)
            )

        # Sample sequences
        print(f"\n[3/3] Sampling sequences for activity tuple: {activity_tuple}")
        if config.model == 'deepstarr':
            sequences_onehot, conditioning_labels = sample_deepstarr_sequences(
                d3_model, graph, noise, activity_tuple,
                config.num_samples_per_tuple, config.num_steps,
                config.batch_size, device
            )
        elif config.model == 'lentimpra':
            sequences_onehot, conditioning_labels = sample_lentimpra_sequences(
                d3_model, graph, noise, activity_tuple,
                config.num_samples_per_tuple, config.num_steps,
                config.batch_size, device
            )

    # Predict activities with oracle
    print(f"\nPredicting activities with oracle model...")
    if config.model == 'deepstarr':
        oracle_predictions = predict_with_deepstarr_oracle(
            oracle_model, sequences_onehot, config.batch_size, device
        )
    elif config.model == 'lentimpra':
        oracle_predictions = predict_with_lentimpra_oracle(
            oracle_model, sequences_onehot, config.batch_size, device
        )

    # Save results for this tuple
    tuple_output_file = output_dir / f"tuple_{tuple_idx}_samples.h5"

    print(f"  Saving results to: {tuple_output_file}")
    with h5py.File(tuple_output_file, 'w') as f:
        # Save sequences (both one-hot and indices)
        f.create_dataset('sequences_onehot', data=sequences_onehot.numpy(),
                       compression='gzip', compression_opts=9)

        # Convert to indices for convenience
        sequences_indices = torch.argmax(sequences_onehot, dim=-1)
        f.create_dataset('sequences_indices', data=sequences_indices.numpy(),
                       compression='gzip', compression_opts=9)

        # Save conditioning labels (target activities)
        f.create_dataset('conditioning_labels', data=conditioning_labels.numpy())

        # Save oracle predictions
        f.create_dataset('oracle_predictions', data=oracle_predictions.numpy())

        # Metadata
        f.attrs['activity_tuple'] = activity_tuple
        f.attrs['tuple_index'] = tuple_idx
        f.attrs['num_samples'] = len(sequences_onehot)
        f.attrs['model'] = config.model
        f.attrs['d3_checkpoint'] = config.d3_checkpoint
        f.attrs['oracle_checkpoint'] = config.oracle_checkpoint
        f.attrs['num_steps'] = config.num_steps if config.num_steps else (249 if config.model == 'deepstarr' else 230)
        f.attrs['used_preexisting_samples'] = samples_file is not None
        if samples_file is not None:
            f.attrs['source_samples_file'] = samples_file

    # Compute statistics
    mean_prediction = oracle_predictions.mean(dim=0).numpy()
    std_prediction = oracle_predictions.std(dim=0).numpy()

    result = {
        'tuple_index': tuple_idx,
        'activity_tuple': activity_tuple,
        'num_samples': len(sequences_onehot),
        'mean_oracle_prediction': mean_prediction.tolist(),
        'std_oracle_prediction': std_prediction.tolist(),
        'output_file': str(tuple_output_file),
        'used_preexisting_samples': samples_file is not None
    }

    if samples_file is not None:
        result['source_samples_file'] = samples_file

    # Print summary
    if config.model == 'deepstarr':
        print(f"\n  Oracle predictions - Dev: {mean_prediction[0]:.3f}±{std_prediction[0]:.3f}, "
              f"HK: {mean_prediction[1]:.3f}±{std_prediction[1]:.3f}")
    elif config.model == 'lentimpra':
        print(f"\n  Oracle predictions - K562: {mean_prediction[0]:.3f}±{std_prediction[0]:.3f}, "
              f"HepG2: {mean_prediction[1]:.3f}±{std_prediction[1]:.3f}, "
              f"WTC11: {mean_prediction[2]:.3f}±{std_prediction[2]:.3f}")

    print(f"  ✓ Saved to: {tuple_output_file.name}")

    # Save individual result file
    result_file = output_dir / f"tuple_{tuple_idx}_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✓ Experiment completed successfully!")
    print(f"Results saved to: {tuple_output_file}")
    print(f"Summary: {result_file}")
    print("=" * 80)

    return result


# =============================================================================
# Main Function and CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Cell-Type-Specific Activity Prediction Experiment (Single Tuple)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DeepSTARR experiment: Sample sequences and predict with oracle
  python cell_type_specific_sampling.py \\
    --model deepstarr \\
    --d3_checkpoint path/to/d3_deepstarr.ckpt \\
    --oracle_checkpoint path/to/oracle_deepstarr.ckpt \\
    --config model_zoo/deepstarr/configs/transformer.yaml \\
    --activity_tuple "(2.0, 2.0)" \\
    --tuple_idx 0 \\
    --experiment_name deepstarr_high_low_conditions

  # LentiMPRA experiment: Sample sequences and predict with oracle
  python cell_type_specific_sampling.py \\
    --model lentimpra \\
    --d3_checkpoint path/to/d3_lentimpra.ckpt \\
    --oracle_checkpoint path/to/oracle_lentimpra.ckpt \\
    --config model_zoo/lentimpra/configs/transformer.yaml \\
    --activity_tuple "(2.0, 2.0, 2.0)" \\
    --tuple_idx 0 \\
    --experiment_name lentimpra_cell_specific

  # Use pre-existing samples (skip sampling, only oracle prediction)
  python cell_type_specific_sampling.py \\
    --model deepstarr \\
    --oracle_checkpoint path/to/oracle_deepstarr.ckpt \\
    --config model_zoo/deepstarr/configs/transformer.yaml \\
    --activity_tuple "(2.0, 2.0)" \\
    --tuple_idx 0 \\
    --experiment_name oracle_only \\
    --samples path/to/existing_samples.h5

Note: Use SLURM job arrays to run multiple tuples in parallel.
        """
    )

    parser.add_argument('--model', required=True, choices=['deepstarr', 'lentimpra'],
                       help='Model type: deepstarr or lentimpra')
    parser.add_argument('--d3_checkpoint', required=True,
                       help='Path to trained D3 model checkpoint')
    parser.add_argument('--oracle_checkpoint', required=True,
                       help='Path to oracle model checkpoint')
    parser.add_argument('--config', required=True,
                       help='Path to D3 model config file')
    parser.add_argument('--activity_tuple', required=True,
                       help='Single activity tuple as string, e.g., "(2.0, 2.0)" or "(2.0, 2.0, 2.0)"')
    parser.add_argument('--tuple_idx', type=int, required=True,
                       help='Index for this tuple (used for output file naming)')
    parser.add_argument('--experiment_name', required=True,
                       help='Name for this experiment (used for output directory)')
    parser.add_argument('--num_samples_per_tuple', type=int, default=500,
                       help='Number of samples to generate (default: 500)')
    parser.add_argument('--num_steps', type=int, default=None,
                       help='Number of sampling steps (default: sequence length)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for sampling and prediction (default: 128)')
    parser.add_argument('--architecture', choices=['transformer', 'convolutional'],
                       default='transformer', help='D3 model architecture (default: transformer)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--output_dir', default='./outputs/cell_type_specific',
                       help='Base output directory (default: ./outputs/cell_type_specific)')
    parser.add_argument('--samples', type=str, default=None,
                       help='Path to pre-existing samples H5 file to skip sampling and only run oracle prediction')

    args = parser.parse_args()

    # Parse activity tuple from string
    try:
        import ast
        activity_tuple = ast.literal_eval(args.activity_tuple)

        # Convert single value to tuple if needed
        if not isinstance(activity_tuple, tuple):
            if isinstance(activity_tuple, (list, int, float)):
                activity_tuple = tuple(activity_tuple) if isinstance(activity_tuple, list) else (activity_tuple,)
            else:
                raise ValueError(f"Invalid activity_tuple format: {activity_tuple}")

        # Validate tuple dimensions
        if args.model == 'deepstarr':
            if len(activity_tuple) != 2:
                raise ValueError(f"DeepSTARR requires 2D tuples (dev, hk), got: {activity_tuple}")
        elif args.model == 'lentimpra':
            if len(activity_tuple) != 3:
                raise ValueError(f"LentiMPRA requires 3D tuples (k562, hepg2, wtc11), got: {activity_tuple}")

    except Exception as e:
        print(f"Error parsing activity_tuple: {e}")
        print("Expected format: \"(val1, val2)\" for DeepSTARR or \"(val1, val2, val3)\" for LentiMPRA")
        return 1

    # Create experiment configuration
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        model=args.model,
        d3_checkpoint=args.d3_checkpoint,
        oracle_checkpoint=args.oracle_checkpoint,
        config_path=args.config,
        num_samples_per_tuple=args.num_samples_per_tuple,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        architecture=args.architecture,
        device=args.device,
        output_dir=args.output_dir
    )

    # Validate samples file if provided
    if args.samples:
        samples_path = Path(args.samples)
        if not samples_path.exists():
            print(f"Error: Samples file not found: {args.samples}")
            return 1
        if not samples_path.suffix == '.h5':
            print(f"Error: Samples file must be in H5 format (.h5)")
            return 1

    # Run experiment for single tuple
    try:
        result = run_single_tuple_experiment(config, activity_tuple, args.tuple_idx, args.samples)
        return 0
    except Exception as e:
        print(f"\nError running experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
