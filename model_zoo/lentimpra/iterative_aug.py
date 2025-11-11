#!/usr/bin/env python3
"""
LentIMPRA iterative augmentation sampling for small data experiments (Multi-class).

Iteratively augments the train set by sampling sequences and adding them:
- Iteration 0: Use train set as baseline (oracles train on train set, evaluate on test set)
- Iteration 1+: Train set + accumulated sampled sequences (conditioned on test set labels)
At each iteration, multiple mpralegnet oracles are trained on the augmented dataset
and evaluated on the test set to compute Pearson R for 3 cell types (K562, HepG2, WTC11).

Each iteration adds a constant number of generated samples equal to the original train set size.
Sequences are conditioned on test set labels (repeating test labels as necessary to meet target size).

Adapted from DeepSTARR version to use LentIMPRA multi-class data and mpralegnet oracle.
Signal dimension: 3 (k562, hepg2, wtc11)
Architecture: transformer_multi_class
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import h5py
import json
from typing import Tuple, Dict, Any, Optional, List
import random
import tempfile
import shutil
from scipy import stats
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# EvoAug imports
try:
    from evoaug.augment import (
        RandomDeletion, RandomRC, RandomInsertion,
        RandomTranslocation, RandomMutation, RandomNoise
    )
    from evoaug.evoaug import RobustLoader
    EVOAUG_AVAILABLE = True
except Exception:
    EVOAUG_AVAILABLE = False

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts import sampling
from model_zoo.lentimpra.data import get_lentimpra_datasets
from model_zoo.lentimpra.models import load_trained_model

# Import mpralegnet components
from model_zoo.lentimpra.mpralegnet import (
    LegNet, LitModel, TrainingConfig,
    HDF5Dataset, set_global_seed, initialize_weights
)

# Ensure deterministic CuBLAS workspace config is set before any CUDA operations
if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def check_existing_oracle_checkpoints(oracle_dir: str, dataset_stem: str, num_models: int = 3) -> Dict[str, Any]:
    """Check for existing oracle checkpoints and return their status.

    Args:
        oracle_dir: Directory containing oracle checkpoints
        dataset_stem: The dataset filename stem (e.g., iteration_1_dataset)
        num_models: Expected number of oracle models

    Returns:
        Dictionary with checkpoint status information
    """
    checkpoint_status = {
        'found_checkpoints': [],
        'missing_checkpoints': [],
        'all_exist': False,
        'num_found': 0,
        'paths': {}
    }
    os.makedirs(oracle_dir, exist_ok=True)
    for model_idx in range(num_models):
        ckpt_name = f"oracle_mpralegnet_{dataset_stem}_m{model_idx}.ckpt"
        ckpt_path = os.path.join(oracle_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            checkpoint_status['found_checkpoints'].append(model_idx)
            checkpoint_status['num_found'] += 1
            checkpoint_status['paths'][model_idx] = ckpt_path
        else:
            checkpoint_status['missing_checkpoints'].append(model_idx)
            checkpoint_status['paths'][model_idx] = ckpt_path
    checkpoint_status['all_exist'] = (checkpoint_status['num_found'] == num_models)
    return checkpoint_status


# =============================================================================
# PyTorch Lightning Wrapper for mpralegnet
# =============================================================================

class PL_LegNet(pl.LightningModule):
    """PyTorch Lightning wrapper for LegNet model."""

    def __init__(self,
                 config: TrainingConfig,
                 batch_size: int = 1024,
                 train_max_epochs: int = 25,
                 patience: int = 10,
                 input_h5_file: str = 'lentimpra_data.h5'):
        super().__init__()
        self.save_hyperparameters()

        # Model configuration
        self.config = config
        self.model = config.get_model()
        self.model.apply(initialize_weights)
        self.name = 'LegNet'

        # Training configuration
        self.batch_size = batch_size
        self.train_max_epochs = train_max_epochs
        self.patience = patience

        # Data configuration
        self.input_h5_file = input_h5_file

    def training_step(self, batch, batch_idx):
        """Training step for one batch."""
        self.model.train()
        inputs, labels = batch
        loss_fn = nn.MSELoss()
        outputs = self.model(inputs)
        loss = loss_fn(outputs, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True,
                prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.max_lr / 25,
            weight_decay=self.config.weight_decay
        )

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.max_lr,
            three_phase=False,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            cycle_momentum=False
        )

        return [optimizer], [{
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "cycle_lr"
        }]

    def validation_step(self, batch, batch_idx):
        """Validation step for one batch."""
        self.model.eval()
        inputs, labels = batch
        loss_fn = nn.MSELoss()
        outputs = self.model(inputs)
        loss = loss_fn(outputs, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True,
                prog_bar=True, logger=True, sync_dist=True)

        # Calculate and log Pearson correlations for each cell type
        out_cpu = outputs.detach().cpu().numpy()
        lab_cpu = labels.detach().cpu().numpy()

        # Ensure outputs and labels are 2D (batch_size, 3)
        if out_cpu.ndim == 1:
            out_cpu = out_cpu.reshape(-1, 1)
        if lab_cpu.ndim == 1:
            lab_cpu = lab_cpu.reshape(-1, 1)

        # Calculate Pearson R for each cell type (dim 0=k562, dim 1=hepg2, dim 2=wtc11)
        cell_types = ['k562', 'hepg2', 'wtc11']
        pearson_values = []
        for i, cell_type in enumerate(cell_types):
            if i < out_cpu.shape[1]:
                # Handle potential NaN values
                valid_mask = ~(np.isnan(out_cpu[:, i]) | np.isnan(lab_cpu[:, i]))
                if valid_mask.sum() > 1:
                    pearson_r = np.corrcoef(out_cpu[valid_mask, i], lab_cpu[valid_mask, i])[0, 1]
                else:
                    pearson_r = 0.0
                pearson_values.append(pearson_r)
                self.log(f"val_pearson_{cell_type}", pearson_r, on_step=False, on_epoch=True,
                        prog_bar=False, logger=True)

        # Log average Pearson correlation
        avg_pearson = np.mean(pearson_values) if pearson_values else 0.0
        self.log("val_pearson", avg_pearson, on_step=False, on_epoch=True,
                prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        """Test step for one batch."""
        self.model.eval()
        inputs, labels = batch
        loss_fn = nn.MSELoss()
        outputs = self.model(inputs)
        loss = loss_fn(outputs, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True,
                prog_bar=True, logger=True)

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def predict_custom(self, X, keepgrad=False):
        """Custom prediction function with batch processing."""
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(
            X, batch_size=self.batch_size, shuffle=False
        )
        preds_list = []
        for x in tqdm(dataloader, total=len(dataloader)):
            x = x.to(self.device)
            pred = self.model(x)
            if not keepgrad:
                pred = pred.detach().cpu()
            preds_list.append(pred)
        if not preds_list:
            return torch.empty(0)
        if keepgrad:
            return torch.cat(preds_list, dim=0).to(self.device)
        return torch.cat(preds_list, dim=0)


def training_with_PL(dataset_path: str,
                     train_max_epochs: int = 25,
                     batch_size: int = 1024,
                     patience: int = 10,
                     verbose: bool = False,
                     seed: int = 42,
                     out_dir: Optional[str] = None,
                     filename: Optional[str] = None,
                     use_augmentation: bool = False) -> Dict[str, Any]:
    """Train mpralegnet model using PyTorch Lightning.

    For every oracle model trained, the dataloader can optionally use augmentation
    (reverse complement + shift) when use_augmentation is True.

    Args:
        dataset_path: Path to H5 file containing training data
        train_max_epochs: Maximum training epochs
        batch_size: Training batch size
        patience: Early stopping patience
        verbose: Whether to print verbose output
        seed: Random seed
        out_dir: Directory to save checkpoints
        filename: Base filename (without extension) for the checkpoint
        use_augmentation: If True, enable reverse + shift augmentation for training

    Returns:
        Dictionary containing trained model and metrics
    """
    try:
        # Set random seeds for reproducibility
        set_global_seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load data from H5 file
        with h5py.File(dataset_path, 'r') as f:
            # LentIMPRA format: onehot_train, y_train
            if 'onehot_train' in f:
                X_train = torch.tensor(np.array(f['onehot_train']), dtype=torch.float32)
                Y_train = torch.tensor(np.array(f['y_train']), dtype=torch.float32).squeeze()
            else:
                # Fallback to X_train, Y_train format
                X_train = torch.tensor(np.array(f['X_train']), dtype=torch.float32)
                Y_train = torch.tensor(np.array(f['Y_train']), dtype=torch.float32).squeeze()

            # Check if we have validation data, otherwise split from training
            if 'onehot_val' in f or 'onehot_valid' in f:
                val_key = 'onehot_val' if 'onehot_val' in f else 'onehot_valid'
                y_val_key = 'y_val' if 'y_val' in f else 'y_valid'
                X_val = torch.tensor(np.array(f[val_key]), dtype=torch.float32)
                Y_val = torch.tensor(np.array(f[y_val_key]), dtype=torch.float32).squeeze()
            elif 'X_val' in f:
                X_val = torch.tensor(np.array(f['X_val']), dtype=torch.float32)
                Y_val = torch.tensor(np.array(f['Y_val']), dtype=torch.float32).squeeze()
            else:
                # Split 10% for validation
                n_val = int(0.1 * len(X_train))
                indices = torch.randperm(len(X_train))
                val_indices = indices[:n_val]
                train_indices = indices[n_val:]

                X_val = X_train[val_indices]
                Y_val = Y_train[val_indices]
                X_train = X_train[train_indices]
                Y_train = Y_train[train_indices]

        # Convert to (N, 4, L) one-hot format for LegNet model
        if X_train.dim() == 2:  # (N, L) indices format
            X_train = F.one_hot(X_train.long(), num_classes=4).float().permute(0, 2, 1)
            X_val = F.one_hot(X_val.long(), num_classes=4).float().permute(0, 2, 1)
        elif X_train.dim() == 3 and X_train.shape[-1] == 4:  # (N, L, 4) format
            X_train = X_train.permute(0, 2, 1)  # Convert to (N, 4, L)
            X_val = X_val.permute(0, 2, 1)
        elif X_train.dim() == 3 and X_train.shape[1] == 4:  # already (N, 4, L)
            pass
        else:
            raise ValueError(f"Unexpected X_train shape: {X_train.shape}")

        # Ensure Y is 2D (N, 3) for multi-class (k562, hepg2, wtc11)
        if Y_train.dim() == 1:
            raise ValueError(f"Expected 2D labels for multi-class, but got 1D with shape {Y_train.shape}")
        if Y_train.shape[-1] != 3:
            raise ValueError(f"Expected 3 outputs for multi-class (k562, hepg2, wtc11), but got {Y_train.shape[-1]}")
        if Y_val.dim() == 1:
            raise ValueError(f"Expected 2D labels for multi-class, but got 1D with shape {Y_val.shape}")
        if Y_val.shape[-1] != 3:
            raise ValueError(f"Expected 3 outputs for multi-class (k562, hepg2, wtc11), but got {Y_val.shape[-1]}")

        if verbose:
            print(f"Training data shape: {X_train.shape}")
            print(f"Training labels shape: {Y_train.shape}")
            print(f"Validation data shape: {X_val.shape}")
            print(f"Validation labels shape: {Y_val.shape}")

        # Create training configuration for multi-class (k562, hepg2, wtc11)
        config = TrainingConfig(
            reverse_augment=use_augmentation,
            use_shift=use_augmentation,
            max_lr=0.01,
            weight_decay=0.1,
            epoch_num=train_max_epochs,
            train_batch_size=batch_size,
            valid_batch_size=batch_size,
            seed=seed,
            output_dim=3  # Multi-class output for k562, hepg2, wtc11
        )

        # Initialize model
        model = PL_LegNet(
            config=config,
            input_h5_file=dataset_path,
            batch_size=batch_size,
            train_max_epochs=train_max_epochs,
            patience=patience
        )

        # Setup data loaders
        num_workers = 4
        train_dataset = TensorDataset(X_train, Y_train)
        val_dataset = TensorDataset(X_val, Y_val)

        # Use EvoAug RobustLoader for training if requested and available
        if use_augmentation and EVOAUG_AVAILABLE:
            # Augmentation list matching DeepSTARR (lines 446-451)
            augment_list = [
                RandomDeletion(delete_min=0, delete_max=20),
                RandomTranslocation(shift_min=0, shift_max=20),
                RandomMutation(mut_frac=0.05),
                RandomNoise(noise_mean=0, noise_std=0.2),
            ]

            # Training loader with RobustLoader
            train_dataloader = RobustLoader(
                base_dataset=train_dataset,
                augment_list=augment_list,
                max_augs_per_seq=2,
                hard_aug=True,
                batch_size=batch_size,
                sampler=None,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=True,
            )

            # Validation loader: standard DataLoader (no augmentation)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            if use_augmentation and not EVOAUG_AVAILABLE:
                print("Warning: EvoAug requested but not available. Falling back to standard dataloaders.")

            # Standard DataLoaders
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )

            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

        # Setup callbacks
        dataset_name = Path(dataset_path).stem
        ckptfile = filename if filename is not None else f"oracle_mpralegnet_{dataset_name}"
        ckpt_dir = out_dir if out_dir is not None else "./"

        callback_ckpt = ModelCheckpoint(
            monitor='val_pearson',
            mode='max',
            save_top_k=1,
            save_weights_only=True,
            dirpath=ckpt_dir,
            filename=ckptfile,
        )

        early_stop_callback = EarlyStopping(
            monitor='val_pearson',
            min_delta=0.001,
            patience=patience,
            verbose=False,
            mode='max'
        )

        # Setup trainer with minimal logging
        trainer = pl.Trainer(
            accelerator='cuda' if torch.cuda.is_available() else 'cpu',
            devices=1,
            max_epochs=train_max_epochs,
            logger=False,  # Disable logging for cleaner output
            callbacks=[callback_ckpt, early_stop_callback],
            deterministic=True,
            enable_progress_bar=verbose,
            enable_model_summary=False,
            precision='16-mixed' if torch.cuda.is_available() else 32,
        )

        # Train
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # Finetune on the original data without augmentations (if augmentation was used)
        if use_augmentation and EVOAUG_AVAILABLE:
            # Build non-augmented dataloaders for finetuning
            finetune_train_loader = DataLoader(
                TensorDataset(X_train, Y_train),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )
            finetune_val_loader = DataLoader(
                TensorDataset(X_val, Y_val),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            # Finetune callbacks (reuse same filename so the best finetuned weights overwrite prior best)
            finetune_ckpt = ModelCheckpoint(
                monitor='val_pearson',
                mode='max',
                save_top_k=1,
                save_weights_only=True,
                dirpath=ckpt_dir,
                filename=ckptfile,
            )
            finetune_early_stop = EarlyStopping(
                monitor='val_pearson',
                min_delta=0.001,
                patience=5,
                verbose=False,
                mode='max'
            )
            finetune_trainer = pl.Trainer(
                accelerator='cuda' if torch.cuda.is_available() else 'cpu',
                devices=1,
                max_epochs=10,
                logger=False,
                callbacks=[finetune_ckpt, finetune_early_stop],
                deterministic=True,
                enable_progress_bar=verbose,
                enable_model_summary=False,
                precision='16-mixed' if torch.cuda.is_available() else 32,
            )
            finetune_trainer.fit(model, train_dataloaders=finetune_train_loader, val_dataloaders=finetune_val_loader)

            # Get checkpoint path from finetune
            checkpoint_path = finetune_ckpt.best_model_path
        else:
            # Get checkpoint path from main training
            checkpoint_path = callback_ckpt.best_model_path

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            # Fallback to expected path
            checkpoint_path = os.path.join(ckpt_dir, f"{ckptfile}.ckpt")

        return {
            "model": model,
            "checkpoint": checkpoint_path if os.path.exists(checkpoint_path) else None,
            "trainer": trainer,
            "success": True
        }

    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model": None,
            "checkpoint": None,
            "trainer": None,
            "success": False,
            "error": str(e)
        }


def load_pl_legnet_from_checkpoint(ckpt_path: str, batch_size: int = 1024, patience: int = 10) -> Optional[PL_LegNet]:
    """Load PL_LegNet model weights from a checkpoint saved with save_weights_only=True."""
    try:
        # Create a default config for multi-class (k562, hepg2, wtc11)
        config = TrainingConfig(
            train_batch_size=batch_size,
            valid_batch_size=batch_size,
            output_dim=3  # Multi-class output for k562, hepg2, wtc11
        )

        model = PL_LegNet(config=config, batch_size=batch_size, patience=patience)
        state = torch.load(ckpt_path, map_location='cpu')
        state_dict = state.get('state_dict', state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            # Non-fatal, but informative
            print(f"Warning: when loading {ckpt_path}, missing keys: {missing}, unexpected keys: {unexpected}")
        return model
    except Exception as e:
        print(f"Failed to load checkpoint {ckpt_path}: {e}")
        return None


class LentIMPRAIterativeAugmentationSampler:
    """LentIMPRA iterative augmentation sampling for small data experiments."""

    def __init__(self, config: Any, device: str = 'cuda', seed: int = 42, use_augmentation: bool = False):
        self.config = config
        self.device = torch.device(device)
        self.sequence_length = 230  # LentIMPRA sequence length
        self.accumulated_sequences = []
        self.accumulated_targets = []
        self.seed = int(seed)
        self.use_augmentation = use_augmentation

    def _maybe_len(self, dataset) -> int:
        """Safely get length of a dataset without triggering typing complaints."""
        length_method = getattr(dataset, '__len__', None)
        try:
            if callable(length_method):
                length_val = length_method()
                return int(length_val) if isinstance(length_val, int) else -1
            return -1
        except Exception:
            return -1

    def load_model(self, checkpoint_path: str, architecture: str = 'transformer'):
        """Load trained D3 model for sampling."""
        print(f"Loading model from: {checkpoint_path}")
        model, graph, noise = load_trained_model(
            checkpoint_path, self.config, architecture, str(self.device)
        )
        model.eval()
        return model, graph, noise

    def _get_dataset_split(self, data_file: str, split: str):
        """Get specific dataset split with better error handling."""
        try:
            train_ds, val_ds, test_ds = get_lentimpra_datasets(data_file)
            splits = {'train': train_ds, 'val': val_ds, 'test': test_ds}
            if split not in splits:
                raise ValueError(f"Unknown split: {split}")
            return splits[split]
        except Exception as e:
            print(f"\nError loading dataset splits from {data_file}:")
            print(f"Error: {e}")
            raise RuntimeError(f"Cannot load datasets from {data_file}. Please check the file structure.")

    def _extract_sequences_targets(self, dataset):
        """Extract sequences and targets from dataset."""
        sequences, targets = [], []
        for seq, target in dataset:
            sequences.append(seq)
            targets.append(target)
        return torch.stack(sequences), torch.stack(targets)

    def create_conditioning_dataloader(self, data_file: str, target_size: int, batch_size: int = 128) -> DataLoader:
        """Create dataloader with conditioning labels from test set, but meeting target size requirements."""
        # Load test targets directly from H5 file
        with h5py.File(data_file, 'r') as f:
            if 'y_test' in f:
                test_targets = torch.tensor(np.array(f['y_test']), dtype=torch.float32)
            elif 'Y_test' in f:
                test_targets = torch.tensor(np.array(f['Y_test']), dtype=torch.float32)
            else:
                # Fallback: try using _get_dataset_split
                test_dataset = self._get_dataset_split(data_file, 'test')
                test_sequences, test_targets = self._extract_sequences_targets(test_dataset)

        # Ensure targets are 2D (N, signal_dim) before any operations
        if test_targets.dim() == 1:
            test_targets = test_targets.unsqueeze(-1)

        # Handle edge case: if target_size is 0, return empty dataloader
        if target_size <= 0:
            print(f"Warning: target_size={target_size}, creating empty dataloader")
            dummy_sequences = torch.zeros(0, self.sequence_length, 4)
            sampled_targets = torch.zeros(0, test_targets.shape[-1])
            dataset = TensorDataset(dummy_sequences, sampled_targets)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

        # Use test targets, but repeat/sample to meet target size
        if len(test_targets) >= target_size:
            # Sample subset
            indices = torch.randperm(len(test_targets))[:target_size]
            sampled_targets = test_targets[indices]
        else:
            # Repeat to meet target size
            repeats = (target_size + len(test_targets) - 1) // len(test_targets)
            repeated_targets = test_targets.repeat(repeats, 1)  # Repeat along batch dimension
            sampled_targets = repeated_targets[:target_size]

        # Create dummy sequences matching the actual number of targets
        actual_target_size = len(sampled_targets)
        dummy_sequences = torch.zeros(actual_target_size, self.sequence_length, 4)

        dataset = TensorDataset(dummy_sequences, sampled_targets)
        print(f"Created conditioning dataloader with {actual_target_size} samples from test set (requested: {target_size})")
        print(f"  - Target shape: {sampled_targets.shape} (signal_dim={sampled_targets.shape[-1]})")

        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)

    def create_dataloader(self, data_file: str, split: str = 'train',
                         batch_size: int = 32, augment_iteration: int = 0) -> DataLoader:
        """Create dataloader for iterative augmentation (for oracle training only)."""
        dataset = self._get_dataset_split(data_file, split)

        if augment_iteration == 0:
            # First iteration: use only test set baseline
            size_str = str(self._maybe_len(dataset))
            print(f"Iteration {augment_iteration}: Using test set baseline with {size_str} samples")
        else:
            # Subsequent iterations: combine subset + accumulated data
            subset_sequences, subset_targets = self._extract_sequences_targets(dataset)
            subset_sequences_onehot = F.one_hot(subset_sequences.long(), num_classes=4).float()

            if self.accumulated_sequences and self.accumulated_targets:
                all_sequences = torch.cat([subset_sequences_onehot] + self.accumulated_sequences, dim=0)
                all_targets = torch.cat([subset_targets] + self.accumulated_targets, dim=0)
            else:
                all_sequences, all_targets = subset_sequences_onehot, subset_targets

            dataset = TensorDataset(all_sequences, all_targets)
            print(f"Iteration {augment_iteration}: Combined dataset with {int(all_sequences.shape[0])} samples")
            print(f"  - Subset: {len(subset_sequences)}, Accumulated: {len(all_sequences) - len(subset_sequences)}")

        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)

    def sample_sequences_for_iteration(self, model, graph, noise, conditioning_dataloader: DataLoader,
                                     num_steps: int, show_progress: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample sequences for current iteration using GPU-optimized sampling."""
        # Optimization: Use larger batch sizes for sampling
        optimal_batch_size = min(512, conditioning_dataloader.batch_size * 8)

        # Collect all targets on GPU to minimize device transfers
        print("Collecting conditioning targets...")
        all_conditioning_targets = []
        for _, (_, targets) in enumerate(conditioning_dataloader):
            all_conditioning_targets.append(targets)

        # Handle empty dataloader
        if not all_conditioning_targets:
            print("Warning: No conditioning targets found in dataloader. Returning empty results.")
            return torch.zeros(0, self.sequence_length, 4), torch.zeros(0, 3)

        all_conditioning_targets = torch.cat(all_conditioning_targets, dim=0).to(self.device)

        total_samples = len(all_conditioning_targets)
        num_batches = (total_samples + optimal_batch_size - 1) // optimal_batch_size

        # Pre-allocate GPU tensors to avoid repeated allocations
        sampled_sequences_gpu = []
        all_targets_gpu = []

        iterator = tqdm(range(num_batches), desc=f"GPU-optimized sampling (steps={num_steps})") if show_progress else range(num_batches)

        for i in iterator:
            start_idx = i * optimal_batch_size
            end_idx = min(start_idx + optimal_batch_size, total_samples)
            batch_targets = all_conditioning_targets[start_idx:end_idx]
            current_batch_size = batch_targets.shape[0]

            # Create optimized sampler for this batch
            sampling_fn = sampling.get_pc_sampler(
                graph, noise, (current_batch_size, self.sequence_length), 'analytic',
                num_steps, device=self.device
            )

            # Keep everything on GPU during sampling
            sample = sampling_fn(model, batch_targets)

            # Convert to one-hot on GPU
            seq_pred_one_hot = F.one_hot(sample, num_classes=4).float()

            # Store on GPU and only move to CPU at the very end
            sampled_sequences_gpu.append(seq_pred_one_hot)
            all_targets_gpu.append(batch_targets)

            # Optimization: Clear cache less frequently
            if i % 20 == 0:
                torch.cuda.empty_cache()

        # Final concatenation on GPU, then single transfer to CPU
        print("Finalizing results...")
        final_sequences_gpu = torch.cat(sampled_sequences_gpu, dim=0)
        final_targets_gpu = torch.cat(all_targets_gpu, dim=0)

        # Single device transfer at the end
        sampled_sequences_cpu = final_sequences_gpu.cpu()
        all_targets_cpu = final_targets_gpu.cpu()

        # Clean up GPU memory
        del sampled_sequences_gpu, all_targets_gpu, final_sequences_gpu, final_targets_gpu, all_conditioning_targets
        torch.cuda.empty_cache()

        return sampled_sequences_cpu, all_targets_cpu

    def _format_sequences_for_oracle(self, sequences: torch.Tensor) -> torch.Tensor:
        """Format sequences for oracle model (expects B, 4, L format)."""
        if sequences.dim() == 2:  # (B, L) indices
            seq_one_hot = F.one_hot(sequences.long(), num_classes=4).float()
            return seq_one_hot.permute(0, 2, 1)
        elif sequences.dim() == 3:
            if sequences.shape[-1] == 4:  # (B, L, 4)
                return sequences.permute(0, 2, 1)
            elif sequences.shape[1] == 4:  # Already (B, 4, L)
                return sequences
            else:
                seq_one_hot = F.one_hot(sequences.long(), num_classes=4).float()
                return seq_one_hot.permute(0, 2, 1)
        else:
            raise ValueError(f"Unexpected sequence tensor shape: {sequences.shape}")

    def save_iteration_dataset(self, sequences: torch.Tensor, targets: torch.Tensor,
                             output_path: str, iteration: int):
        """Save dataset from current iteration as H5 file."""
        # Ensure parent directory exists with proper error handling
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except PermissionError:
                # Fall back to temp directory if we can't write to requested location
                temp_dir = tempfile.mkdtemp()
                filename = os.path.basename(output_path)
                output_path = os.path.join(temp_dir, filename)
                print(f"Warning: Cannot write to {parent_dir}, using temporary directory: {temp_dir}")

        # Convert sequences to indices if they are in one-hot format
        if sequences.dim() == 3 and sequences.shape[-1] == 4:  # (B, L, 4) one-hot format
            seq_indices = torch.argmax(sequences, dim=-1)
        elif sequences.dim() == 2:  # Already indices format
            seq_indices = sequences
        else:
            raise ValueError(f"Unexpected sequence tensor shape: {sequences.shape}")

        # Ensure targets are 2D (N, 3) for multi-class (k562, hepg2, wtc11)
        if targets.dim() == 1:
            raise ValueError(f"Expected 2D targets for multi-class, but got 1D with shape {targets.shape}")
        if targets.shape[-1] != 3:
            raise ValueError(f"Expected 3 outputs for multi-class (k562, hepg2, wtc11), but got {targets.shape[-1]}")

        try:
            with h5py.File(output_path, 'w') as f:
                # Use LentIMPRA format: X_train, Y_train (simpler than onehot_train for indices)
                f.create_dataset('X_train', data=seq_indices.cpu().numpy())
                f.create_dataset('Y_train', data=targets.cpu().numpy())
                f.attrs['iteration'] = iteration
                f.attrs['num_samples'] = len(sequences)
                f.attrs['sequence_length'] = sequences.shape[1] if sequences.dim() == 3 else self.sequence_length

            print(f"✓ Saved iteration {iteration} dataset with {len(sequences)} samples to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving dataset: {e}")
            # Try saving to temp directory as fallback
            temp_path = os.path.join(tempfile.mkdtemp(), f"iteration_{iteration}_dataset.h5")
            with h5py.File(temp_path, 'w') as f:
                f.create_dataset('X_train', data=seq_indices.cpu().numpy())
                f.create_dataset('Y_train', data=targets.cpu().numpy())
                f.attrs['iteration'] = iteration
                f.attrs['num_samples'] = len(sequences)
                f.attrs['sequence_length'] = sequences.shape[1] if sequences.dim() == 3 else self.sequence_length
            print(f"✓ Saved iteration {iteration} dataset to fallback location: {temp_path}")
            return temp_path

    def _prepare_model_specific_output_dir(self, model_checkpoint: str, output_dir: str) -> str:
        """Create and return a D3-model-specific output directory for datasets and results."""
        ckpt_stem = Path(model_checkpoint).stem
        model_dir = os.path.join(output_dir, f"d3_{ckpt_stem}")

        # Try to create directories with proper error handling
        try:
            os.makedirs(model_dir, exist_ok=True)
            datasets_dir = os.path.join(model_dir, "datasets")
            oracles_dir = os.path.join(model_dir, "oracles")
            os.makedirs(datasets_dir, exist_ok=True)
            os.makedirs(oracles_dir, exist_ok=True)
            return os.path.abspath(model_dir)
        except PermissionError:
            # Fall back to temporary directory if we can't write to requested location
            temp_base = tempfile.mkdtemp()
            model_dir = os.path.join(temp_base, f"d3_{ckpt_stem}")
            os.makedirs(model_dir, exist_ok=True)
            datasets_dir = os.path.join(model_dir, "datasets")
            oracles_dir = os.path.join(model_dir, "oracles")
            os.makedirs(datasets_dir, exist_ok=True)
            os.makedirs(oracles_dir, exist_ok=True)
            print(f"Warning: Cannot write to {output_dir}, using temporary directory: {temp_base}")
            return os.path.abspath(model_dir)

    def _train_multiple_oracles_for_dataset(self, dataset_path: str, work_dir: str,
                                          test_data_path: str, num_models: int = 3) -> Dict[str, Any]:
        """Train multiple mpralegnet oracles on the given dataset and return aggregated metrics.

        Training 3 mpralegnet models using different random seeds.
        Now tracks 3 separate Pearson R values for k562, hepg2, wtc11.
        """
        try:
            os.makedirs(work_dir, exist_ok=True)
        except PermissionError:
            work_dir = tempfile.mkdtemp()
            print(f"Warning: Cannot write to work directory, using temp: {work_dir}")

        # Get test set for evaluation
        test_dataset = self._get_dataset_split(test_data_path, 'test')
        test_sequences, test_targets = self._extract_sequences_targets(test_dataset)
        test_oracle_input = self._format_sequences_for_oracle(test_sequences)

        # Ensure test_targets is 2D (N, signal_dim)
        if test_targets.dim() == 1:
            test_targets = test_targets.unsqueeze(-1)

        k562_correlations = []  # K562 Pearson correlations from all models
        hepg2_correlations = []  # HepG2 Pearson correlations from all models
        wtc11_correlations = []  # WTC11 Pearson correlations from all models
        checkpoints = []

        print(f"Training {num_models} oracle models for dataset: {os.path.basename(dataset_path)}")

        dataset_stem = Path(dataset_path).stem
        # Support legacy single-checkpoint naming (no per-model suffix), if present
        legacy_ckpt_path = os.path.join(work_dir, f"oracle_mpralegnet_{dataset_stem}.ckpt")

        for model_idx in range(num_models):
            print(f"  Training oracle model {model_idx + 1}/{num_models}...")
            ckpt_name = f"oracle_mpralegnet_{dataset_stem}_m{model_idx}"
            ckpt_path = os.path.join(work_dir, f"{ckpt_name}.ckpt")

            # Use different seed for each model
            model_seed = self.seed + model_idx

            try:
                if os.path.exists(ckpt_path) or os.path.exists(legacy_ckpt_path):
                    # Resume: load model from checkpoint
                    use_path = ckpt_path if os.path.exists(ckpt_path) else legacy_ckpt_path
                    trained_model = load_pl_legnet_from_checkpoint(use_path, batch_size=1024, patience=10)
                    checkpoint_path = use_path
                    print(f"    Found existing checkpoint, skipping training: {use_path}")
                else:
                    # Train model
                    train_result = training_with_PL(
                        dataset_path=dataset_path,
                        train_max_epochs=25,
                        batch_size=1024,
                        patience=10,
                        verbose=False,
                        seed=model_seed,
                        out_dir=work_dir,
                        filename=ckpt_name,
                        use_augmentation=self.use_augmentation,
                    )
                    trained_model = train_result["model"]
                    checkpoint_path = train_result["checkpoint"]

                if trained_model is not None and checkpoint_path is not None:
                    # Evaluate on test set
                    trained_model.eval()
                    trained_model = trained_model.to(self.device)

                    with torch.no_grad():
                        predictions = trained_model.predict_custom(test_oracle_input.to(self.device))

                    # Compute Pearson correlations for each cell type
                    pred_np = predictions.cpu().numpy()
                    target_np = test_targets.cpu().numpy()

                    # Ensure 2D arrays (N, signal_dim)
                    if pred_np.ndim == 1:
                        pred_np = pred_np.reshape(-1, 1)
                    if target_np.ndim == 1:
                        target_np = target_np.reshape(-1, 1)

                    # Calculate Pearson R for each cell type (dim 0=k562, dim 1=hepg2, dim 2=wtc11)
                    cell_pearson = []
                    for i, cell_name in enumerate(['k562', 'hepg2', 'wtc11']):
                        if i < pred_np.shape[1]:
                            # Handle NaN values
                            valid_mask = ~(np.isnan(pred_np[:, i]) | np.isnan(target_np[:, i]))
                            if valid_mask.sum() > 1:
                                pearson_r = float(np.corrcoef(pred_np[valid_mask, i], target_np[valid_mask, i])[0, 1])
                            else:
                                pearson_r = 0.0
                            cell_pearson.append(pearson_r)
                        else:
                            cell_pearson.append(np.nan)

                    k562_correlations.append(cell_pearson[0])
                    hepg2_correlations.append(cell_pearson[1])
                    wtc11_correlations.append(cell_pearson[2])
                    checkpoints.append(checkpoint_path)
                    print(f"    Model {model_idx + 1}: K562={cell_pearson[0]:.4f}, HepG2={cell_pearson[1]:.4f}, WTC11={cell_pearson[2]:.4f}")
                else:
                    print(f"    Model {model_idx + 1}: Training or loading failed")
                    k562_correlations.append(np.nan)
                    hepg2_correlations.append(np.nan)
                    wtc11_correlations.append(np.nan)
                    checkpoints.append(None)

            except Exception as e:
                print(f"    Model {model_idx + 1}: Error during training/evaluation: {e}")
                import traceback
                traceback.print_exc()
                k562_correlations.append(np.nan)
                hepg2_correlations.append(np.nan)
                wtc11_correlations.append(np.nan)
                checkpoints.append(None)

        # Calculate aggregate statistics for each cell type
        valid_k562 = [x for x in k562_correlations if not np.isnan(x)]
        valid_hepg2 = [x for x in hepg2_correlations if not np.isnan(x)]
        valid_wtc11 = [x for x in wtc11_correlations if not np.isnan(x)]

        results = {
            "k562_correlations": k562_correlations,
            "hepg2_correlations": hepg2_correlations,
            "wtc11_correlations": wtc11_correlations,
            "checkpoints": checkpoints,
            "avg_k562_pearson": np.mean(valid_k562) if valid_k562 else np.nan,
            "std_k562_pearson": np.std(valid_k562) if valid_k562 else np.nan,
            "avg_hepg2_pearson": np.mean(valid_hepg2) if valid_hepg2 else np.nan,
            "std_hepg2_pearson": np.std(valid_hepg2) if valid_hepg2 else np.nan,
            "avg_wtc11_pearson": np.mean(valid_wtc11) if valid_wtc11 else np.nan,
            "std_wtc11_pearson": np.std(valid_wtc11) if valid_wtc11 else np.nan,
            "num_successful_models": len(valid_k562)  # Assuming all three have same validity
        }

        print(f"  Summary: K562={results['avg_k562_pearson']:.4f}±{results['std_k562_pearson']:.4f}, "
              f"HepG2={results['avg_hepg2_pearson']:.4f}±{results['std_hepg2_pearson']:.4f}, "
              f"WTC11={results['avg_wtc11_pearson']:.4f}±{results['std_wtc11_pearson']:.4f} "
              f"({results['num_successful_models']}/{num_models} models)")

        return results

    def run_iterative_augmentation_experiment(self, model_checkpoint: str, oracle_checkpoint: str,
                                            data_path: str, output_dir: str,
                                            max_iterations: int = 6,
                                            num_steps: Optional[int] = None,
                                            batch_size: int = 32,
                                            architecture: str = 'transformer',
                                            num_oracle_models: int = 3) -> Dict[str, Any]:

        """
        Run iterative augmentation experiment following the paper's methodology:
        1. Baseline: train set (oracles train on train set, evaluate on test set)
        2. Train set + train_set_size generated sequences (conditioned on test set labels)
        3. Train set + 2×train_set_size generated sequences
        4. Train set + 3×train_set_size generated sequences
        ...

        Each iteration adds a constant number of generated samples equal to the original train set size.
        Sequences are conditioned on test set labels (repeating test labels as necessary to meet target size).
        For each augmented dataset, train N mpralegnet oracles and evaluate on test set.
        """
        # Set seeds at the start of the experiment
        set_global_seed(self.seed)

        if num_steps is None:
            num_steps = self.sequence_length

        # Prepare output dir under D3 model name
        model_specific_dir = self._prepare_model_specific_output_dir(model_checkpoint, output_dir)
        datasets_dir = os.path.join(model_specific_dir, "datasets")
        oracles_dir = os.path.join(model_specific_dir, "oracles")

        # Results path and potential resume from existing
        results_path = os.path.join(model_specific_dir, "iterative_augmentation_results.json")
        results: Dict[str, Any]
        existing_iterations: set = set()
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    loaded = json.load(f)
                # Basic validation
                if isinstance(loaded, dict) and 'iteration_results' in loaded:
                    results = loaded
                    existing_iterations = {int(item.get('iteration', -1)) for item in results.get('iteration_results', [])}
                else:
                    results = {
                        'experiment_config': {},
                        'iteration_results': []
                    }
            except Exception:
                # On any load issue, start a fresh results container
                results = {
                    'experiment_config': {},
                    'iteration_results': []
                }
        else:
            results = {
                'experiment_config': {},
                'iteration_results': []
            }

        # Get train dataset size and sequences to use as baseline
        # Load train set directly from H5 file to get accurate size
        with h5py.File(data_path, 'r') as f:
            if 'onehot_train' in f:
                train_set_size = f['onehot_train'].shape[0]
            elif 'y_train' in f:
                train_set_size = f['y_train'].shape[0]
            elif 'X_train' in f:
                train_set_size = f['X_train'].shape[0]
            else:
                # Fallback: try using _get_dataset_split
                train_ds = self._get_dataset_split(data_path, 'train')
                train_sequences, train_targets = self._extract_sequences_targets(train_ds)
                train_set_size = len(train_sequences)

        print(f"Train set size: {train_set_size} samples")

        # Use TRAIN SET as the baseline (iteration 0)
        train_ds = self._get_dataset_split(data_path, 'train')
        subset_sequences, subset_targets = self._extract_sequences_targets(train_ds)

        # Target sizes based on train set size: iteration 0 = 1x train_set_size, iteration i = (i+1)x train_set_size
        baseline_size = train_set_size
        generation_size = train_set_size
        target_sizes = {i: (i + 1) * train_set_size for i in range(max_iterations + 1)}

        # Save iteration 0 dataset (baseline) only if not already present
        iter0_path = os.path.join(datasets_dir, f"iteration_0_dataset.h5")
        if os.path.exists(iter0_path):
            print(f"Found existing baseline dataset at {iter0_path}; skipping creation.")
            # Read subset size from file if possible
            try:
                with h5py.File(iter0_path, 'r') as f:
                    if 'X_train' in f:
                        subset_size = int(np.array(f['X_train']).shape[0])
                    else:
                        subset_size = int(subset_sequences.shape[0])
            except Exception:
                subset_size = int(subset_sequences.shape[0])
        else:
            iter0_path = self.save_iteration_dataset(subset_sequences, subset_targets, iter0_path, 0)
            subset_size = int(subset_sequences.shape[0])

        # Print experiment header and load D3 model
        print("=" * 70)
        print("LENTIMPRA ITERATIVE AUGMENTATION EXPERIMENT")
        print("=" * 70)
        print(f"Model checkpoint: {model_checkpoint}")
        print(f"Oracle checkpoint (initial evaluation): {oracle_checkpoint}")
        print(f"Baseline data size (train set): {subset_size} samples")
        print(f"Max iterations: {max_iterations}")
        print(f"Sampling steps: {num_steps}")
        print(f"Number of oracle models per condition: {num_oracle_models}")
        print(f"Use augmentation: {self.use_augmentation}")
        print(f"Output directory: {model_specific_dir}")
        print("=" * 70)

        model, graph, noise = self.load_model(model_checkpoint, architecture)

        # Initialize results container if empty / update experiment config
        if not results.get('experiment_config'):
            results['experiment_config'] = {
                'model_checkpoint': model_checkpoint,
                'oracle_checkpoint_initial': oracle_checkpoint,
                'data_path': data_path,
                'max_iterations': max_iterations,
                'num_steps': num_steps,
                'batch_size': batch_size,
                'architecture': architecture,
                'subset_size': subset_size,
                'num_oracle_models': num_oracle_models,
                'model_specific_dir': model_specific_dir,
                'seed': self.seed,
                'use_augmentation': self.use_augmentation,
            }
        else:
            # Keep prior settings but update dynamic ones
            results['experiment_config'].update({
                'max_iterations': max_iterations,
                'num_steps': num_steps,
                'batch_size': batch_size,
                'num_oracle_models': num_oracle_models,
                'seed': self.seed,
                'use_augmentation': self.use_augmentation,
            })

        # Train/evaluate oracles for iteration 0 only if not already recorded
        if 0 not in existing_iterations:
            iter0_oracle_dir = os.path.join(oracles_dir, "iteration_0")
            iter0_train_info = self._train_multiple_oracles_for_dataset(
                iter0_path, iter0_oracle_dir, data_path, num_oracle_models
            )

            # Record iteration 0 results (multi-class: k562, hepg2, wtc11)
            results['iteration_results'].append({
                'iteration': 0,
                'dataset_path': iter0_path,
                'dataset_size': subset_size,
                'original_size': subset_size,
                'generated_size': 0,
                'oracle_checkpoints': iter0_train_info.get("checkpoints"),
                # Multi-class correlations
                'k562_correlations': iter0_train_info.get("k562_correlations"),
                'hepg2_correlations': iter0_train_info.get("hepg2_correlations"),
                'wtc11_correlations': iter0_train_info.get("wtc11_correlations"),
                # Average and std for each cell type
                'avg_test_pearson_k562': iter0_train_info.get("avg_k562_pearson"),
                'std_test_pearson_k562': iter0_train_info.get("std_k562_pearson"),
                'avg_test_pearson_hepg2': iter0_train_info.get("avg_hepg2_pearson"),
                'std_test_pearson_hepg2': iter0_train_info.get("std_hepg2_pearson"),
                'avg_test_pearson_wtc11': iter0_train_info.get("avg_wtc11_pearson"),
                'std_test_pearson_wtc11': iter0_train_info.get("std_wtc11_pearson"),
                'num_successful_models': iter0_train_info.get("num_successful_models"),
                'description': "Baseline: train set"
            })

            print(f"\nIteration 0 (Baseline): {subset_size} samples (train set)")
            print(f"Test Pearson R - K562: {iter0_train_info.get('avg_k562_pearson'):.4f} ± {iter0_train_info.get('std_k562_pearson'):.4f}")
            print(f"Test Pearson R - HepG2: {iter0_train_info.get('avg_hepg2_pearson'):.4f} ± {iter0_train_info.get('std_hepg2_pearson'):.4f}")
            print(f"Test Pearson R - WTC11: {iter0_train_info.get('avg_wtc11_pearson'):.4f} ± {iter0_train_info.get('std_wtc11_pearson'):.4f}")

            # Persist results incrementally
            try:
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            except Exception as e:
                print(f"Warning: failed to save results after iteration 0: {e}")
        else:
            print("Iteration 0 already present in results; skipping re-evaluation.")

        # Run augmentation iterations (1..max_iterations)
        for iteration in range(1, max_iterations + 1):
            # If iteration exists in results, only skip when all oracles are complete
            if iteration in existing_iterations:
                # Find the recorded entry for this iteration
                recorded = None
                for item in results.get('iteration_results', []):
                    if int(item.get('iteration', -1)) == iteration:
                        recorded = item
                        break
                if recorded is not None:
                    # Check k562 correlations to determine if iteration is complete
                    k562_corrs = recorded.get('k562_correlations') or []
                    # Count valid non-NaN correlations
                    valid_count = 0
                    for v in k562_corrs:
                        try:
                            is_nan = np.isnan(v)
                        except Exception:
                            is_nan = False
                        if v is not None and not is_nan:
                            valid_count += 1
                    if valid_count >= int(num_oracle_models):
                        print(f"\n{'='*20} ITERATION {iteration} (resume: already complete) {'='*20}")
                        continue
                    else:
                        print(f"\n{'='*20} ITERATION {iteration} (resume: incomplete oracles, continuing) {'='*20}")
                else:
                    print(f"\n{'='*20} ITERATION {iteration} (resume: no recorded entry, continuing) {'='*20}")
            else:
                print(f"\n{'='*20} ITERATION {iteration} {'='*20}")

            # Dataset path for this iteration
            iter_path = os.path.join(datasets_dir, f"iteration_{iteration}_dataset.h5")

            # If dataset exists, skip sampling for this iteration but still (re)train missing oracles
            if os.path.exists(iter_path):
                print(f"Found existing dataset for iteration {iteration} at {iter_path}; skipping sampling.")
                # Load sizes for reporting
                try:
                    with h5py.File(iter_path, 'r') as f:
                        total_size = int(np.array(f['X_train']).shape[0]) if 'X_train' in f else None
                except Exception:
                    total_size = None
                if total_size is None:
                    total_size = subset_size  # Best-effort fallback
                generated_size = max(0, total_size - subset_size)

                # Train multiple oracles on augmented dataset and evaluate on test set
                oracle_iter_dir = os.path.join(oracles_dir, f"iteration_{iteration}")
                train_info = self._train_multiple_oracles_for_dataset(
                    iter_path, oracle_iter_dir, data_path, num_oracle_models
                )

                # Store or update results entry for this iteration (multi-class)
                updated = False
                for idx, item in enumerate(results['iteration_results']):
                    if int(item.get('iteration', -1)) == iteration:
                        results['iteration_results'][idx].update({
                            'dataset_path': iter_path,
                            'dataset_size': int(total_size),
                            'original_size': subset_size,
                            'generated_size': int(generated_size),
                            'oracle_checkpoints': train_info.get("checkpoints"),
                            # Multi-class correlations
                            'k562_correlations': train_info.get("k562_correlations"),
                            'hepg2_correlations': train_info.get("hepg2_correlations"),
                            'wtc11_correlations': train_info.get("wtc11_correlations"),
                            # Average and std for each cell type
                            'avg_test_pearson_k562': train_info.get("avg_k562_pearson"),
                            'std_test_pearson_k562': train_info.get("std_k562_pearson"),
                            'avg_test_pearson_hepg2': train_info.get("avg_hepg2_pearson"),
                            'std_test_pearson_hepg2': train_info.get("std_hepg2_pearson"),
                            'avg_test_pearson_wtc11': train_info.get("avg_wtc11_pearson"),
                            'std_test_pearson_wtc11': train_info.get("std_wtc11_pearson"),
                            'num_successful_models': train_info.get("num_successful_models"),
                            'description': results['iteration_results'][idx].get('description', f"Original + {iteration} set{'s' if iteration > 1 else ''} of generated sequences")
                        })
                        updated = True
                        break
                if not updated:
                    results['iteration_results'].append({
                        'iteration': iteration,
                        'dataset_path': iter_path,
                        'dataset_size': int(total_size),
                        'original_size': subset_size,
                        'generated_size': int(generated_size),
                        'oracle_checkpoints': train_info.get("checkpoints"),
                        # Multi-class correlations
                        'k562_correlations': train_info.get("k562_correlations"),
                        'hepg2_correlations': train_info.get("hepg2_correlations"),
                        'wtc11_correlations': train_info.get("wtc11_correlations"),
                        # Average and std for each cell type
                        'avg_test_pearson_k562': train_info.get("avg_k562_pearson"),
                        'std_test_pearson_k562': train_info.get("std_k562_pearson"),
                        'avg_test_pearson_hepg2': train_info.get("avg_hepg2_pearson"),
                        'std_test_pearson_hepg2': train_info.get("std_hepg2_pearson"),
                        'avg_test_pearson_wtc11': train_info.get("avg_wtc11_pearson"),
                        'std_test_pearson_wtc11': train_info.get("std_wtc11_pearson"),
                        'num_successful_models': train_info.get("num_successful_models"),
                        'description': f"Original + {iteration} set{'s' if iteration > 1 else ''} of generated sequences"
                    })

                print(f"Test Pearson R - K562: {train_info.get('avg_k562_pearson'):.4f} ± {train_info.get('std_k562_pearson'):.4f}")
                print(f"Test Pearson R - HepG2: {train_info.get('avg_hepg2_pearson'):.4f} ± {train_info.get('std_hepg2_pearson'):.4f}")
                print(f"Test Pearson R - WTC11: {train_info.get('avg_wtc11_pearson'):.4f} ± {train_info.get('std_wtc11_pearson'):.4f}")

                # Persist results incrementally after each iteration
                try:
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                except Exception as e:
                    print(f"Warning: failed to save results after iteration {iteration}: {e}")

                # Move on to next iteration
                continue

            # Otherwise: sample new sequences for this iteration and then train oracles
            # Each iteration adds a constant number of generated samples equal to the original train set size
            samples_to_generate = subset_size  # Constant: always generate train_set_size samples per iteration
            print(f"  Generating {samples_to_generate} new samples (constant: original train set size)")

            conditioning_dataloader = self.create_conditioning_dataloader(data_path, samples_to_generate, batch_size)

            # Sample new sequences conditioned on test set labels
            print(f"Sampling {samples_to_generate} sequences for iteration {iteration} using test set conditioning...")
            sampled_sequences, sampled_targets = self.sample_sequences_for_iteration(
                model, graph, noise, conditioning_dataloader, num_steps, show_progress=True
            )

            # Build upon the complete dataset from the previous iteration
            if iteration == 1:
                # First augmentation: start from baseline + new samples
                subset_sequences_onehot = F.one_hot(subset_sequences.long(), num_classes=4).float()
                current_sequences = torch.cat([subset_sequences_onehot, sampled_sequences], dim=0)
                current_targets = torch.cat([subset_targets, sampled_targets.squeeze()], dim=0)
                print(f"  Building from baseline ({len(subset_sequences)}) + new samples ({len(sampled_sequences)})")
            else:
                # Load complete dataset from previous iteration
                prev_iter_path = os.path.join(datasets_dir, f"iteration_{iteration-1}_dataset.h5")
                if os.path.exists(prev_iter_path):
                    print(f"  Loading previous iteration dataset from: {prev_iter_path}")
                    with h5py.File(prev_iter_path, 'r') as f:
                        prev_sequences = torch.tensor(np.array(f['X_train']), dtype=torch.long)
                        prev_targets = torch.tensor(np.array(f['Y_train']), dtype=torch.float32)

                    # Convert previous sequences to one-hot if needed
                    if prev_sequences.dim() == 2:  # (N, L) indices format
                        prev_sequences_onehot = F.one_hot(prev_sequences.long(), num_classes=4).float()
                    else:
                        prev_sequences_onehot = prev_sequences.float()

                    # Add new samples to previous complete dataset
                    current_sequences = torch.cat([prev_sequences_onehot, sampled_sequences], dim=0)
                    current_targets = torch.cat([prev_targets, sampled_targets.squeeze()], dim=0)
                    print(f"  Building from previous iteration ({len(prev_sequences)}) + new samples ({len(sampled_sequences)})")
                else:
                    print(f"  Warning: Previous iteration file not found, falling back to accumulated method")
                    # Fallback to original logic
                    self.accumulated_sequences.append(sampled_sequences)
                    self.accumulated_targets.append(sampled_targets.squeeze())
                    subset_sequences_onehot = F.one_hot(subset_sequences.long(), num_classes=4).float()
                    all_sequences = [subset_sequences_onehot] + self.accumulated_sequences
                    all_targets = [subset_targets] + self.accumulated_targets
                    current_sequences = torch.cat(all_sequences, dim=0)
                    current_targets = torch.cat(all_targets, dim=0)

            # Update accumulated sequences for potential fallback use
            if iteration == 1:
                self.accumulated_sequences = [sampled_sequences]
                self.accumulated_targets = [sampled_targets.squeeze()]
            else:
                self.accumulated_sequences.append(sampled_sequences)
                self.accumulated_targets.append(sampled_targets.squeeze())

            # Verify we hit the expected target size
            desired_total = target_sizes.get(iteration, current_sequences.shape[0])
            if abs(current_sequences.shape[0] - desired_total) > 10:  # Allow small tolerance
                print(f"Warning: iteration {iteration} has {current_sequences.shape[0]} samples, expected {desired_total}")
            # Trim to exact size if needed
            if current_sequences.shape[0] > desired_total:
                current_sequences = current_sequences[:desired_total]
                current_targets = current_targets[:desired_total]

            total_size = len(current_sequences)
            generated_size = total_size - subset_size

            print(f"Iteration {iteration}: {total_size} total samples")
            print(f"  - Original (train set): {subset_size}")
            print(f"  - Generated: {generated_size}")
            print(f"  - Ratio: {generated_size/subset_size:.2f}x augmentation")

            # Save dataset for this iteration
            iter_path = self.save_iteration_dataset(current_sequences, current_targets, iter_path, iteration)

            # Train multiple oracles on augmented dataset and evaluate on test set
            oracle_iter_dir = os.path.join(oracles_dir, f"iteration_{iteration}")
            train_info = self._train_multiple_oracles_for_dataset(
                iter_path, oracle_iter_dir, data_path, num_oracle_models
            )

            # Store results (multi-class)
            results['iteration_results'].append({
                'iteration': iteration,
                'dataset_path': iter_path,
                'dataset_size': int(total_size),
                'original_size': subset_size,
                'new_samples_this_iteration': len(sampled_sequences),
                'generated_size': int(generated_size),
                'oracle_checkpoints': train_info.get("checkpoints"),
                # Multi-class correlations
                'k562_correlations': train_info.get("k562_correlations"),
                'hepg2_correlations': train_info.get("hepg2_correlations"),
                'wtc11_correlations': train_info.get("wtc11_correlations"),
                # Average and std for each cell type
                'avg_test_pearson_k562': train_info.get("avg_k562_pearson"),
                'std_test_pearson_k562': train_info.get("std_k562_pearson"),
                'avg_test_pearson_hepg2': train_info.get("avg_hepg2_pearson"),
                'std_test_pearson_hepg2': train_info.get("std_hepg2_pearson"),
                'avg_test_pearson_wtc11': train_info.get("avg_wtc11_pearson"),
                'std_test_pearson_wtc11': train_info.get("std_wtc11_pearson"),
                'num_successful_models': train_info.get("num_successful_models"),
                'description': f"Original + {iteration} set{'s' if iteration > 1 else ''} of generated sequences"
            })

            print(f"Test Pearson R - K562: {train_info.get('avg_k562_pearson'):.4f} ± {train_info.get('std_k562_pearson'):.4f}")
            print(f"Test Pearson R - HepG2: {train_info.get('avg_hepg2_pearson'):.4f} ± {train_info.get('std_hepg2_pearson'):.4f}")
            print(f"Test Pearson R - WTC11: {train_info.get('avg_wtc11_pearson'):.4f} ± {train_info.get('std_wtc11_pearson'):.4f}")

            # Persist results incrementally after each iteration
            try:
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            except Exception as e:
                print(f"Warning: failed to save results after iteration {iteration}: {e}")

        # Final save (redundant but ensures latest results are persisted)
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)  # default=str to handle numpy types
        except Exception as e:
            # Fallback to temp file
            temp_results = os.path.join(tempfile.mkdtemp(), "iterative_augmentation_results.json")
            with open(temp_results, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            results_path = temp_results
            print(f"Warning: Saved results to temp location: {temp_results}")

        print(f"\n✓ Iterative augmentation experiment completed!")
        print(f"Results saved to: {results_path}")
        return results


def main():
    """Main function for running iterative augmentation experiments."""
    parser = argparse.ArgumentParser(description='LentIMPRA Iterative Augmentation Experiment')
    parser.add_argument('--model_checkpoint', required=True, help='Path to trained D3 model checkpoint')
    parser.add_argument('--oracle_checkpoint', required=False, default='', help='(Optional) Path to oracle model checkpoint for initial eval log')
    parser.add_argument('--data_path', required=True, help='Path to LentIMPRA data file (test set will be used as baseline)')
    parser.add_argument('--output_dir', required=True, help='Directory to save iteration datasets and oracles')
    parser.add_argument('--config', help='Path to config file (optional)')
    parser.add_argument('--max_iterations', type=int, default=6, help='Maximum iterations (default: 6)')
    parser.add_argument('--num_steps', type=int, help='Sampling steps (default: sequence length)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--architecture', choices=['transformer', 'convolutional', 'transformer_multi_class'],
                       default='transformer_multi_class', help='Model architecture (default: transformer_multi_class for multi-class)')
    parser.add_argument('--seed', type=int, default=42, help='Global seed for reproducibility')
    parser.add_argument('--num_oracle_models', type=int, default=3, help='Number of oracle models to train per condition (default: 3)')
    parser.add_argument('--use_augmentation', action='store_true', help='Enable EvoAug data augmentation for training')

    args = parser.parse_args()

    # Load config
    if args.config:
        config = OmegaConf.load(args.config)
    else:
        config_file = Path(__file__).parent / 'configs' / 'transformer.yaml'
        if config_file.exists():
            config = OmegaConf.load(config_file)
            print(f"Using default config: {config_file}")
        else:
            print("Error: No config provided and default config not found")
            return 1

    # Set global seed as early as possible
    set_global_seed(int(args.seed))

    # Create sampler and run experiment
    sampler = LentIMPRAIterativeAugmentationSampler(
        config,
        seed=int(args.seed),
        use_augmentation=args.use_augmentation
    )

    try:
        results = sampler.run_iterative_augmentation_experiment(
            model_checkpoint=args.model_checkpoint,
            oracle_checkpoint=args.oracle_checkpoint,
            data_path=args.data_path,
            output_dir=args.output_dir,
            max_iterations=args.max_iterations,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            architecture=args.architecture,
            num_oracle_models=args.num_oracle_models,
        )

        # Print summary (multi-class: k562, hepg2, wtc11)
        print("\n" + "=" * 70)
        print("EXPERIMENT SUMMARY")
        print("=" * 70)
        for result in results['iteration_results']:
            k562_avg = result.get('avg_test_pearson_k562')
            k562_std = result.get('std_test_pearson_k562')
            hepg2_avg = result.get('avg_test_pearson_hepg2')
            hepg2_std = result.get('std_test_pearson_hepg2')
            wtc11_avg = result.get('avg_test_pearson_wtc11')
            wtc11_std = result.get('std_test_pearson_wtc11')

            print(f"Iteration {result['iteration']}: {result['dataset_size']} samples")
            if k562_avg is not None and hepg2_avg is not None and wtc11_avg is not None:
                print(f"  K562:  {k562_avg:.4f} ± {k562_std:.4f}")
                print(f"  HepG2: {hepg2_avg:.4f} ± {hepg2_std:.4f}")
                print(f"  WTC11: {wtc11_avg:.4f} ± {wtc11_std:.4f}")
            else:
                print("  Results: N/A")

        return 0

    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
