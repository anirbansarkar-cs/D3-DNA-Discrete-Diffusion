#!/usr/bin/env python3
"""
DeepSTARR iterative augmentation sampling for small data experiments.

Iteratively augments a subset dataset by sampling sequences and adding them:
- Iteration 0: Use provided subset data (25% baseline)
- Iteration 1+: Subset data + accumulated sampled sequences
At each iteration, multiple DeepSTARR oracles are trained on the augmented dataset
and evaluated on the test set to compute Pearson R for Dev task.

Follows the experiment from Section 4.2:
- 4 conditions with progressively more generated sequences
- 5 DeepSTARR models trained per condition with different generated sequences
- Pearson correlation on Dev task measured on test set
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

# EvoAug imports (optional)
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
from model_zoo.deepstarr.data import get_deepstarr_datasets
from model_zoo.deepstarr.models import load_trained_model

# Ensure deterministic CuBLAS workspace config is set before any CUDA operations
if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def set_global_seed(seed: int):
    """Set seeds for Python, NumPy, and Torch for reproducibility."""
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def check_existing_oracle_checkpoints(oracle_dir: str, dataset_stem: str, num_models: int = 5) -> Dict[str, Any]:
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
        ckpt_name = f"oracle_DeepSTARR_{dataset_stem}_m{model_idx}.ckpt"
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
# DeepSTARR Oracle Model (Local Implementation)
# =============================================================================

class DeepSTARR(nn.Module):
    """DeepSTARR model from de Almeida et al., 2022."""
    
    def __init__(self, output_dim, d=256):
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


class PL_DeepSTARR(pl.LightningModule):
    """PyTorch Lightning wrapper for DeepSTARR model."""
    
    def __init__(self,
                 batch_size: int = 128,
                 train_max_epochs: int = 100,
                 patience: int = 10,
                 min_delta: float = 0.001,
                 input_h5_file: str = 'DeepSTARR_data.h5',
                 lr: float = 0.002,
                 weight_decay: float = 1e-6,
                 min_lr: float = 0.0,
                 lr_patience: int = 10,
                 decay_factor: float = 0.1):
        super().__init__()
        self.save_hyperparameters()
        
        # Model configuration
        self.model = DeepSTARR(output_dim=2)
        self.name = 'DeepSTARR'
        
        # Training configuration
        self.batch_size = batch_size
        self.train_max_epochs = train_max_epochs
        self.patience = patience
        self.lr = lr
        self.min_delta = min_delta
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.lr_patience = lr_patience
        self.decay_factor = decay_factor
        
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
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=self.lr_patience, 
            min_lr=self.min_lr, 
            factor=self.decay_factor
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "monitor": "val_loss"
            }
        } 
    
    def validation_step(self, batch, batch_idx):
        """Validation step for one batch."""
        self.model.eval()
        inputs, labels = batch
        loss_fn = nn.MSELoss()
        outputs = self.model(inputs)
        loss = loss_fn(outputs, labels)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, 
                prog_bar=True, logger=True, sync_dist=True)
        
        # Calculate and log PCC metric
        out_cpu = outputs.detach().cpu()
        lab_cpu = labels.detach().cpu()
        pcc = torch.tensor(self.metrics(out_cpu, lab_cpu)['PCC'].mean())
        self.log("val_pcc", pcc, on_step=False, on_epoch=True, 
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
    
    def metrics(self, y_score, y_true):
        """Calculate Pearson and Spearman correlation metrics."""
        # Spearman correlation
        spearman_vals = []
        for output_index in range(y_score.shape[1]):
            spearman_vals.append(
                stats.spearmanr(y_true[:, output_index], y_score[:, output_index])[0]
            )
        spearmanr_vals = np.array(spearman_vals)
        
        # Pearson correlation
        pearson_vals = []
        for output_index in range(y_score.shape[-1]):
            pearson_vals.append(
                stats.pearsonr(y_true[:, output_index], y_score[:, output_index])[0]
            )
        pearsonr_vals = np.array(pearson_vals)
        
        return {'Spearman': spearmanr_vals, 'PCC': pearsonr_vals}

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
                     train_max_epochs: int = 100,
                     batch_size: int = 128,
                     patience: int = 10,
                     verbose: bool = False,
                     seed: int = 42,
                     out_dir: Optional[str] = None,
                     filename: Optional[str] = None,
                     use_evoaug: bool = False) -> Dict[str, Any]:
    """Train DeepSTARR model using PyTorch Lightning.

    For every oracle model trained, the dataloader used should come from RobustLoader
    (only for the train set, not validation set) when use_evoaug is True.

    Args:
        dataset_path: Path to H5 file containing training data
        train_max_epochs: Maximum training epochs
        batch_size: Training batch size
        patience: Early stopping patience
        verbose: Whether to print verbose output
        seed: Random seed
        out_dir: Directory to save checkpoints
        filename: Base filename (without extension) for the checkpoint
        use_evoaug: If True, enable EvoAug with RobustLoader for training

    Returns:
        Dictionary containing trained model and metrics
    """
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load data from H5 file
        with h5py.File(dataset_path, 'r') as f:
            X_train = torch.tensor(np.array(f['X_train']), dtype=torch.float32)
            Y_train = torch.tensor(np.array(f['Y_train']), dtype=torch.float32)
            
            # Check if we have validation data, otherwise split from training
            if 'X_val' in f:
                X_val = torch.tensor(np.array(f['X_val']), dtype=torch.float32)
                Y_val = torch.tensor(np.array(f['Y_val']), dtype=torch.float32)
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
        
        # Convert to (N, 4, L) one-hot format for DeepSTARR model
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
        
        if verbose:
            print(f"Training data shape: {X_train.shape}")
            print(f"Training labels shape: {Y_train.shape}")
            print(f"Validation data shape: {X_val.shape}")
            print(f"Validation labels shape: {Y_val.shape}")
        
        # Initialize model
        model = PL_DeepSTARR(
            input_h5_file=dataset_path,
            batch_size=batch_size,
            train_max_epochs=train_max_epochs,
            patience=patience
        )
        
        # Setup data loaders
        num_workers = 4

        # Create base datasets
        train_dataset = TensorDataset(X_train, Y_train)
        val_dataset = TensorDataset(X_val, Y_val)

        # Use EvoAug RobustLoader for training if requested and available
        if use_evoaug and EVOAUG_AVAILABLE:
            # Augmentation list matching data.py (lines 143-150)
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
            if use_evoaug and not EVOAUG_AVAILABLE:
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
        ckptfile = filename if filename is not None else f"oracle_DeepSTARR_{dataset_name}"
        ckpt_dir = out_dir if out_dir is not None else "./"
        
        callback_ckpt = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_weights_only=True,
            dirpath=ckpt_dir,
            filename=ckptfile,
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=model.min_delta,
            patience=patience,
            verbose=False,
            mode='min'
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
            enable_model_summary=False
        )
        
        # Train
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # Finetune on the original data without augmentations
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
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_weights_only=True,
            dirpath=ckpt_dir,
            filename=ckptfile,
        )
        finetune_early_stop = EarlyStopping(
            monitor='val_loss',
            min_delta=model.min_delta,
            patience=5,
            verbose=False,
            mode='min'
        )
        finetune_trainer = pl.Trainer(
            accelerator='cuda' if torch.cuda.is_available() else 'cpu',
            devices=1,
            max_epochs=10,
            logger=False,
            callbacks=[finetune_ckpt, finetune_early_stop],
            deterministic=True,
            enable_progress_bar=verbose,
            enable_model_summary=False
        )
        finetune_trainer.fit(model, train_dataloaders=finetune_train_loader, val_dataloaders=finetune_val_loader)

        # Get checkpoint path
        checkpoint_path = finetune_ckpt.best_model_path
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
        return {
            "model": None,
            "checkpoint": None,
            "trainer": None,
            "success": False,
            "error": str(e)
        }


def load_pl_deepstarr_from_checkpoint(ckpt_path: str, batch_size: int = 128, patience: int = 10) -> Optional[PL_DeepSTARR]:
    """Load PL_DeepSTARR model weights from a checkpoint saved with save_weights_only=True."""
    try:
        model = PL_DeepSTARR(batch_size=batch_size, patience=patience)
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


class DeepSTARRIterativeAugmentationSampler:
    """DeepSTARR iterative augmentation sampling for small data experiments."""

    def __init__(self, config: Any, device: str = 'cuda', seed: int = 42, use_evoaug: bool = False, class_balance: bool = False):
        self.config = config
        self.device = torch.device(device)
        self.sequence_length = 249
        self.accumulated_sequences = []
        self.accumulated_targets = []
        self.seed = int(seed)
        self.use_evoaug = use_evoaug
        self.class_balance = class_balance

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
            train_ds, val_ds, test_ds = get_deepstarr_datasets(data_file)
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
        # Get test set for conditioning labels
        test_dataset = self._get_dataset_split(data_file, 'test')
        test_sequences, test_targets = self._extract_sequences_targets(test_dataset)

        if self.class_balance:
            # Apply class balancing: create 4 equally-sized groups
            balanced_targets = self._create_balanced_targets(test_targets, target_size)
        else:
            # Use test targets, but repeat/sample to meet target size
            if len(test_targets) >= target_size:
                # Sample subset
                indices = torch.randperm(len(test_targets))[:target_size]
                balanced_targets = test_targets[indices]
            else:
                # Repeat to meet target size
                repeats = (target_size + len(test_targets) - 1) // len(test_targets)
                repeated_targets = test_targets.repeat(repeats, 1)
                balanced_targets = repeated_targets[:target_size]

        # Create dummy sequences matching the actual number of targets
        actual_target_size = len(balanced_targets)
        dummy_sequences = torch.zeros(actual_target_size, self.sequence_length, 4)

        dataset = TensorDataset(dummy_sequences, balanced_targets)
        print(f"Created conditioning dataloader with {actual_target_size} samples from test set (requested: {target_size})")
        if self.class_balance:
            print(f"  - Class balanced: 4 groups with {actual_target_size//4} samples each")

        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)

    def _create_balanced_targets(self, test_targets: torch.Tensor, target_size: int) -> torch.Tensor:
        """Create class-balanced targets for 4 combinations: (hi,hi), (hi,lo), (lo,hi), (lo,lo)."""
        # Classify test targets into 4 classes based on threshold 0
        dev_hi = test_targets[:, 0] > 0  # Dev > 0
        hk_hi = test_targets[:, 1] > 0   # HK > 0

        # Create 4 class masks
        class_masks = {
            'hi_hi': dev_hi & hk_hi,
            'hi_lo': dev_hi & ~hk_hi,
            'lo_hi': ~dev_hi & hk_hi,
            'lo_lo': ~dev_hi & ~hk_hi
        }

        samples_per_class = target_size // 4
        balanced_targets = []

        for class_name, mask in class_masks.items():
            class_targets = test_targets[mask]
            if len(class_targets) == 0:
                # If no samples in this class, create synthetic targets
                dev_val = 1.0 if 'hi' in class_name.split('_')[0] else -1.0
                hk_val = 1.0 if 'hi' in class_name.split('_')[1] else -1.0
                synthetic_targets = torch.tensor([[dev_val, hk_val]], dtype=test_targets.dtype).repeat(samples_per_class, 1)
                balanced_targets.append(synthetic_targets)
                print(f"  Warning: No {class_name} samples found, using synthetic targets")
            elif len(class_targets) >= samples_per_class:
                # Sample from available targets
                indices = torch.randperm(len(class_targets))[:samples_per_class]
                balanced_targets.append(class_targets[indices])
            else:
                # Repeat to meet target
                repeats = (samples_per_class + len(class_targets) - 1) // len(class_targets)
                repeated = class_targets.repeat(repeats, 1)
                balanced_targets.append(repeated[:samples_per_class])

            print(f"  {class_name}: {len(class_targets)} available → {samples_per_class} used")

        concatenated_targets = torch.cat(balanced_targets, dim=0)

        # Ensure we return exactly the requested target_size (trim if slightly over due to rounding)
        if len(concatenated_targets) > target_size:
            concatenated_targets = concatenated_targets[:target_size]

        print(f"  Total balanced targets created: {len(concatenated_targets)} (requested: {target_size})")
        return concatenated_targets

    def create_dataloader(self, data_file: str, split: str = 'train',
                         batch_size: int = 32, augment_iteration: int = 0) -> DataLoader:
        """Create dataloader for iterative augmentation (for oracle training only)."""
        dataset = self._get_dataset_split(data_file, split)

        if augment_iteration == 0:
            # First iteration: use only subset data
            size_str = str(self._maybe_len(dataset))
            print(f"Iteration {augment_iteration}: Using subset data with {size_str} samples")
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
        # Optimization 1: Use larger batch sizes for sampling (but keep 100 steps)
        optimal_batch_size = min(512, conditioning_dataloader.batch_size * 8)

        # Collect all targets on GPU to minimize device transfers
        print("Collecting conditioning targets...")
        all_conditioning_targets = []
        for _, (_, targets) in enumerate(conditioning_dataloader):
            all_conditioning_targets.append(targets)
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
            # Note: Autocast disabled to avoid bfloat16/float16 compatibility issues
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
        
        try:
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('X_train', data=seq_indices.cpu().numpy())
                f.create_dataset('Y_train', data=targets.cpu().numpy())
                f.attrs['iteration'] = iteration
                f.attrs['num_samples'] = len(sequences)
                f.attrs['sequence_length'] = sequences.shape[1]
            
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
                f.attrs['sequence_length'] = sequences.shape[1]
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
                                          test_data_path: str, num_models: int = 1) -> Dict[str, Any]:
        """Train multiple DeepSTARR oracles on the given dataset and return aggregated metrics.

        Modified from Section 4.2: Now training 3 DeepSTARR models (reduced from 5) using
        different sets of generated sequences for faster experimentation.
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
        
        dev_correlations = []  # Dev task (index 0) correlations from all models
        hk_correlations = []   # Housekeeping task (index 1) correlations from all models
        checkpoints = []
        
        print(f"Training {num_models} oracle models for dataset: {os.path.basename(dataset_path)} (optimized from 5 to 3)")
        
        dataset_stem = Path(dataset_path).stem
        # Support legacy single-checkpoint naming (no per-model suffix), if present
        legacy_ckpt_path = os.path.join(work_dir, f"oracle_DeepSTARR_{dataset_stem}.ckpt")
        
        for model_idx in range(num_models):
            print(f"  Training oracle model {model_idx + 1}/{num_models}...")
            ckpt_name = f"oracle_DeepSTARR_{dataset_stem}_m{model_idx}"
            ckpt_path = os.path.join(work_dir, f"{ckpt_name}.ckpt")
            
            # Use different seed for each model
            model_seed = self.seed + model_idx
            
            try:
                if os.path.exists(ckpt_path) or os.path.exists(legacy_ckpt_path):
                    # Resume: load model from checkpoint
                    use_path = ckpt_path if os.path.exists(ckpt_path) else legacy_ckpt_path
                    trained_model = load_pl_deepstarr_from_checkpoint(use_path, batch_size=128, patience=10)
                    checkpoint_path = use_path
                    print(f"    Found existing checkpoint, skipping training: {use_path}")
                else:
                    # Train model
                    train_result = training_with_PL(
                        dataset_path=dataset_path,
                        train_max_epochs=100,
                        batch_size=128,
                        patience=10,
                        verbose=False,
                        seed=model_seed,
                        out_dir=work_dir,
                        filename=ckpt_name,
                        use_evoaug=self.use_evoaug,
                    )
                    trained_model = train_result["model"]
                    checkpoint_path = train_result["checkpoint"]
                
                if trained_model is not None and checkpoint_path is not None:
                    # Evaluate on test set
                    trained_model.eval()
                    trained_model = trained_model.to(self.device)
                    
                    with torch.no_grad():
                        predictions = trained_model.predict_custom(test_oracle_input.to(self.device))
                    
                    # Compute correlations for both tasks
                    pred_dev = predictions[:, 0].cpu().numpy()
                    target_dev = test_targets[:, 0].cpu().numpy()
                    dev_corr = float(np.corrcoef(pred_dev, target_dev)[0, 1])
                    dev_correlations.append(dev_corr)
                    
                    pred_hk = predictions[:, 1].cpu().numpy()
                    target_hk = test_targets[:, 1].cpu().numpy()
                    hk_corr = float(np.corrcoef(pred_hk, target_hk)[0, 1])
                    hk_correlations.append(hk_corr)
                    
                    checkpoints.append(checkpoint_path)
                    print(f"    Model {model_idx + 1}: Dev={dev_corr:.4f}, HK={hk_corr:.4f}")
                else:
                    print(f"    Model {model_idx + 1}: Training or loading failed")
                    dev_correlations.append(np.nan)
                    hk_correlations.append(np.nan)
                    checkpoints.append(None)
                    
            except Exception as e:
                print(f"    Model {model_idx + 1}: Error during training/evaluation: {e}")
                dev_correlations.append(np.nan)
                hk_correlations.append(np.nan)
                checkpoints.append(None)
        
        # Calculate aggregate statistics
        valid_dev = [x for x in dev_correlations if not np.isnan(x)]
        valid_hk = [x for x in hk_correlations if not np.isnan(x)]
        
        results = {
            "dev_correlations": dev_correlations,
            "hk_correlations": hk_correlations,
            "checkpoints": checkpoints,
            "avg_dev_pearson": np.mean(valid_dev) if valid_dev else np.nan,
            "std_dev_pearson": np.std(valid_dev) if valid_dev else np.nan,
            "avg_hk_pearson": np.mean(valid_hk) if valid_hk else np.nan,
            "std_hk_pearson": np.std(valid_hk) if valid_hk else np.nan,
            "num_successful_models": len(valid_dev)
        }
        
        print(f"  Summary: Dev={results['avg_dev_pearson']:.4f}±{results['std_dev_pearson']:.4f}, "
              f"HK={results['avg_hk_pearson']:.4f}±{results['std_hk_pearson']:.4f} "
              f"({results['num_successful_models']}/{num_models} models)")
        
        return results
    
    def run_iterative_augmentation_experiment(self, model_checkpoint: str, oracle_checkpoint: str, 
                                            data_path: str, output_dir: str, 
                                            max_iterations: int = 6,
                                            num_steps: Optional[int] = None, 
                                            batch_size: int = 32,
                                            architecture: str = 'transformer',
                                            num_oracle_models: int = 1) -> Dict[str, Any]:

        """
        Run iterative augmentation experiment following the paper's methodology:
        1. Baseline: 25% of original data (100,569 sequences)
        2. Original + 1 set of generated sequences (201,138 sequences)  
        3. Original + 2 sets of generated sequences (301,707 sequences)
        4. Original + 3 sets of generated sequences (402,276 sequences)
        
        For each augmented dataset, train 5 DeepSTARR oracles and evaluate on test set.
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
        
        # Get subset dataset size (25% baseline)
        train_ds = self._get_dataset_split(data_path, 'train')
        subset_sequences, subset_targets = self._extract_sequences_targets(train_ds)

        # Each iteration adds 25% of original training data size (100,569 samples)
        baseline_size = 100569
        generation_size = baseline_size  # Each generation is same size as baseline (25% of original)
        target_sizes = {i: baseline_size + (i * generation_size) for i in range(max_iterations + 1)}
        # Trim baseline to 100,569 if larger
        if subset_sequences.shape[0] > target_sizes[0]:
            subset_sequences = subset_sequences[:target_sizes[0]]
            subset_targets = subset_targets[:target_sizes[0]]
        elif subset_sequences.shape[0] < target_sizes[0]:
            print(f"Warning: baseline subset size {subset_sequences.shape[0]} is less than expected {target_sizes[0]}")

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
        print("DEEPSTARR ITERATIVE AUGMENTATION EXPERIMENT")
        print("=" * 70)
        print(f"Model checkpoint: {model_checkpoint}")
        print(f"Oracle checkpoint (initial evaluation): {oracle_checkpoint}")
        print(f"Subset data size (25% baseline): {subset_size} samples")
        print(f"Max iterations: {max_iterations}")
        print(f"Sampling steps: {num_steps}")
        print(f"Number of oracle models per condition: {num_oracle_models}")
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
            }
        else:
            # Keep prior settings but update dynamic ones
            results['experiment_config'].update({
                'max_iterations': max_iterations,
                'num_steps': num_steps,
                'batch_size': batch_size,
                'num_oracle_models': num_oracle_models,
                'seed': self.seed,
            })
        
        # Train/evaluate oracles for iteration 0 only if not already recorded
        if 0 not in existing_iterations:
            iter0_oracle_dir = os.path.join(oracles_dir, "iteration_0")
            iter0_train_info = self._train_multiple_oracles_for_dataset(
                iter0_path, iter0_oracle_dir, data_path, num_oracle_models
            )
            
            # Record iteration 0 results
            results['iteration_results'].append({
                'iteration': 0,
                'dataset_path': iter0_path,
                'dataset_size': subset_size,
                'original_size': subset_size,
                'generated_size': 0,
                'oracle_checkpoints': iter0_train_info.get("checkpoints"),
                'dev_correlations': iter0_train_info.get("dev_correlations"),
                'hk_correlations': iter0_train_info.get("hk_correlations"),
                'avg_test_pearson_dev': iter0_train_info.get("avg_dev_pearson"),
                'std_test_pearson_dev': iter0_train_info.get("std_dev_pearson"),
                'avg_test_pearson_hk': iter0_train_info.get("avg_hk_pearson"),
                'std_test_pearson_hk': iter0_train_info.get("std_hk_pearson"),
                'num_successful_models': iter0_train_info.get("num_successful_models"),
                'description': "Baseline: 25% original data"
            })
            
            print(f"\nIteration 0 (Baseline): {subset_size} samples")
            print(f"Test Pearson Dev: {iter0_train_info.get('avg_dev_pearson'):.4f} ± {iter0_train_info.get('std_dev_pearson'):.4f}")
            print(f"Test Pearson HK:  {iter0_train_info.get('avg_hk_pearson'):.4f} ± {iter0_train_info.get('std_dev_pearson'):.4f}")
            
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
                    dev_corrs = recorded.get('dev_correlations') or []
                    # Count valid non-NaN correlations
                    valid_count = 0
                    for v in dev_corrs:
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
                
                # Store or update results entry for this iteration
                updated = False
                for idx, item in enumerate(results['iteration_results']):
                    if int(item.get('iteration', -1)) == iteration:
                        results['iteration_results'][idx].update({
                            'dataset_path': iter_path,
                            'dataset_size': int(total_size),
                            'original_size': subset_size,
                            'generated_size': int(generated_size),
                            'oracle_checkpoints': train_info.get("checkpoints"),
                            'dev_correlations': train_info.get("dev_correlations"),
                            'hk_correlations': train_info.get("hk_correlations"),
                            'avg_test_pearson_dev': train_info.get("avg_dev_pearson"),
                            'std_test_pearson_dev': train_info.get("std_dev_pearson"),
                            'avg_test_pearson_hk': train_info.get("avg_hk_pearson"),
                            'std_test_pearson_hk': train_info.get("std_hk_pearson"),
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
                        'dev_correlations': train_info.get("dev_correlations"),
                        'hk_correlations': train_info.get("hk_correlations"),
                        'avg_test_pearson_dev': train_info.get("avg_dev_pearson"),
                        'std_test_pearson_dev': train_info.get("std_dev_pearson"),
                        'avg_test_pearson_hk': train_info.get("avg_hk_pearson"),
                        'std_test_pearson_hk': train_info.get("std_hk_pearson"),
                        'num_successful_models': train_info.get("num_successful_models"),
                        'description': f"Original + {iteration} set{'s' if iteration > 1 else ''} of generated sequences"
                    })
                
                print(f"Test Pearson Dev: {train_info.get('avg_dev_pearson'):.4f} ± {train_info.get('std_dev_pearson'):.4f}")
                print(f"Test Pearson HK:  {train_info.get('avg_hk_pearson'):.4f} ± {train_info.get('std_hk_pearson'):.4f}")
                
                # Persist results incrementally after each iteration
                try:
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                except Exception as e:
                    print(f"Warning: failed to save results after iteration {iteration}: {e}")
                
                # Move on to next iteration
                continue
            
            # Otherwise: sample new sequences for this iteration and then train oracles
            # NEW LOGIC: Generate samples to add to PREVIOUS iteration's complete dataset
            current_target_size = target_sizes.get(iteration, subset_size)
            if iteration == 1:
                # First augmentation: start from baseline
                previous_size = subset_size
            else:
                # Subsequent augmentations: start from previous iteration's total size
                previous_size = target_sizes.get(iteration - 1, subset_size)

            samples_to_generate = current_target_size - previous_size  # Always generates baseline_size (25%) new samples
            print(f"  Previous iteration size: {previous_size}, Target size: {current_target_size}")
            print(f"  Generating {samples_to_generate} new samples (25% of original training data)")

            conditioning_dataloader = self.create_conditioning_dataloader(data_path, samples_to_generate, batch_size)
            
            # Sample new sequences conditioned on test set labels
            print(f"Sampling {samples_to_generate} sequences for iteration {iteration} using test set conditioning...")
            sampled_sequences, sampled_targets = self.sample_sequences_for_iteration(
                model, graph, noise, conditioning_dataloader, num_steps, show_progress=True
            )
            
            # NEW LOGIC: Build upon the complete dataset from the previous iteration
            if iteration == 1:
                # First augmentation: start from baseline + new samples
                subset_sequences_onehot = F.one_hot(subset_sequences.long(), num_classes=4).float()
                current_sequences = torch.cat([subset_sequences_onehot, sampled_sequences], dim=0)
                current_targets = torch.cat([subset_targets, sampled_targets], dim=0)
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
                    current_targets = torch.cat([prev_targets, sampled_targets], dim=0)
                    print(f"  Building from previous iteration ({len(prev_sequences)}) + new samples ({len(sampled_sequences)})")
                else:
                    print(f"  Warning: Previous iteration file not found, falling back to accumulated method")
                    # Fallback to original logic
                    self.accumulated_sequences.append(sampled_sequences)
                    self.accumulated_targets.append(sampled_targets)
                    subset_sequences_onehot = F.one_hot(subset_sequences.long(), num_classes=4).float()
                    all_sequences = [subset_sequences_onehot] + self.accumulated_sequences
                    all_targets = [subset_targets] + self.accumulated_targets
                    current_sequences = torch.cat(all_sequences, dim=0)
                    current_targets = torch.cat(all_targets, dim=0)

            # Update accumulated sequences for potential fallback use
            if iteration == 1:
                self.accumulated_sequences = [sampled_sequences]
                self.accumulated_targets = [sampled_targets]
            else:
                self.accumulated_sequences.append(sampled_sequences)
                self.accumulated_targets.append(sampled_targets)
            
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
            print(f"  - Original (25%): {subset_size}")
            print(f"  - Generated: {generated_size}")
            print(f"  - Ratio: {generated_size/subset_size:.2f}x augmentation")
            
            # Save dataset for this iteration
            iter_path = self.save_iteration_dataset(current_sequences, current_targets, iter_path, iteration)
            
            # Train multiple oracles on augmented dataset and evaluate on test set
            oracle_iter_dir = os.path.join(oracles_dir, f"iteration_{iteration}")
            train_info = self._train_multiple_oracles_for_dataset(
                iter_path, oracle_iter_dir, data_path, num_oracle_models
            )
            
            # Store results
            results['iteration_results'].append({
                'iteration': iteration,
                'dataset_path': iter_path,
                'dataset_size': int(total_size),
                'original_size': subset_size,
                'new_samples_this_iteration': len(sampled_sequences),
                'generated_size': int(generated_size),
                'oracle_checkpoints': train_info.get("checkpoints"),
                'dev_correlations': train_info.get("dev_correlations"),
                'hk_correlations': train_info.get("hk_correlations"),
                'avg_test_pearson_dev': train_info.get("avg_dev_pearson"),
                'std_test_pearson_dev': train_info.get("std_dev_pearson"),
                'avg_test_pearson_hk': train_info.get("avg_hk_pearson"),
                'std_test_pearson_hk': train_info.get("std_hk_pearson"),
                'num_successful_models': train_info.get("num_successful_models"),
                'description': f"Original + {iteration} set{'s' if iteration > 1 else ''} of generated sequences"
            })
            
            print(f"Test Pearson Dev: {train_info.get('avg_dev_pearson'):.4f} ± {train_info.get('std_dev_pearson'):.4f}")
            print(f"Test Pearson HK:  {train_info.get('avg_hk_pearson'):.4f} ± {train_info.get('std_hk_pearson'):.4f}")
            
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
    parser = argparse.ArgumentParser(description='DeepSTARR Iterative Augmentation Experiment')
    parser.add_argument('--model_checkpoint', required=True, help='Path to trained D3 model checkpoint')
    parser.add_argument('--oracle_checkpoint', required=False, default='', help='(Optional) Path to oracle model checkpoint for initial eval log')
    parser.add_argument('--data_path', required=True, help='Path to subset data file (25% of DeepSTARR)')
    parser.add_argument('--output_dir', required=True, help='Directory to save iteration datasets and oracles')
    parser.add_argument('--config', help='Path to config file (optional)')
    parser.add_argument('--max_iterations', type=int, default=6, help='Maximum iterations (default: 6)')
    parser.add_argument('--num_steps', type=int, help='Sampling steps (default: sequence length)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--architecture', choices=['transformer', 'convolutional'], 
                       default='transformer', help='Model architecture')
    parser.add_argument('--seed', type=int, default=42, help='Global seed for reproducibility')
    parser.add_argument('--num_oracle_models', type=int, default=3, help='Number of oracle models to train per condition (default: 3)')
    parser.add_argument('--use_evoaug', action='store_true', help='Enable EvoAug for training')
    parser.add_argument('--class_balance', action='store_true', help='Balance conditioning labels across 4 classes (hi/lo combinations)')
    
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
    sampler = DeepSTARRIterativeAugmentationSampler(config, seed=int(args.seed), use_evoaug=args.use_evoaug, class_balance=args.class_balance)
    
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
        
        # Print summary
        print("\n" + "=" * 70)
        print("EXPERIMENT SUMMARY")
        print("=" * 70)
        for result in results['iteration_results']:
            dev_avg = result['avg_test_pearson_dev']
            dev_std = result['std_test_pearson_dev']
            hk_avg = result['avg_test_pearson_hk']
            hk_std = result['std_test_pearson_hk']
            
            print(f"Iteration {result['iteration']}: {result['dataset_size']} samples")
            if dev_avg is not None and hk_avg is not None:
                print(f"  Dev task:  {dev_avg:.4f} ± {dev_std:.4f}")
                print(f"  HK task:   {hk_avg:.4f} ± {hk_std:.4f}")
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

