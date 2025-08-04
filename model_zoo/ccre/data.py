"""
cCRE Dataset Loader

This module provides dataset loading functionality specific to the cCRE dataset.
It handles loading the cCRE data format and provides appropriate preprocessing for
512bp sequences without labels.
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, DistributedSampler
from typing import Tuple, Optional
from utils.data_utils import cycle_loader


class cCREDataset(Dataset):
    """
    cCRE dataset loader.
    
    Loads cCRE H5 files and provides proper preprocessing for D3 training.
    The dataset consists of one-hot encoded DNA sequences of 512bp length
    without any labels (unsupervised learning).
    """
    
    def __init__(self, h5_file_path: str, split: str = 'train', 
                 train_ratio: float = 0.95, valid_ratio: float = 0.05,
                 seed: int = 42):
        """
        Initialize the cCRE dataset.
        
        Args:
            h5_file_path: Path to the cCRE H5 data file
            split: Dataset split ('train', 'valid')
            train_ratio: Ratio of data to use for training (default: 0.95)
            valid_ratio: Ratio of data to use for validation (default: 0.05)
            seed: Random seed for reproducible splits
        """
        self.h5_file_path = h5_file_path
        self.split = split.lower()
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.seed = seed
        
        if not os.path.exists(h5_file_path):
            raise FileNotFoundError(f"cCRE data file not found: {h5_file_path}")
        
        if abs(train_ratio + valid_ratio - 1.0) > 1e-6:
            raise ValueError(f"train_ratio + valid_ratio must equal 1.0, got {train_ratio + valid_ratio}")
        
        if valid_ratio == 0.0:
            print(f"Note: Using train_ratio=1.0, valid_ratio=0.0 (no validation split)")
        
        # Load and preprocess data
        self.X = self._load_data()
        
    def _load_data(self) -> torch.Tensor:
        """Load and preprocess data from H5 file with train/valid splitting."""
        with h5py.File(self.h5_file_path, 'r') as data:
            # Load all sequences from the single 'seqs' key
            X = torch.tensor(np.array(data['seqs'])).permute(0, 2, 1)
            
            # Convert one-hot to indices for D3 processing
            # X shape: (n_samples, 4, seq_length) -> (n_samples, seq_length)
            X = torch.argmax(X, dim=1)
            
            # Create reproducible train/valid split
            total_samples = X.shape[0]
            np.random.seed(self.seed)
            indices = np.random.permutation(total_samples)
            
            train_size = int(total_samples * self.train_ratio)
            
            if self.split == 'train':
                selected_indices = indices[:train_size]
            elif self.split == 'valid':
                selected_indices = indices[train_size:]
                if len(selected_indices) == 0:
                    print(f"Warning: Validation split is empty (valid_ratio={self.valid_ratio}). "
                          f"Consider setting valid_ratio > 0 for proper validation.")
            else:
                raise ValueError(f"Unknown split: {self.split}. Only 'train' and 'valid' are supported.")
            
            X = X[selected_indices]
            
        return X
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # For cCRE, we only return sequences (no labels)
        # Return the sequence twice to maintain compatibility with training code
        # that expects (input, target) pairs - target will be ignored
        return self.X[idx], self.X[idx]


def get_ccre_datasets(h5_file_path: str, train_ratio: float = 0.95, 
                     valid_ratio: float = 0.05, seed: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Get cCRE train and validation datasets.
    
    Args:
        h5_file_path: Path to the cCRE H5 data file
        train_ratio: Ratio of data to use for training (default: 0.95)
        valid_ratio: Ratio of data to use for validation (default: 0.05)  
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_dataset, valid_dataset)
    """
    
    # Pass in data file path directly from config.paths.data_file
    train_set = cCREDataset(h5_file_path, split='train', 
                           train_ratio=train_ratio, valid_ratio=valid_ratio, seed=seed)
    valid_set = cCREDataset(h5_file_path, split='valid',
                           train_ratio=train_ratio, valid_ratio=valid_ratio, seed=seed)
    
    return train_set, valid_set


def get_ccre_dataloaders(config, distributed: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Get cCRE dataloaders for training and validation.
    
    Args:
        config: Configuration object with training parameters
        distributed: Whether to use distributed training
        
    Returns:
        Tuple of (train_loader, valid_loader)
    """
    # Validation checks
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(
            f"Train Batch Size {config.training.batch_size} is not divisible by "
            f"{config.ngpus} gpus with accumulation {config.training.accum}."
        )
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(
            f"Eval Batch Size {config.eval.batch_size} is not divisible by "
            f"{config.ngpus} gpus with accumulation {config.training.accum}."
        )
    
    # Get split configuration from config
    train_ratio = getattr(config.data, 'train_ratio', 0.95)
    valid_ratio = getattr(config.data, 'valid_ratio', 0.05)
    split_seed = getattr(config.data, 'split_seed', 42)
    
    # Get datasets
    train_set, valid_set = get_ccre_datasets(
        config.paths.data_file, 
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=split_seed
    )
    
    print(f"cCRE dataset sizes - Train: {len(train_set)}, Valid: {len(valid_set)}")
    
    # Setup samplers
    if distributed:
        train_sampler = DistributedSampler(train_set)
        valid_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        valid_sampler = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    )
    
    valid_loader = DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=valid_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )
    
    return train_loader, valid_loader


def get_ccre_dataloaders_with_cycle(config, distributed: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Get cCRE dataloaders with cycle_loader applied for training and validation.
    
    Args:
        config: Configuration object with training parameters
        distributed: Whether to use distributed training
        
    Returns:
        Tuple of (train_loader, valid_loader) with cycle_loader applied
    """
    train_loader, valid_loader = get_ccre_dataloaders(config, distributed)
    
    # Apply cycle_loader
    train_sampler = train_loader.sampler if hasattr(train_loader, 'sampler') else None
    valid_sampler = valid_loader.sampler if hasattr(valid_loader, 'sampler') else None
    
    cycled_train_loader = cycle_loader(train_loader, train_sampler)
    cycled_valid_loader = cycle_loader(valid_loader, valid_sampler)
    
    return cycled_train_loader, cycled_valid_loader