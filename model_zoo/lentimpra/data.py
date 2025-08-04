"""
LentIMPRA Dataset Loader

This module provides dataset loading functionality specific to the LentIMPRA dataset.
It handles loading the LentIMPRA data format and provides appropriate preprocessing.
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, DistributedSampler
from typing import Tuple, Optional
from utils.data_utils import cycle_loader


class LentIMPRADataset(Dataset):
    """
    LentIMPRA dataset loader.
    
    Loads LentIMPRA H5 files and provides proper preprocessing for D3 training.
    The dataset consists of one-hot encoded DNA sequences and their corresponding
    regulatory activity measurements from lentiviral MPRA experiments.
    """
    
    def __init__(self, h5_file_path: str, split: str = 'train'):
        """
        Initialize the LentIMPRA dataset.
        
        Args:
            h5_file_path: Path to the LentIMPRA H5 data file
            split: Dataset split ('train', 'valid', 'test')
        """
        self.h5_file_path = h5_file_path
        self.split = split.lower()
        
        if not os.path.exists(h5_file_path):
            raise FileNotFoundError(f"LentIMPRA data file not found: {h5_file_path}")
        
        # Load and preprocess data
        self.X, self.y = self._load_data()
        
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess data from H5 file."""
        with h5py.File(self.h5_file_path, 'r') as data:
            # Load using lentimpra-specific H5 keys: onehot_{split}, y_{split}
            if self.split == 'train':
                X = torch.tensor(np.array(data['onehot_train']))  # (N, 230, 4)
                y = torch.tensor(np.array(data['y_train']))       # (N, 1)
            elif self.split == 'valid':
                X = torch.tensor(np.array(data['onehot_valid']))  # (N, 230, 4)
                y = torch.tensor(np.array(data['y_valid']))       # (N, 1)
            elif self.split == 'test':
                X = torch.tensor(np.array(data['onehot_test']))   # (N, 230, 4)
                y = torch.tensor(np.array(data['y_test']))        # (N, 1)
            else:
                raise ValueError(f"Unknown split: {self.split}")
            
            # Convert one-hot to indices for D3 processing
            # X shape: (N, 230, 4) -> (N, 4, 230) -> (N, 230)
            X = X.permute(0, 2, 1)  # (N, 230, 4) -> (N, 4, 230)
            X = torch.argmax(X, dim=1)  # (N, 4, 230) -> (N, 230)
            
            # Ensure targets are proper shape (N, signal_dim)
            if y.dim() == 1:
                y = y.unsqueeze(1)  # (N,) -> (N, 1)
            # Keep (N, 1) shape for signal_dim=1
            
        return X, y
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_lentimpra_datasets(h5_file_path: str) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get LentIMPRA train, validation, and test datasets.
    
    Args:
        h5_file_path: Path to the LentIMPRA H5 data file
        
    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """
    train_set = LentIMPRADataset(h5_file_path, split='train')
    valid_set = LentIMPRADataset(h5_file_path, split='valid')
    test_set = LentIMPRADataset(h5_file_path, split='test')
    
    return train_set, valid_set, test_set


def get_lentimpra_dataloaders(config, distributed: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Get LentIMPRA dataloaders for training and validation.
    
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
    
    # Get datasets
    train_set, valid_set, _ = get_lentimpra_datasets(config.paths.data_file)
    
    print(f"LentIMPRA dataset sizes - Train: {len(train_set)}, Valid: {len(valid_set)}")
    
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


def get_lentimpra_dataloaders_with_cycle(config, distributed: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Get LentIMPRA dataloaders with cycle_loader applied for training and validation.
    
    Args:
        config: Configuration object with training parameters
        distributed: Whether to use distributed training
        
    Returns:
        Tuple of (train_loader, valid_loader) with cycle_loader applied
    """
    train_loader, valid_loader = get_lentimpra_dataloaders(config, distributed)
    
    # Apply cycle_loader
    train_sampler = train_loader.sampler if hasattr(train_loader, 'sampler') else None
    valid_sampler = valid_loader.sampler if hasattr(valid_loader, 'sampler') else None
    
    cycled_train_loader = cycle_loader(train_loader, train_sampler)
    cycled_valid_loader = cycle_loader(valid_loader, valid_sampler)
    
    return cycled_train_loader, cycled_valid_loader