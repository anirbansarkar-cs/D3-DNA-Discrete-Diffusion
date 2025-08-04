"""
DeepSTARR Dataset Loader

This module provides dataset loading functionality specific to the DeepSTARR dataset.
It handles loading the DeepSTARR data format and provides appropriate preprocessing.
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, DistributedSampler
from typing import Tuple, Optional
from utils.data_utils import cycle_loader


class ATACSeqDataset(Dataset):
    """
    DeepSTARR dataset loader.
    
    Loads AtacSeq H5 files and provides proper preprocessing for D3 training.
    The dataset consists of one-hot encoded DNA sequences and their corresponding
    enhancer activity labels across 18 tissue types.
    """
    
    def __init__(self, h5_file_path: str, split: str = 'train'):
        """
        Initialize the ATACSeq dataset.
        
        Args:
            h5_file_path: Path to the DeepSTARR H5 data file
            split: Dataset split ('train', 'valid', 'test')
        """
        self.h5_file_path = h5_file_path
        self.split = split.lower()
        
        if not os.path.exists(h5_file_path):
            raise FileNotFoundError(f"DeepSTARR data file not found: {h5_file_path}")
        
        # Load and preprocess data
        self.X, self.y = self._load_data()
        
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess data from H5 file."""
        with h5py.File(self.h5_file_path, 'r') as data:
            # Determine which split to load
            if self.split == 'train':
                X = torch.tensor(np.array(data['x_train']))
                y = torch.tensor(np.array(data['y_train']))
            elif self.split == 'valid':
                X = torch.tensor(np.array(data['x_valid']))
                y = torch.tensor(np.array(data['y_valid']))
            elif self.split == 'test':
                X = torch.tensor(np.array(data['x_test']))
                y = torch.tensor(np.array(data['y_test']))
            else:
                raise ValueError(f"Unknown split: {self.split}")
            
            # Permute dimensions: (n_samples, seq_length, 4) -> (n_samples, 4, seq_length)
            X = X.permute(0, 2, 1)

            # Convert one-hot to indices for D3 processing
            X = torch.argmax(X, dim=1)
            
        return X, y
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# def get_deepstarr_datasets(data_dir: Optional[str] = None) -> Tuple[Dataset, Dataset]:
def get_atacseq_datasets(h5_file_path: str) -> Tuple[Dataset, Dataset]:
    """
    Get ATACSeq train and validation datasets.
    
    Args:
        data_dir: Directory containing the data files. If None, uses default location.
        
    Returns:
        Tuple of (train_dataset, valid_dataset)
    """
    
    # pass in data file path directly from config.paths.data_file
    train_set = ATACSeqDataset(h5_file_path, split='train')
    valid_set = ATACSeqDataset(h5_file_path, split='valid')
    
    return train_set, valid_set


def get_atacseq_dataloaders(config, distributed: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Get DeepSTARR dataloaders for training and validation.
    
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
    train_set, valid_set = get_atacseq_datasets(config.paths.data_file)
    
    print(f"ATACSeq dataset sizes - Train: {len(train_set)}, Valid: {len(valid_set)}")
    
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


def get_atacseq_dataloaders_with_cycle(config, distributed: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Get ATACSeq dataloaders with cycle_loader applied for training and validation.
    
    Args:
        config: Configuration object with training parameters
        distributed: Whether to use distributed training
        
    Returns:
        Tuple of (train_loader, valid_loader) with cycle_loader applied
    """
    train_loader, valid_loader = get_atacseq_dataloaders(config, distributed)
    
    # Apply cycle_loader
    train_sampler = train_loader.sampler if hasattr(train_loader, 'sampler') else None
    valid_sampler = valid_loader.sampler if hasattr(valid_loader, 'sampler') else None
    
    cycled_train_loader = cycle_loader(train_loader, train_sampler)
    cycled_valid_loader = cycle_loader(valid_loader, valid_sampler)
    
    return cycled_train_loader, cycled_valid_loader