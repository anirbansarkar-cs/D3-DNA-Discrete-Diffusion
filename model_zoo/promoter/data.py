"""
Promoter Dataset Loader

This module provides dataset loading functionality specific to the Promoter dataset.
It handles loading promoter sequence data from NPZ files and provides appropriate 
preprocessing for D3 training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Tuple, Optional
from utils.data_utils import cycle_loader


class PromoterDataset(Dataset):
    """
    Promoter dataset loader.
    
    Loads promoter NPZ files and provides proper preprocessing for D3 training.
    The dataset consists of one-hot encoded DNA sequences and their corresponding
    regulatory activity labels.
    """
    
    def __init__(self, data_file: str, split: str = 'train'):
        """
        Initialize the Promoter dataset.
        
        Args:
            data_file: Path to the Promoter NPZ data file
            split: Dataset split ('train', 'valid', 'test')
        """
        self.data_file = data_file
        self.split = split.lower()
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Promoter data file not found: {data_file}")
        
        # Load and preprocess data
        self.X, self.y = self._load_data()
        
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess data from NPZ file."""
        promo_data = np.load(self.data_file)
        
        # Determine which split to load
        if self.split == 'train':
            data = promo_data['train']
        elif self.split == 'valid':
            data = promo_data['valid'] 
        elif self.split == 'test':
            data = promo_data['test']
        else:
            raise ValueError(f"Unknown split: {self.split}")
            
        # Extract sequences and labels from the data
        # data shape: (N, 1024, 6) -> seq_one_hot: (N, 1024, 4), label: (N, 1024, 1)
        seq_one_hot = data[:, :, :4]  # One-hot encoded sequences
        label = data[:, :, 4:5]       # Regulatory activity labels
        
        # Convert to tensors
        seq_one_hot = torch.tensor(seq_one_hot, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        # Convert one-hot to indices for D3 processing
        # seq_one_hot shape: (n_samples, seq_length, 4) -> (n_samples, seq_length)
        X = torch.argmax(seq_one_hot, dim=-1)
        
        return X, label
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_promoter_datasets(data_file: str) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get Promoter train, validation, and test datasets.
    
    Args:
        data_file: Path to the Promoter NPZ data file
        
    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """
    train_set = PromoterDataset(data_file, split='train')
    valid_set = PromoterDataset(data_file, split='valid')
    test_set = PromoterDataset(data_file, split='test')
    
    return train_set, valid_set, test_set


def get_promoter_dataloaders(config, distributed: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Get Promoter dataloaders for training and validation.
    
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
    train_set, valid_set, _ = get_promoter_datasets(config.paths.data_file)
    
    print(f"Promoter dataset sizes - Train: {len(train_set)}, Valid: {len(valid_set)}")
    
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


def get_promoter_dataloaders_with_cycle(config, distributed: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Get Promoter dataloaders with cycle_loader applied for training and validation.
    
    Args:
        config: Configuration object with training parameters
        distributed: Whether to use distributed training
        
    Returns:
        Tuple of (train_loader, valid_loader) with cycle_loader applied
    """
    train_loader, valid_loader = get_promoter_dataloaders(config, distributed)
    
    # Apply cycle_loader
    train_sampler = train_loader.sampler if hasattr(train_loader, 'sampler') else None
    valid_sampler = valid_loader.sampler if hasattr(valid_loader, 'sampler') else None
    
    cycled_train_loader = cycle_loader(train_loader, train_sampler)
    cycled_valid_loader = cycle_loader(valid_loader, valid_sampler)
    
    return cycled_train_loader, cycled_valid_loader


