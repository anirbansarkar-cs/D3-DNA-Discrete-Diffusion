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

# EvoAug imports (optional)
try:
    from evoaug.augment import (
        RandomDeletion, RandomRC, RandomInsertion,
        RandomTranslocation, RandomMutation, RandomNoise
    )
    from evoaug.evoaug import RobustLoader
    EvoAug_AVAILABLE = True
except ImportError:
    EvoAug_AVAILABLE = False
    print("Warning: EvoAug not available. Install with: pip install evoaug")


class DeepSTARRDataset(Dataset):
    """
    DeepSTARR dataset loader.
    
    Loads DeepSTARR H5 files and provides proper preprocessing for D3 training.
    The dataset consists of one-hot encoded DNA sequences and their corresponding
    enhancer activity labels for developmental and housekeeping promoters.
    """
    
    def __init__(self, h5_file_path: str, split: str = 'train'):
        """
        Initialize the DeepSTARR dataset.
        
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
                X = torch.tensor(np.array(data['X_train']))
                y = torch.tensor(np.array(data['Y_train']))
            elif self.split == 'valid':
                X = torch.tensor(np.array(data['X_valid']))
                y = torch.tensor(np.array(data['Y_valid']))
            elif self.split == 'test':
                X = torch.tensor(np.array(data['X_test']))
                y = torch.tensor(np.array(data['Y_test']))
            else:
                raise ValueError(f"Unknown split: {self.split}")
            
            # Convert one-hot to indices for D3 processing
            # X shape: (n_samples, 4, seq_length) -> (n_samples, seq_length)
            X = torch.argmax(X, dim=1)
            
        return X, y
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DeepSTARREvoAugDataset(Dataset):
    """
    DeepSTARR dataset wrapper that provides one-hot encoded sequences for EvoAug.
    
    EvoAug requires one-hot encoded sequences, so this class maintains the original
    one-hot format while providing the same interface as DeepSTARRDataset.
    """
    
    def __init__(self, h5_file_path: str, split: str = 'train'):
        """
        Initialize the DeepSTARR EvoAug dataset.
        
        Args:
            h5_file_path: Path to the DeepSTARR H5 data file
            split: Dataset split ('train', 'valid', 'test')
        """
        self.h5_file_path = h5_file_path
        self.split = split.lower()
        
        if not os.path.exists(h5_file_path):
            raise FileNotFoundError(f"DeepSTARR data file not found: {h5_file_path}")
        
        # Load and preprocess data (keep one-hot format for EvoAug)
        self.X, self.y = self._load_data()
        
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load data from H5 file, keeping one-hot format for EvoAug."""
        with h5py.File(self.h5_file_path, 'r') as data:
            # Determine which split to load
            if self.split == 'train':
                X = torch.tensor(np.array(data['X_train']))
                y = torch.tensor(np.array(data['Y_train']))
            elif self.split == 'valid':
                X = torch.tensor(np.array(data['X_valid']))
                y = torch.tensor(np.array(data['Y_valid']))
            elif self.split == 'test':
                X = torch.tensor(np.array(data['X_test']))
                y = torch.tensor(np.array(data['Y_test']))
            else:
                raise ValueError(f"Unknown split: {self.split}")
            
            # Keep one-hot format for EvoAug (X shape: n_samples, 4, seq_length)
            
        return X, y
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_evoaug_augmentation_list():
    """Create augmentation list with optimal DeepSTARR hyperparameters."""
    if not EvoAug_AVAILABLE:
        print("Warning: EvoAug not available. Returning empty augmentation list.")
        return []
    
    # Based on optimal DeepSTARR hyperparameters from the EvoAug paper
    augment_list = [
        RandomDeletion(delete_min=0, delete_max=20),
        # augment.RandomRC(rc_prob=0.5),
        # augment.RandomInsertion(insert_min=0, insert_max=20),
        RandomTranslocation(shift_min=0, shift_max=20),
        RandomMutation(mut_frac=0.05),
        RandomNoise(noise_mean=0, noise_std=0.2),
        ]
    
    return augment_list


def get_deepstarr_datasets(h5_file_path: str) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get DeepSTARR train and validation datasets.
    
    Args:
        h5_file_path: Path to the DeepSTARR H5 data file
        
    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """
    
    # pass in data file path directly from config.paths.data_file
    train_set = DeepSTARRDataset(h5_file_path, split='train')
    valid_set = DeepSTARRDataset(h5_file_path, split='valid')
    test_set = DeepSTARRDataset(h5_file_path, split='test')
    
    return train_set, valid_set, test_set


def get_deepstarr_evoaug_datasets(h5_file_path: str) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get DeepSTARR datasets in one-hot format for EvoAug training.
    
    Args:
        h5_file_path: Path to the DeepSTARR H5 data file
        
    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset) in one-hot format
    """
    
    train_set = DeepSTARREvoAugDataset(h5_file_path, split='train')
    valid_set = DeepSTARREvoAugDataset(h5_file_path, split='valid')
    test_set = DeepSTARREvoAugDataset(h5_file_path, split='test')
    
    return train_set, valid_set, test_set


def get_deepstarr_dataloaders(config, distributed: bool = True) -> Tuple[DataLoader, DataLoader]:
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
    train_set, valid_set, _ = get_deepstarr_datasets(config.paths.data_file)
    
    print(f"DeepSTARR dataset sizes - Train: {len(train_set)}, Valid: {len(valid_set)}")
    
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


def get_deepstarr_evoaug_dataloaders(config, distributed: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Get DeepSTARR EvoAug dataloaders for training and validation.
    Validations are disabled for the validation set.
    
    Args:
        config: Configuration object with training parameters
        distributed: Whether to use distributed training
        
    Returns:
        Tuple of (train_loader, valid_loader) with EvoAug augmentations
    """
    if not EvoAug_AVAILABLE:
        print("Warning: EvoAug not available. Falling back to standard dataloaders.")
        return get_deepstarr_dataloaders(config, distributed)
    
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
    
    # Get one-hot datasets for EvoAug
    train_set, valid_set, _ = get_deepstarr_evoaug_datasets(config.paths.data_file)
    
    print(f"DeepSTARR EvoAug dataset sizes - Train: {len(train_set)}, Valid: {len(valid_set)}")
    
    # Create augmentation list
    augment_list = create_evoaug_augmentation_list()
    
    # Setup samplers
    if distributed:
        train_sampler = DistributedSampler(train_set)
        valid_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        valid_sampler = None
    
    # Create EvoAug dataloaders
    train_loader = RobustLoader(
        base_dataset=train_set,
        augment_list=augment_list,
        max_augs_per_seq=2,  # DeepSTARR optimal: maximum 2 augmentations per sequence
        hard_aug=True,        # DeepSTARR uses hard setting: always apply exactly 2 augmentations
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    )
    
    # For validation, disable augmentations
    valid_loader = RobustLoader(
        base_dataset=valid_set,
        augment_list=augment_list,
        max_augs_per_seq=2,
        hard_aug=True,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=valid_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )
    valid_loader.disable_augmentations()
    
    return train_loader, valid_loader


def get_deepstarr_dataloaders_with_cycle(config, distributed: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Get DeepSTARR dataloaders with cycle_loader applied for training and validation.
    
    Args:
        config: Configuration object with training parameters
        distributed: Whether to use distributed training
        
    Returns:
        Tuple of (train_loader, valid_loader) with cycle_loader applied
    """
    train_loader, valid_loader, _ = get_deepstarr_dataloaders(config, distributed)
    
    # Apply cycle_loader
    train_sampler = train_loader.sampler if hasattr(train_loader, 'sampler') else None
    valid_sampler = valid_loader.sampler if hasattr(valid_loader, 'sampler') else None
    
    cycled_train_loader = cycle_loader(train_loader, train_sampler)
    cycled_valid_loader = cycle_loader(valid_loader, valid_sampler)
    
    return cycled_train_loader, cycled_valid_loader