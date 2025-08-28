#!/usr/bin/env python3
"""
DeepSTARR Training Script

This script provides training functionality specifically for the DeepSTARR dataset,
inheriting from the base training classes and implementing DeepSTARR-specific
model creation and data loading. Now supports both standard training and EvoAug training.
"""

import os
import sys
from pathlib import Path
import numpy as np
import random
import torch
import datetime


# Package imports

from scripts.train import BaseD3LightningModule, BaseD3DataModule, BaseTrainer, parse_base_args
from model_zoo.deepstarr.models import create_model
from model_zoo.deepstarr.data import get_deepstarr_datasets, get_deepstarr_evoaug_dataloaders, get_deepstarr_dataloaders
from model_zoo.deepstarr.sp_mse_callback import create_deepstarr_sp_mse_callback
from omegaconf import OmegaConf
from utils.utils import update_cfg_with_unknown_args

class DeepSTARRLightningModule(BaseD3LightningModule):
    """Lightning module specifically for DeepSTARR dataset."""
    
    def __init__(self, cfg, architecture: str = 'transformer'):
        super().__init__(cfg, dataset_name='deepstarr')
        self.architecture = architecture
        
    def create_model(self):
        """Create DeepSTARR-specific model."""
        return create_model(self.cfg, self.architecture)
        
    def process_batch(self, batch):
        """Process DeepSTARR batch data."""
        # DeepSTARR data comes as (inputs, targets) pairs
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            
            # Handle EvoAug one-hot encoded data: convert (batch_size, 4, seq_length) to (batch_size, seq_length)
            if len(inputs.shape) == 3 and inputs.shape[1] == 4:
                # Convert one-hot to indices: (batch_size, 4, seq_length) -> (batch_size, seq_length)
                inputs = torch.argmax(inputs, dim=1)
            
            return inputs, targets
        else:
            raise ValueError(f"Expected (inputs, targets) pair, got {type(batch)}")


class DeepSTARRDataModule(BaseD3DataModule):
    """Data module specifically for DeepSTARR dataset."""
    
    def __init__(self, cfg, use_evoaug: bool = False):
        super().__init__(cfg, dataset_name='deepstarr')
        self.use_evoaug = use_evoaug
        
    def setup(self, stage: str = None):
        """Setup DeepSTARR datasets."""
        if self.use_evoaug:
            # Use EvoAug datasets (one-hot format)
            from model_zoo.deepstarr.data import get_deepstarr_evoaug_datasets
            self.train_ds, self.val_ds, _ = get_deepstarr_evoaug_datasets(self.cfg.paths.data_file)
            print(f"DeepSTARR EvoAug dataset loaded: {len(self.train_ds)} train, {len(self.val_ds)} val samples")
        else:
            # Use standard datasets (index format for D3)
            self.train_ds, self.val_ds, _ = get_deepstarr_datasets(self.cfg.paths.data_file)
            print(f"DeepSTARR standard dataset loaded: {len(self.train_ds)} train, {len(self.val_ds)} val samples")
    
    def train_dataloader(self):
        """Create training dataloader."""
        from torch.utils.data import DataLoader
        
        if self.use_evoaug:
            # Use EvoAug dataloaders with augmentations
            train_loader, _ = get_deepstarr_evoaug_dataloaders(self.cfg, distributed=False)
            return train_loader
        else:
            # Use standard dataloaders
            return DataLoader(
                self.train_ds,
                batch_size=self.cfg.training.batch_size // (self.cfg.ngpus * self.cfg.training.accum),
                num_workers=2,
                pin_memory=True,
                shuffle=True,
                persistent_workers=True,
            )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        from torch.utils.data import DataLoader
        
        if self.use_evoaug:
            # Use EvoAug dataloaders with augmentations disabled for validation
            _, val_loader = get_deepstarr_evoaug_dataloaders(self.cfg, distributed=False)
            return val_loader
        else:
            # Use standard dataloaders
            return DataLoader(
                self.val_ds,
                batch_size=self.cfg.eval.batch_size // (self.cfg.ngpus * self.cfg.training.accum),
                num_workers=2,
                pin_memory=True,
                shuffle=False,
            )


class DeepSTARRTrainer(BaseTrainer):
    """Trainer specifically for DeepSTARR dataset."""
    
    def __init__(self, architecture: str, config_path: str = None, work_dir: str = None, use_evoaug: bool = False):
        # Load DeepSTARR config
        if config_path:
            cfg = OmegaConf.load(config_path)
        else:
            # Use default DeepSTARR config
            config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            cfg = OmegaConf.load(config_file)
            
        super().__init__(cfg, 'deepstarr', work_dir)
        self.architecture = architecture
        self.use_evoaug = use_evoaug
        
        # Update work directory to indicate EvoAug usage
        if self.use_evoaug:
            self.work_dir = self.work_dir.replace('deepstarr', 'deepstarr_evoaug')
        
    def create_lightning_module(self):
        """Create DeepSTARR Lightning module."""
        return DeepSTARRLightningModule(self.cfg, self.architecture)
        
    def create_data_module(self):
        """Create DeepSTARR data module."""
        return DeepSTARRDataModule(self.cfg, use_evoaug=self.use_evoaug)
    
    def setup_callbacks(self):
        """Setup training callbacks including dataset-specific SP-MSE callback."""
        callbacks = super().setup_callbacks()
        
        # Add DeepSTARR-specific SP-MSE callback if enabled
        sp_mse_callback = create_deepstarr_sp_mse_callback(self.cfg)
        if sp_mse_callback is not None:
            callbacks.append(sp_mse_callback)
        
        return callbacks


def main():
    """Main training function."""
    parser = parse_base_args()
    parser.description = 'DeepSTARR Training Script'
    parser.add_argument('--use_evoaug', action='store_true', 
                       help='Use EvoAug augmentations during training')
    args, unknown = parser.parse_known_args()

    # Set all seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create trainer (loads cfg)
    trainer = DeepSTARRTrainer(
        architecture=args.architecture,
        config_path=args.config,
        work_dir=args.work_dir,
        use_evoaug=args.use_evoaug,
    )

    # Override WandB settings if provided
    if args.wandb_project:
        trainer.cfg.wandb.project = args.wandb_project
    if args.wandb_name:
        trainer.cfg.wandb.name = args.wandb_name

    # override other unknown args (e.g. --paths.data_file)
    if unknown:
        update_cfg_with_unknown_args(trainer.cfg, unknown)
    
    # Print training mode
    if args.use_evoaug:
        print("=" * 60)
        print("TRAINING WITH EVOAUG AUGMENTATIONS")
        print("=" * 60)
        print("Stage 1: Training with EvoAug augmentations")
        print("Stage 2: Fine-tuning on original data (if enabled)")
        print("=" * 60)
    else:
        print("=" * 60)
        print("STANDARD TRAINING (NO AUGMENTATIONS)")
        print("=" * 60)
    
    # Train
    try:
        trainer.train(resume_from=args.resume_from)
        return 0
    except Exception as e:
        print(f"Training failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())