#!/usr/bin/env python3
"""
MPRA Training Script

This script provides training functionality specifically for the MPRA dataset,
inheriting from the base training classes and implementing MPRA-specific
model creation and data loading.
"""

import os
import sys
from pathlib import Path
import datetime


# Package imports

from scripts.train import BaseD3LightningModule, BaseD3DataModule, BaseTrainer, parse_base_args
from model_zoo.mpra.models import create_model
from model_zoo.mpra.data import get_mpra_datasets, get_mpra_dataloaders
from model_zoo.mpra.sp_mse_callback import create_mpra_sp_mse_callback
from omegaconf import OmegaConf


class MPRALightningModule(BaseD3LightningModule):
    """Lightning module specifically for MPRA dataset."""
    
    def __init__(self, cfg, architecture: str = 'transformer'):
        super().__init__(cfg, dataset_name='mpra')
        self.architecture = architecture
        
    def create_model(self):
        """Create MPRA-specific model."""
        return create_model(self.cfg, self.architecture)
        
    def process_batch(self, batch):
        """Process MPRA batch data."""
        # MPRA data comes as (inputs, targets) pairs
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            return inputs, targets
        else:
            raise ValueError(f"Expected (inputs, targets) pair, got {type(batch)}")


class MPRADataModule(BaseD3DataModule):
    """Data module specifically for MPRA dataset."""
    
    def __init__(self, cfg):
        super().__init__(cfg, dataset_name='mpra')
        
    def setup(self, stage: str = None):
        """Setup MPRA datasets."""
        # Use MPRA-specific data loading
        self.train_ds, self.val_ds = get_mpra_datasets()
        print(f"MPRA dataset loaded: {len(self.train_ds)} train, {len(self.val_ds)} val samples")


class MPRATrainer(BaseTrainer):
    """Trainer specifically for MPRA dataset."""
    
    def __init__(self, architecture: str, config_path: str = None, work_dir: str = None):
        # Load MPRA config
        if config_path:
            cfg = OmegaConf.load(config_path)
        else:
            # Use default MPRA config
            config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            cfg = OmegaConf.load(config_file)
            
        super().__init__(cfg, 'mpra', work_dir)
        self.architecture = architecture
        
    def create_lightning_module(self):
        """Create MPRA Lightning module."""
        return MPRALightningModule(self.cfg, self.architecture)
        
    def create_data_module(self):
        """Create MPRA data module."""
        return MPRADataModule(self.cfg)
    
    def setup_callbacks(self):
        """Setup training callbacks including dataset-specific SP-MSE callback."""
        callbacks = super().setup_callbacks()
        
        # Add MPRA-specific SP-MSE callback if enabled
        sp_mse_callback = create_mpra_sp_mse_callback(self.cfg)
        if sp_mse_callback is not None:
            callbacks.append(sp_mse_callback)
        
        return callbacks


def main():
    """Main training function."""
    parser = parse_base_args()
    parser.description = 'MPRA Training Script'
    args = parser.parse_args()
    
    # Create trainer
    trainer = MPRATrainer(
        architecture=args.architecture,
        config_path=args.config,
        work_dir=args.work_dir
    )
    
    # Override WandB settings if provided
    if args.wandb_project:
        trainer.cfg.wandb.project = args.wandb_project
    if args.wandb_name:
        trainer.cfg.wandb.name = args.wandb_name
    
    # Train
    try:
        trainer.train(resume_from=args.resume_from)
        return 0
    except Exception as e:
        print(f"Training failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())