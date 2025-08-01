#!/usr/bin/env python3
"""
Promoter Training Script

This script provides training functionality specifically for the Promoter dataset,
inheriting from the base training classes and implementing Promoter-specific
model creation and data loading.
"""

import os
import sys
from pathlib import Path
import datetime


# Package imports

from scripts.train import BaseD3LightningModule, BaseD3DataModule, BaseTrainer, parse_base_args
from model_zoo.promoter.models import create_model
from model_zoo.promoter.data import get_promoter_datasets, get_promoter_dataloaders
from model_zoo.promoter.sp_mse_callback import create_promoter_sp_mse_callback
from omegaconf import OmegaConf
import torch


class PromoterLightningModule(BaseD3LightningModule):
    """Lightning module specifically for Promoter dataset."""
    
    def __init__(self, cfg, architecture: str = 'transformer'):
        super().__init__(cfg, dataset_name='promoter')
        self.architecture = architecture
        
    def create_model(self):
        """Create Promoter-specific model."""
        return create_model(self.cfg, self.architecture)
        
    def process_batch(self, batch):
        """
        Process Promoter batch data.
        
        Promoter data comes as (sequences, targets) tuple from the dataset.
        Sequences are already converted to indices, targets are the regulatory labels.
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            sequences, targets = batch
            # Sequences are already token indices from the dataset
            # Targets are the regulatory activity labels
            return sequences, targets
        else:
            raise ValueError(f"Expected (sequences, targets) tuple, got {type(batch)}")


class PromoterDataModule(BaseD3DataModule):
    """Data module specifically for Promoter dataset."""
    
    def __init__(self, cfg):
        super().__init__(cfg, dataset_name='promoter')
        
    def setup(self, stage: str = None):
        """Setup Promoter datasets."""
        # Use Promoter-specific data loading with data_file from config
        self.train_ds, self.val_ds, _ = get_promoter_datasets(self.cfg.paths.data_file)
        print(f"Promoter dataset loaded: {len(self.train_ds)} train, {len(self.val_ds)} val samples")


class PromoterTrainer(BaseTrainer):
    """Trainer specifically for Promoter dataset."""
    
    def __init__(self, architecture: str, config_path: str = None, work_dir: str = None):
        # Load Promoter config
        if config_path:
            cfg = OmegaConf.load(config_path)
        else:
            # Use default Promoter config
            config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            cfg = OmegaConf.load(config_file)
            
        super().__init__(cfg, 'promoter', work_dir)
        self.architecture = architecture
        
    def create_lightning_module(self):
        """Create Promoter Lightning module."""
        return PromoterLightningModule(self.cfg, self.architecture)
        
    def create_data_module(self):
        """Create Promoter data module."""
        return PromoterDataModule(self.cfg)
    
    def setup_callbacks(self):
        """Setup training callbacks including dataset-specific SP-MSE callback."""
        callbacks = super().setup_callbacks()
        
        # Add Promoter-specific SP-MSE callback if enabled
        sp_mse_callback = create_promoter_sp_mse_callback(self.cfg)
        if sp_mse_callback is not None:
            callbacks.append(sp_mse_callback)
        
        return callbacks


def main():
    """Main training function."""
    parser = parse_base_args()
    parser.description = 'Promoter Training Script'
    args = parser.parse_args()
    
    # Create trainer
    trainer = PromoterTrainer(
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