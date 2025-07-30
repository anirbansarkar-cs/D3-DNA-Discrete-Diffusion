#!/usr/bin/env python3
"""
cCRE Training Script

This script provides training functionality specifically for the cCRE dataset,
inheriting from the base training classes and implementing cCRE-specific
model creation and data loading for unlabeled 512bp sequences.
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
from model_zoo.ccre.models import create_model
from model_zoo.ccre.data import get_ccre_datasets
from omegaconf import OmegaConf
from utils.utils import update_cfg_with_unknown_args


class cCRELightningModule(BaseD3LightningModule):
    """Lightning module specifically for cCRE dataset."""
    
    def __init__(self, cfg, architecture: str = 'transformer'):
        super().__init__(cfg, dataset_name='ccre')
        self.architecture = architecture
        
    def create_model(self):
        """Create cCRE-specific model."""
        return create_model(self.cfg, self.architecture)
        
    def process_batch(self, batch):
        """Process cCRE batch data.
        
        For cCRE, the data loader returns (sequence, sequence) pairs since
        there are no labels. We return inputs and None for unconditional generation.
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            # For unlabeled data, return inputs and None (no labels)
            return inputs, None
        else:
            raise ValueError(f"Expected (inputs, targets) pair, got {type(batch)}")


class cCREDataModule(BaseD3DataModule):
    """Data module specifically for cCRE dataset."""
    
    def __init__(self, cfg):
        super().__init__(cfg, dataset_name='ccre')
        
    def setup(self, stage: str = None):
        """Setup cCRE datasets."""
        # Get split configuration from config
        train_ratio = getattr(self.cfg.data, 'train_ratio', 0.95)
        valid_ratio = getattr(self.cfg.data, 'valid_ratio', 0.05)
        split_seed = getattr(self.cfg.data, 'split_seed', 42)
        
        # Use cCRE-specific data loading
        self.train_ds, self.val_ds = get_ccre_datasets(
            self.cfg.paths.data_file,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            seed=split_seed
        )
        print(f"cCRE dataset loaded: {len(self.train_ds)} train, {len(self.val_ds)} val samples")


class cCRETrainer(BaseTrainer):
    """Trainer specifically for cCRE dataset."""
    
    def __init__(self, architecture: str, config_path: str = None, work_dir: str = None,
                 more_cfg_args: list = None):
        # Load cCRE config
        if config_path:
            cfg = OmegaConf.load(config_path)
        else:
            # Use default cCRE config
            config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            cfg = OmegaConf.load(config_file)
            
        super().__init__(cfg, 'ccre', work_dir)
        self.architecture = architecture
        
    def create_lightning_module(self):
        """Create cCRE Lightning module."""
        return cCRELightningModule(self.cfg, self.architecture)
        
    def create_data_module(self):
        """Create cCRE data module."""
        return cCREDataModule(self.cfg)
    
    def setup_callbacks(self):
        """Setup training callbacks.
        
        Note: No SP-MSE callback for cCRE since there are no labels.
        """
        callbacks = super().setup_callbacks()
        
        # cCRE doesn't need SP-MSE callbacks since there are no labels
        print("Note: SP-MSE validation disabled for unlabeled cCRE dataset")
        
        return callbacks


def main():
    """Main training function."""
    parser = parse_base_args()
    parser.description = 'cCRE Training Script'
    args, unknown = parser.parse_known_args()

    # Set all seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create trainer (loads cfg)
    trainer = cCRETrainer(
        architecture=args.architecture,
        config_path=args.config,
        work_dir=args.work_dir,
        more_cfg_args=unknown,
    )

    # Override WandB settings if provided
    if args.wandb_project:
        trainer.cfg.wandb.project = args.wandb_project
    if args.wandb_name:
        trainer.cfg.wandb.name = args.wandb_name

    # override other unknown args (e.g. --paths.data_file)
    if unknown:
        update_cfg_with_unknown_args(trainer.cfg, unknown)
    
    # Train
    try:
        trainer.train(resume_from=args.resume_from)
        return 0
    except Exception as e:
        print(f"Training failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())