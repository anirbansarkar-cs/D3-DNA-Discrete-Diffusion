#!/usr/bin/env python3
"""
LentIMPRA Training Script

This script provides training functionality specifically for the LentIMPRA dataset,
inheriting from the base training classes and implementing LentIMPRA-specific
model creation and data loading.
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
from model_zoo.lentimpra.models import create_model
from model_zoo.lentimpra.data import get_lentimpra_datasets
from model_zoo.lentimpra.sp_mse_callback import create_lentimpra_sp_mse_callback
from omegaconf import OmegaConf
from utils.utils import update_cfg_with_unknown_args

class LentIMPRALightningModule(BaseD3LightningModule):
    """Lightning module specifically for LentIMPRA dataset."""
    
    def __init__(self, cfg, architecture: str = 'transformer'):
        super().__init__(cfg, dataset_name='lentimpra')
        self.architecture = architecture
        
    def create_model(self):
        """Create LentIMPRA-specific model."""
        return create_model(self.cfg, self.architecture)
        
    def process_batch(self, batch):
        """Process LentIMPRA batch data."""
        # LentIMPRA data comes as (inputs, targets) pairs
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            return inputs, targets
        else:
            raise ValueError(f"Expected (inputs, targets) pair, got {type(batch)}")


class LentIMPRADataModule(BaseD3DataModule):
    """Data module specifically for LentIMPRA dataset."""
    
    def __init__(self, cfg):
        super().__init__(cfg, dataset_name='lentimpra')
        
    def setup(self, stage: str = None):
        """Setup LentIMPRA datasets."""
        # Use LentIMPRA-specific data loading
        self.train_ds, self.val_ds, _ = get_lentimpra_datasets(self.cfg.paths.data_file)
        print(f"LentIMPRA dataset loaded: {len(self.train_ds)} train, {len(self.val_ds)} val samples")


class LentIMPRATrainer(BaseTrainer):
    """Trainer specifically for LentIMPRA dataset."""
    
    def __init__(self, architecture: str, config_path: str = None, work_dir: str = None,
                 more_cfg_args: list = None):
        # Load LentIMPRA config
        if config_path:
            cfg = OmegaConf.load(config_path)
        else:
            # Use default LentIMPRA config
            config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            cfg = OmegaConf.load(config_file)
            
        super().__init__(cfg, 'lentimpra', work_dir)
        self.architecture = architecture
        
    def create_lightning_module(self):
        """Create LentIMPRA Lightning module."""
        return LentIMPRALightningModule(self.cfg, self.architecture)
        
    def create_data_module(self):
        """Create LentIMPRA data module."""
        return LentIMPRADataModule(self.cfg)
    
    def setup_callbacks(self):
        """Setup training callbacks including dataset-specific SP-MSE callback."""
        callbacks = super().setup_callbacks()
        
        # Add LentIMPRA-specific SP-MSE callback if enabled
        sp_mse_callback = create_lentimpra_sp_mse_callback(self.cfg)
        if sp_mse_callback is not None:
            callbacks.append(sp_mse_callback)
        
        return callbacks


def main():
    """Main training function."""
    parser = parse_base_args()
    parser.description = 'LentIMPRA Training Script'
    args, unknown = parser.parse_known_args()

    # Set all seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create trainer (loads cfg)
    trainer = LentIMPRATrainer(
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