#!/usr/bin/env python3
"""
DeepSTARR Training Script

This script provides training functionality specifically for the DeepSTARR dataset,
inheriting from the base training classes and implementing DeepSTARR-specific
model creation and data loading. Now supports both standard training and EvoAug training.
Also supports transfer learning from ATAC-seq models.
"""

import os
import sys
from pathlib import Path
import numpy as np
import random
import torch
import datetime
from typing import Optional


# Package imports

from scripts.train import BaseD3LightningModule, BaseD3DataModule, BaseTrainer, parse_base_args
from model_zoo.deepstarr.models import create_model
from model_zoo.deepstarr.data import get_deepstarr_datasets, get_deepstarr_evoaug_dataloaders, get_deepstarr_dataloaders
from model_zoo.deepstarr.sp_mse_callback import create_deepstarr_sp_mse_callback
from omegaconf import OmegaConf
from utils.utils import update_cfg_with_unknown_args
from utils import graph_lib, noise_lib

class DeepSTARRLightningModule(BaseD3LightningModule):
    """Lightning module specifically for DeepSTARR dataset."""
    
    def __init__(self, cfg, architecture: str = 'transformer', transfer_from: Optional[str] = None):
        super().__init__(cfg, dataset_name='deepstarr')
        self.architecture = architecture
        self.transfer_from = transfer_from
        
    def create_model(self):
        """Create DeepSTARR-specific model."""
        return create_model(self.cfg, self.architecture)
    
    # NOTE: might deprecate and move all transfer learning logic to BaseD3LightningModule
    def setup(self, stage: str = None):
        """Setup method with transfer learning support - overrides setup in BaseD3LightningModule"""
        # Create model if not already created
        if self.score_model is None:
            self.score_model = self.create_model()
            
            # Handle transfer learning if specified
            if self.transfer_from and os.path.exists(self.transfer_from):
                print(f"Loading transfer learning checkpoint: {self.transfer_from}")
                self._load_transfer_checkpoint(self.transfer_from)
            
            # Setup EMA after model creation (and after transfer learning)
            self.setup_ema()
            
            # Verify EMA setup
            if self.ema is None:
                print("Warning: EMA setup failed, attempting to reinitialize...")
                self.setup_ema()
            
            if self.ema is not None:
                print(f"EMA successfully initialized with decay: {self.cfg.training.ema}")
            else:
                print("Error: EMA initialization failed")
        
        # Call parent setup to handle graph, noise, and loss function initialization
        super().setup(stage)
    
    def _load_transfer_checkpoint(self, checkpoint_path: str):
        """Load checkpoint from ATAC-seq model for transfer learning."""
        try:
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract model state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"Checkpoint contains {len(state_dict)} parameters")
                
                # Remove module prefix if present
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('score_model.'):
                        new_key = key.replace('score_model.', '')
                        new_state_dict[new_key] = value
                    elif key.startswith('model.'):
                        new_key = key.replace('model.', '')
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                
                # Load compatible parameters (ignore size mismatches)
                model_dict = self.score_model.state_dict()
                compatible_dict = {}
                skipped_mismatch = 0
                skipped_not_found = 0
                
                for key, value in new_state_dict.items():
                    if key in model_dict:
                        if value.shape == model_dict[key].shape:
                            compatible_dict[key] = value
                        else:
                            print(f"Skipping {key}: shape mismatch {value.shape} -> {model_dict[key].shape}")
                            skipped_mismatch += 1
                    else:
                        print(f"Skipping {key}: not found in target model")
                        skipped_not_found += 1
                
                # Load compatible parameters
                if compatible_dict:
                    missing_keys, unexpected_keys = self.score_model.load_state_dict(compatible_dict, strict=False)
                    print(f"Transfer learning: loaded {len(compatible_dict)} parameters")
                    print(f"Skipped due to shape mismatch: {skipped_mismatch}")
                    print(f"Skipped due to not found: {skipped_not_found}")
                    if missing_keys:
                        print(f"Missing keys: {len(missing_keys)}")
                    if unexpected_keys:
                        print(f"Unexpected keys: {len(unexpected_keys)}")
                else:
                    print("Warning: No compatible parameters found for transfer learning")
                    
        except Exception as e:
            print(f"Warning: Failed to load transfer checkpoint: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing with random initialization")
        
    def process_batch(self, batch):
        """Process DeepSTARR batch data."""
        # DeepSTARR data comes as (inputs, targets) pairs
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            
            # Handle EvoAug one-hot encoded data: convert (batch_size, 4, seq_length) to (batch_size, seq_length)
            if len(inputs.shape) == 3:
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
    
    def __init__(self, architecture: str, config_path: str = None, work_dir: str = None, use_evoaug: bool = False, transfer_from: Optional[str] = None):
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
        self.transfer_from = transfer_from
        
        # Update work directory to indicate EvoAug usage
        if self.use_evoaug:
            self.work_dir = self.work_dir.replace('deepstarr', 'deepstarr_evoaug')
        
        # Update work directory to indicate transfer learning
        if self.transfer_from:
            self.work_dir = self.work_dir.replace('deepstarr', 'deepstarr_transfer')
        
    def create_lightning_module(self):
        """Create DeepSTARR Lightning module."""
        return DeepSTARRLightningModule(self.cfg, self.architecture, self.transfer_from)
        
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
    parser.add_argument('--transfer_from', type=str, default=None,
                       help='Path to ATAC-seq checkpoint for transfer learning')
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
        transfer_from=args.transfer_from,
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
    if args.transfer_from:
        print("=" * 60)
        print("TRANSFER LEARNING MODE")
        print("=" * 60)
        print(f"Loading pretrained model from: {args.transfer_from}")
        print("Adapting to DeepSTARR dataset")
        print("=" * 60)
    elif args.use_evoaug:
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