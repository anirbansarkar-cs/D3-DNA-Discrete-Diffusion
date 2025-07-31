#!/usr/bin/env python3
"""
Base Training Module for D3-DNA Discrete Diffusion

This module provides the base Lightning module and training functionality that can be
inherited by dataset-specific training scripts. It absorbs all functionality from
lightning_trainer.py and train_lightning.py to provide a single base class.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from itertools import chain
import datetime

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import numpy as np
from omegaconf import OmegaConf

torch.set_float32_matmul_precision('medium')

# Package imports

from model.ema import ExponentialMovingAverage
from utils import losses
from utils import graph_lib
from utils import noise_lib
from utils.utils import get_score_fn, load_hydra_config_from_run, get_logger


class BaseD3LightningModule(pl.LightningModule):
    """
    Base PyTorch Lightning module for D3 DNA Discrete Diffusion model.
    
    This class provides all the shared functionality for training D3 models across
    different datasets. Dataset-specific implementations should inherit from this
    class and provide their own model creation and data handling logic.
    """
    
    def __init__(self, cfg, dataset_name: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.dataset_name = dataset_name
        
        # This will be implemented by subclasses
        self.score_model = None
        self.graph = None  # Will be initialized in setup()
        self.noise = None  # Will be initialized in setup()
        
        # EMA setup - will be initialized after score_model is created
        self.ema = None
        
        # Loss function will be set up in setup()
        self.loss_fn = None
        self.sampling_eps = 1e-5
        
        # Accumulation setup
        self.accum_iter = 0
        self.total_loss = 0
        
    def create_model(self):
        """
        Create the score model. This must be implemented by subclasses.
        
        Returns:
            The score model instance
        """
        raise NotImplementedError("Subclasses must implement create_model()")
        
    def setup_ema(self):
        """Setup EMA after model is created."""
        if self.score_model is not None:
            self.ema = ExponentialMovingAverage(
                self.score_model.parameters(), 
                decay=self.cfg.training.ema
            )
    
    def setup(self, stage: str = None):
        """Setup method called after the model is moved to device."""
        # Create model if not already created
        if self.score_model is None:
            self.score_model = self.create_model()
            self.setup_ema()
            
        # Initialize graph and noise on the correct device
        self.graph = graph_lib.get_graph(self.cfg, self.device)
        self.noise = noise_lib.get_noise(self.cfg).to(self.device)
        
        # Move EMA shadow parameters to the correct device (important for distributed training)
        if hasattr(self, 'ema') and hasattr(self.ema, 'shadow_params'):
            for i, shadow_param in enumerate(self.ema.shadow_params):
                self.ema.shadow_params[i] = shadow_param.to(self.device)
        
        # Setup loss function
        self.loss_fn = losses.get_loss_fn(
            self.noise, self.graph, train=True, sampling_eps=self.sampling_eps
        )
        
    def process_batch(self, batch):
        """
        Process batch data into inputs and targets.
        Can be overridden by subclasses for dataset-specific processing.
        
        Args:
            batch: Raw batch data from dataloader
            
        Returns:
            Tuple of (inputs, targets)
        """
        # Default implementation for most datasets
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, target = batch
            return inputs, target
        else:
            # Handle single tensor case (e.g., promoter dataset)
            return batch, None
        
    def training_step(self, batch, batch_idx):
        """Training step with gradient accumulation support."""
        inputs, target = self.process_batch(batch)
            
        # Compute loss
        if target is not None:
            loss = self.loss_fn(self.score_model, inputs, target).mean()
        else:
            # For datasets without separate targets
            loss = self.loss_fn(self.score_model, inputs).mean()
            
        loss = loss / self.cfg.training.accum
        
        # Accumulation logic
        self.accum_iter += 1
        self.total_loss += loss.detach()
        
        if self.accum_iter == self.cfg.training.accum:
            self.accum_iter = 0
            # Update EMA
            if self.ema is not None:
                self.ema.update(self.score_model.parameters())
            
            # Log the accumulated loss (detached for logging)
            accumulated_loss_log = self.total_loss
            self.log('train_loss', accumulated_loss_log, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.total_loss = 0
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step using EMA weights."""
        inputs, target = self.process_batch(batch)
            
        # Use EMA weights for validation
        if self.ema is not None:
            self.ema.store(self.score_model.parameters())
            self.ema.copy_to(self.score_model.parameters())
        
        # Setup eval loss function
        eval_loss_fn = losses.get_loss_fn(
            self.noise, self.graph, train=False, sampling_eps=self.sampling_eps
        )
        
        with torch.no_grad():
            if target is not None:
                loss = eval_loss_fn(self.score_model, inputs, target).mean()
            else:
                loss = eval_loss_fn(self.score_model, inputs).mean()
            
        # Restore original weights
        if self.ema is not None:
            self.ema.restore(self.score_model.parameters())
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        params = chain(self.score_model.parameters(), self.noise.parameters())
        optimizer = losses.get_optimizer(self.cfg, params)
        
        # Setup warmup scheduler if specified
        if self.cfg.optim.warmup > 0:
            def lr_lambda(step):
                if step < self.cfg.optim.warmup:
                    return step / self.cfg.optim.warmup
                return 1.0
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        
        return optimizer
    
    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping before optimizer step."""
        if self.cfg.optim.grad_clip >= 0:
            self.clip_gradients(
                optimizer, 
                gradient_clip_val=self.cfg.optim.grad_clip, 
                gradient_clip_algorithm="norm"
            )
    

    def load_from_original_checkpoint(self, checkpoint_path: str):
        """
        Loads weights from checkpoint_path into self.score_model and self.ema,
        skipping only layers whose shapes don't match (prints info for skipped layers).
        Also updates the step counter if present.
        """
        loaded_state = torch.load(checkpoint_path, map_location=self.device)
        state_dict = loaded_state.get('state_dict', loaded_state)

        # --- Load model weights (partial) ---
        model_dict = self.score_model.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered_dict[k] = v
            else:
                if k in model_dict:
                    print(f"Skipping layer (by shape): {k} (checkpoint: {v.shape}, model: {model_dict[k].shape})")
                else:
                    print(f"Skipping layer (not in model): {k}")
        model_dict.update(filtered_dict)
        self.score_model.load_state_dict(model_dict, strict=False)
        print("✓ Loaded partial model weights (except skipped layers)")

        # --- Load EMA weights if present ---
        if 'ema' in loaded_state and self.ema is not None:
            self.ema.load_state_dict(loaded_state['ema'], device=self.device)
            print("✓ Loaded EMA weights from checkpoint")

        # --- Load step counter if present ---
        if 'step' in loaded_state:
            print(f"✓ Original checkpoint was at step: {loaded_state['step']}")
            
        return loaded_state.get('step', 0)

    
    def state_dict(self):
        """Override to include EMA state in Lightning checkpoints."""
        # Get the default Lightning state dict
        state = super().state_dict()
        
        # Add EMA state with proper prefixes
        if hasattr(self, 'ema') and self.ema is not None:
            ema_state = self.ema.state_dict()
            for key, value in ema_state.items():
                state[f'ema.{key}'] = value
        
        return state
    
    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Override to handle both Lightning and original checkpoint formats."""
        # Check if this is an original D3 checkpoint format
        if 'model' in state_dict and 'ema' in state_dict and 'step' in state_dict:
            # This is an original checkpoint, use our custom loader
            step = self.load_from_original_checkpoint_dict(state_dict)
            return step
        else:
            # This is a Lightning checkpoint
            # Separate EMA state from model state
            model_state = {}
            ema_state = {}
            
            for key, value in state_dict.items():
                if key.startswith('ema.'):
                    ema_key = key.replace('ema.', '')
                    ema_state[ema_key] = value
                else:
                    model_state[key] = value
            
            # Load model state using parent method
            result = super().load_state_dict(model_state, strict=False)
            
            # Load EMA state separately if it exists
            if ema_state and hasattr(self, 'ema') and self.ema is not None:
                try:
                    self.ema.load_state_dict(ema_state, device=self.device)
                    print("✓ Loaded EMA state from Lightning checkpoint")
                except Exception as e:
                    print(f"⚠ Could not load EMA state: {e}")
            
            return result


class BaseD3DataModule(pl.LightningDataModule):
    """
    Base PyTorch Lightning DataModule for D3 datasets.
    
    This provides the common interface for data loading. Dataset-specific
    implementations should inherit from this and implement their own
    dataset loading logic.
    """
    
    def __init__(self, cfg, dataset_name: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.train_ds = None
        self.val_ds = None
        
    def prepare_data(self):
        """Download or prepare data. Called only from main process."""
        pass
        
    def setup(self, stage: str = None):
        """Setup train and validation datasets. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement setup()")
    
    def train_dataloader(self):
        """Create training dataloader."""
        from torch.utils.data import DataLoader
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
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.eval.batch_size // (self.cfg.ngpus * self.cfg.training.accum),
            num_workers=2,
            pin_memory=True,
            shuffle=False,
        )


class BaseTrainer:
    """
    Base trainer class that provides common training functionality.
    
    Dataset-specific training scripts should inherit from this class and
    implement the abstract methods for their specific needs.
    """
    
    def __init__(self, cfg, dataset_name: str, work_dir: Optional[str] = None):
        self.cfg = cfg
        self.dataset_name = dataset_name
        if work_dir:
            self.work_dir = work_dir
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.work_dir = f"experiments/{dataset_name}/{timestamp}"
    
    def _extract_timestamp_from_checkpoint(self, checkpoint_path: str) -> Optional[str]:
        """Extract timestamp from checkpoint path if it follows expected format."""
        path_parts = Path(checkpoint_path).parts
        # Look for pattern: experiments/{dataset}/{timestamp}/checkpoints/...
        try:
            exp_idx = path_parts.index('experiments')
            if (exp_idx + 3 < len(path_parts) and 
                path_parts[exp_idx + 1] == self.dataset_name and
                path_parts[exp_idx + 3] == 'checkpoints'):
                return path_parts[exp_idx + 2]  # This should be the timestamp
        except (ValueError, IndexError):
            pass
        return None
    
    def _update_work_dir_for_resume(self, resume_from: str):
        """Update work_dir to reuse existing directory if resuming from a checkpoint."""
        if not resume_from or not os.path.exists(resume_from):
            return
            
        timestamp = self._extract_timestamp_from_checkpoint(resume_from)
        if timestamp:
            # Reuse the existing timestamp directory
            self.work_dir = f"experiments/{self.dataset_name}/{timestamp}"
            print(f"Reusing existing work directory: {self.work_dir}")
        else:
            print(f"Could not extract timestamp from checkpoint path, using new directory: {self.work_dir}")
    
    def create_lightning_module(self) -> BaseD3LightningModule:
        """Create the Lightning module. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement create_lightning_module()")
        
    def create_data_module(self) -> BaseD3DataModule:
        """Create the data module. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement create_data_module()")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = os.path.join(self.work_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        loggers = []
        
        # TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=self.work_dir,
            name="lightning_logs",
            version=None
        )
        loggers.append(tb_logger)
        
        # WandB logger if configured
        if hasattr(self.cfg, 'wandb') and self.cfg.wandb.get('enabled', False):
            # config_dict = OmegaConf.to_yaml(self.cfg)
            wandb_logger = WandbLogger(
                project=self.cfg.wandb.get('project', 'd3-dna-diffusion'),
                name=self.cfg.wandb.get('name', f"{self.dataset_name}_{self.cfg.model.architecture}"),
                entity=self.cfg.wandb.get('entity', None),
                # config=config_dict,
                save_dir=self.work_dir, 
                id=self.cfg.wandb.get('id', None),
            )
            loggers.append(wandb_logger)
        
        return loggers
    
    def setup_callbacks(self):
        """Setup training callbacks."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.work_dir, "checkpoints"),
            filename="model-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
            every_n_epochs=self.cfg.training.get('checkpoint_every_n_epochs', 10)  # Save every 10 epochs by default
        )
        callbacks.append(checkpoint_callback)
        
        # Learning rate monitor
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        
        # Early stopping if configured
        if hasattr(self.cfg.training, 'early_stopping_patience') and self.cfg.training.early_stopping_patience:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=self.cfg.training.early_stopping_patience,
                mode='min'
            )
            callbacks.append(early_stop)
            
# SP-MSE callbacks are now handled by dataset-specific trainers
        
        return callbacks
    
    def create_trainer(self, **trainer_kwargs):
        """Create PyTorch Lightning trainer."""
        
        # Extract relevant training parameters - using epochs instead of steps
        default_trainer_args = {
            'max_epochs': self.cfg.training.get('max_epochs', 300),  # Default to 300 epochs
            'log_every_n_steps': self.cfg.training.get('log_freq', 50),
            'check_val_every_n_epoch': self.cfg.training.get('val_every_n_epochs', 4),  # Validate every 4 epochs
            'accumulate_grad_batches': self.cfg.training.accum,
            'precision': 'bf16-mixed',  # Use mixed precision like original
            'gradient_clip_val': self.cfg.optim.grad_clip if self.cfg.optim.grad_clip >= 0 else None,
            'enable_checkpointing': True,
            'enable_progress_bar': True,
            'enable_model_summary': True,
            'callbacks': self.setup_callbacks(),
            'logger': self.setup_logging(),
        }
        
        # Handle multi-GPU setup
        if self.cfg.ngpus > 1:
            default_trainer_args.update({
                'devices': self.cfg.ngpus,
                'num_nodes': getattr(self.cfg, 'nnodes', 1),  # Default to 1 node if not specified
                'strategy': 'ddp_find_unused_parameters_true',  # More robust for cluster environments
                'sync_batchnorm': True,
            })
        else:
            default_trainer_args['devices'] = 1
        
        # Override with any custom trainer arguments
        default_trainer_args.update(trainer_kwargs)
        
        # Create trainer
        trainer = pl.Trainer(**default_trainer_args)
        
        return trainer
    
    def train(self, resume_from: Optional[str] = None):
        """Main training method."""
        # Update work directory if resuming from checkpoint
        if resume_from:
            self._update_work_dir_for_resume(resume_from)
            
        # Create work directory
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Create Lightning components
        lightning_module = self.create_lightning_module()
        data_module = self.create_data_module()
        trainer = self.create_trainer()
        
        # Handle checkpoint resuming
        ckpt_path = None
        if resume_from:
            if os.path.exists(resume_from):
                ckpt_path = resume_from
                print(f"Resuming from checkpoint: {resume_from}")
            else:
                print(f"Warning: Checkpoint not found: {resume_from}")
        
        # Train
        trainer.fit(lightning_module, data_module, ckpt_path=ckpt_path)
        
        print(f"Training completed. Results saved to: {self.work_dir}")
        return trainer, lightning_module

def parse_base_args():
    """Parse common command line arguments for training scripts."""
    parser = argparse.ArgumentParser(description='D3 Training Script')
    parser.add_argument('--architecture', required=True, help='Architecture (transformer or convolutional)')
    parser.add_argument('--config', help='Override config file (optional)')
    parser.add_argument('--work_dir', help='Working directory for outputs')
    parser.add_argument('--resume_from', help='Checkpoint to resume from')
    parser.add_argument('--wandb_project', help='Weights & Biases project name')
    parser.add_argument('--wandb_name', help='Weights & Biases run name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser