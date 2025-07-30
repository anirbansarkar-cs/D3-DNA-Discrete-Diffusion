import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import Optional, Any, Tuple
from abc import ABC, abstractmethod
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer

from scripts import sampling


class BaseSPMSEValidationCallback(Callback, ABC):
    """
    Base callback for evaluating generated sequences using oracle models and computing SP-MSE.
    
    This base class provides common functionality for SP-MSE validation during training.
    Dataset-specific implementations should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(
        self,
        oracle_path: str,
        data_path: str,
        validation_freq_epochs: int = 4,
        validation_samples: int = 1000,
        enabled: bool = True,
        sampling_steps: Optional[int] = None,
        early_stopping_patience: Optional[int] = None
    ):
        """
        Args:
            oracle_path: Path to oracle model checkpoint
            data_path: Path to dataset file
            validation_freq_epochs: Frequency in epochs to run SP-MSE validation
            validation_samples: Number of samples to use from validation set (subsampling)
            enabled: Whether SP-MSE validation is enabled
            sampling_steps: Number of sampling steps (defaults to sequence length)
            early_stopping_patience: Stop training if no improvement for N validations
        """
        super().__init__()
        self.oracle_path = oracle_path
        self.data_path = data_path
        self.validation_freq_epochs = validation_freq_epochs
        self.validation_samples = validation_samples
        self.enabled = enabled
        self.sampling_steps = sampling_steps
        self.early_stopping_patience = early_stopping_patience
        
        # State tracking
        self.oracle_model = None
        self.best_sp_mse = float('inf')
        self.steps_since_improvement = 0
        self.validation_data_cache = None
        self.best_checkpoint_path = None  # Track current best checkpoint for replacement
        
        if not self.enabled:
            return
        
        # Set default sampling steps if not provided
        if self.sampling_steps is None:
            self.sampling_steps = self.get_default_sampling_steps()
    
    @abstractmethod
    def get_default_sampling_steps(self) -> int:
        """Get default sampling steps for this dataset."""
        pass
    
    @abstractmethod
    def load_oracle_model(self):
        """Load the oracle model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_oracle_predictions(self, sequences: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Get oracle predictions for sequences.
        
        Args:
            sequences: Input sequences (format depends on implementation)
            device: Device to run on
            
        Returns:
            Oracle predictions
        """
        pass
    
    @abstractmethod
    def process_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process batch data into sequences and targets.
        
        Args:
            batch: Raw batch data
            
        Returns:
            Tuple of (sequences, targets)
        """
        pass
    
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup callback - load oracle model"""
        if not self.enabled:
            return
            
        if stage == 'fit':
            self.oracle_model = self.load_oracle_model()
            if self.oracle_model is not None:
                # Move oracle model to the same device as the main model  
                self.oracle_model = self.oracle_model.to(pl_module.device)
                self.oracle_model.eval()
                print(f"SP-MSE Callback: Loaded oracle model from {self.oracle_path} on device {pl_module.device}")
            else:
                print(f"SP-MSE Callback: Failed to load oracle model, disabling callback")
                self.enabled = False
    
    def on_train_epoch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule
    ) -> None:
        """Check if it's time to run SP-MSE validation"""
        if not self.enabled or self.oracle_model is None:
            return
            
        current_epoch = trainer.current_epoch
        
        # Check if it's time for SP-MSE validation
        if current_epoch > 0 and current_epoch % self.validation_freq_epochs == 0:
            self._run_sp_mse_validation(trainer, pl_module)
    
    def _get_validation_data(self, trainer: Trainer):
        """Get validation data with optional subsampling"""
        if self.validation_data_cache is not None:
            return self.validation_data_cache
            
        # Get validation dataloader
        val_dataloader = trainer.datamodule.val_dataloader()
        
        # Collect all validation data
        all_sequences = []
        all_targets = []
        
        for batch in val_dataloader:
            sequences, targets = self.process_batch(batch)
            all_sequences.append(sequences)
            all_targets.append(targets)
        
        all_sequences = torch.cat(all_sequences, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Subsample if requested
        if self.validation_samples > 0 and len(all_sequences) > self.validation_samples:
            indices = torch.randperm(len(all_sequences))[:self.validation_samples]
            all_sequences = all_sequences[indices]
            all_targets = all_targets[indices]
        
        self.validation_data_cache = (all_sequences, all_targets)
        return self.validation_data_cache
    
    def _run_sp_mse_validation(self, trainer: Trainer, pl_module: LightningModule):
        """Run SP-MSE validation"""
        device = pl_module.device
        
        # Ensure oracle model is on the correct device
        if self.oracle_model is not None:
            self.oracle_model = self.oracle_model.to(device)
        
        # Get validation data
        val_sequences, val_targets = self._get_validation_data(trainer)
        val_targets = val_targets.to(device)
        
        # Get sequence length
        seq_length = self.get_default_sampling_steps()
        
        # Initialize sampling function
        sampling_fn = sampling.get_pc_sampler(
            pl_module.graph, pl_module.noise, 
            (len(val_sequences), seq_length), 
            'analytic', self.sampling_steps, device=device
        )
        
        # Generate sequences using the diffusion model
        with torch.no_grad():
            # Set model to eval mode temporarily
            was_training = pl_module.training
            pl_module.eval()
            
            # Use EMA parameters if available
            if hasattr(pl_module, 'ema') and pl_module.ema is not None:
                pl_module.ema.store(pl_module.score_model.parameters())
                pl_module.ema.copy_to(pl_module.score_model.parameters())
            
            try:
                # Generate samples
                generated_sequences = sampling_fn(pl_module.score_model, val_targets)
                
                # Get oracle predictions
                val_score = self.get_oracle_predictions(val_sequences, device)
                generated_score = self.get_oracle_predictions(generated_sequences, device)
                
                # Calculate SP-MSE
                sp_mse = (val_score - generated_score) ** 2
                mean_sp_mse = torch.mean(sp_mse).cpu().item()
                
                # Log metrics to all loggers (including WandB)
                if trainer.logger:
                    # Handle multiple loggers (TensorBoard and WandB)
                    if hasattr(trainer.logger, 'experiment'):
                        # Single logger
                        trainer.logger.log_metrics({
                            'sp_mse/validation': mean_sp_mse,
                            'sp_mse/best': self.best_sp_mse
                        }, step=trainer.global_step)
                    else:
                        # Multiple loggers (LoggerCollection)
                        for logger in trainer.logger:
                            logger.log_metrics({
                                'sp_mse/validation': mean_sp_mse,
                                'sp_mse/best': self.best_sp_mse
                            }, step=trainer.global_step)
                
                print(f"Step {trainer.global_step}: SP-MSE = {mean_sp_mse:.6f}, Best = {self.best_sp_mse:.6f}")
                
                # Check if this is the best SP-MSE
                if mean_sp_mse < self.best_sp_mse:
                    self.best_sp_mse = mean_sp_mse
                    self.steps_since_improvement = 0
                    
                    # Remove previous best checkpoint if it exists
                    if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
                        os.remove(self.best_checkpoint_path)
                        # print(f"Removed previous best SP-MSE checkpoint: {self.best_checkpoint_path}")
                    
                    # Save new best checkpoint in checkpoints directory (matching normal checkpoint location)
                    # Use logger's save_dir to match the work_dir used in normal checkpoint saving
                    if hasattr(trainer.logger, 'save_dir') and trainer.logger.save_dir:
                        work_dir = trainer.logger.save_dir
                    else:
                        work_dir = trainer.default_root_dir
                    checkpoints_dir = os.path.join(work_dir, "checkpoints")
                    os.makedirs(checkpoints_dir, exist_ok=True)
                    
                    checkpoint_filename = f"sp-mse_{mean_sp_mse:.6f}_step_{trainer.global_step}.ckpt"
                    self.best_checkpoint_path = os.path.join(checkpoints_dir, checkpoint_filename)
                    
                    trainer.save_checkpoint(self.best_checkpoint_path)
                    # print(f"Saved best SP-MSE checkpoint: {self.best_checkpoint_path}")
                    
                else:
                    self.steps_since_improvement += 1
                    
                    # Check early stopping
                    if (self.early_stopping_patience is not None and 
                        self.steps_since_improvement >= self.early_stopping_patience):
                        print(f"Early stopping: No SP-MSE improvement for {self.early_stopping_patience} validations")
                        trainer.should_stop = True
                
            finally:
                # Restore EMA parameters if used
                if hasattr(pl_module, 'ema') and pl_module.ema is not None:
                    pl_module.ema.restore(pl_module.score_model.parameters())
                
                # Restore training mode
                if was_training:
                    pl_module.train()