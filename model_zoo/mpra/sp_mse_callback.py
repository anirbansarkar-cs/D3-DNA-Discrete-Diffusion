"""
MPRA-specific SP-MSE Validation Callback

This module implements the SP-MSE validation callback specifically for the MPRA dataset,
inheriting from the base callback and providing MPRA-specific oracle loading and prediction.
"""

import torch
import torch.nn.functional as F
from typing import Tuple
from utils.sp_mse_callback import BaseSPMSEValidationCallback
from model_zoo.mpra.mpra import PL_mpra


class MPRASPMSECallback(BaseSPMSEValidationCallback):
    """SP-MSE validation callback specifically for MPRA dataset."""
    
    def get_default_sampling_steps(self) -> int:
        """Get default sampling steps for MPRA (sequence length)."""
        return 200
    
    def load_oracle_model(self):
        """Load MPRA oracle model."""
        try:
            oracle = PL_mpra.load_from_checkpoint(
                self.oracle_path,
                input_h5_file=self.data_path
            ).eval()
            return oracle
        except Exception as e:
            print(f"Failed to load MPRA oracle model: {e}")
            return None
    
    def get_oracle_predictions(self, sequences: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Get MPRA oracle predictions for sequences.
        
        Args:
            sequences: Input sequences (can be indices or one-hot)
            device: Device to run on
            
        Returns:
            Oracle predictions tensor
        """
        if self.oracle_model is None:
            raise RuntimeError("Oracle model not loaded")
        
        # Convert to one-hot if needed
        if sequences.dtype == torch.long:
            sequences_one_hot = F.one_hot(sequences, num_classes=4).float()
        else:
            sequences_one_hot = sequences
        
        # MPRA expects input as (batch_size, channels, length)
        # Convert from (batch_size, length, channels) to (batch_size, channels, length)
        sequences_input = sequences_one_hot.permute(0, 2, 1).to(device)
        
        # Get oracle predictions
        with torch.no_grad():
            predictions = self.oracle_model.predict_custom(sequences_input)
        
        return predictions
    
    def process_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process MPRA batch data.
        
        Args:
            batch: Raw batch data (sequences, targets)
            
        Returns:
            Tuple of (sequences, targets)
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            sequences, targets = batch
            return sequences, targets
        else:
            raise ValueError(f"Expected (sequences, targets) pair, got {type(batch)}")


def create_mpra_sp_mse_callback(cfg, dataset_name: str = 'mpra'):
    """
    Create MPRA SP-MSE callback from configuration.
    
    Args:
        cfg: Configuration object
        dataset_name: Dataset name (default: 'mpra')
        
    Returns:
        MPRASPMSECallback instance or None if not enabled
    """
    if not hasattr(cfg, 'sp_mse_validation') or not cfg.sp_mse_validation.get('enabled', False):
        return None
    
    sp_mse_cfg = cfg.sp_mse_validation
    
    # Auto-resolve paths if not provided
    oracle_path = sp_mse_cfg.get('oracle_path')
    if oracle_path is None:
        oracle_path = 'model_zoo/mpra/oracle_models/oracle_mpra_mpra_data.ckpt'
    
    data_path = sp_mse_cfg.get('data_path')
    if data_path is None:
        data_path = 'model_zoo/mpra/mpra_data.h5'
    
    return MPRASPMSECallback(
        oracle_path=oracle_path,
        data_path=data_path,
        validation_freq_epochs=sp_mse_cfg.get('validation_freq_epochs', 4),
        validation_samples=sp_mse_cfg.get('validation_samples', 1000),
        enabled=sp_mse_cfg.get('enabled', False),
        sampling_steps=sp_mse_cfg.get('sampling_steps'),
        early_stopping_patience=sp_mse_cfg.get('early_stopping_patience')
    )