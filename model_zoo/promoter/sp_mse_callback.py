"""
Promoter-specific SP-MSE Validation Callback

This module implements the SP-MSE validation callback specifically for the Promoter dataset,
inheriting from the base callback and providing Promoter-specific oracle loading and prediction.
"""

import torch
import torch.nn.functional as F
from typing import Tuple
from utils.sp_mse_callback import BaseSPMSEValidationCallback
from model_zoo.promoter.sei import Sei, NonStrandSpecific


class PromoterSPMSECallback(BaseSPMSEValidationCallback):
    """SP-MSE validation callback specifically for Promoter dataset."""
    
    def get_default_sampling_steps(self) -> int:
        """Get default sampling steps for Promoter (sequence length)."""
        return 1024
    
    def load_oracle_model(self):
        """Load Promoter SEI oracle model."""
        try:
            # Load SEI model
            sei_model = Sei()
            oracle = NonStrandSpecific(sei_model)
            
            # Load checkpoint if provided
            if self.oracle_path and self.oracle_path != 'null':
                checkpoint = torch.load(self.oracle_path, map_location='cpu')
                oracle.load_state_dict(checkpoint, strict=False)
            
            return oracle.eval()
        except Exception as e:
            print(f"Failed to load Promoter SEI oracle model: {e}")
            return None
    
    def get_oracle_predictions(self, sequences: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Get Promoter SEI oracle predictions for sequences.
        
        Args:
            sequences: Input sequences (can be indices, one-hot, or promoter format)
            device: Device to run on
            
        Returns:
            Oracle predictions tensor
        """
        if self.oracle_model is None:
            raise RuntimeError("Oracle model not loaded")
        
        # Handle different input formats
        if sequences.dim() == 3 and sequences.shape[-1] == 5:
            # Promoter format: (batch_size, seq_length, 5) where last dim is [A, C, G, T, target]
            sequences_one_hot = sequences[:, :, :4]  # Extract sequence part
        elif sequences.dtype == torch.long:
            # Token indices: convert to one-hot
            sequences_one_hot = F.one_hot(sequences, num_classes=4).float()
        else:
            # Already one-hot
            sequences_one_hot = sequences
        
        # SEI expects input as (batch_size, channels, length)
        # Convert from (batch_size, length, channels) to (batch_size, channels, length)
        sequences_input = sequences_one_hot.permute(0, 2, 1).to(device)
        
        # Get oracle predictions
        with torch.no_grad():
            predictions = self.oracle_model(sequences_input)
            # SEI outputs many targets, take mean for simplicity
            predictions = torch.mean(predictions, dim=1, keepdim=True)
        
        return predictions
    
    def process_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process Promoter batch data.
        
        Args:
            batch: Raw batch data (promoter format or sequences/targets)
            
        Returns:
            Tuple of (sequences, targets)
        """
        if batch.dim() == 3 and batch.shape[-1] == 5:
            # Promoter format: (batch_size, seq_length, 5)
            sequences = batch  # Keep full format for oracle
            targets = batch[:, :, 4:5]  # Extract target part
            return sequences, targets
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            sequences, targets = batch
            return sequences, targets
        else:
            raise ValueError(f"Unexpected batch format: {batch.shape if hasattr(batch, 'shape') else type(batch)}")


def create_promoter_sp_mse_callback(cfg, dataset_name: str = 'promoter'):
    """
    Create Promoter SP-MSE callback from configuration.
    
    Args:
        cfg: Configuration object
        dataset_name: Dataset name (default: 'promoter')
        
    Returns:
        PromoterSPMSECallback instance or None if not enabled
    """
    if not hasattr(cfg, 'sp_mse_validation') or not cfg.sp_mse_validation.get('enabled', False):
        return None
    
    sp_mse_cfg = cfg.sp_mse_validation
    
    # Auto-resolve paths if not provided
    oracle_path = sp_mse_cfg.get('oracle_path')
    if oracle_path is None:
        oracle_path = 'model_zoo/promoter/oracle_models/best.sei.model.pth.tar'
    
    data_path = sp_mse_cfg.get('data_path')
    if data_path is None:
        data_path = None  # Promoter doesn't need a specific data file for SEI
    
    return PromoterSPMSECallback(
        oracle_path=oracle_path,
        data_path=data_path,
        validation_freq_epochs=sp_mse_cfg.get('validation_freq_epochs', 4),
        validation_samples=sp_mse_cfg.get('validation_samples', 1000),
        enabled=sp_mse_cfg.get('enabled', False),
        sampling_steps=sp_mse_cfg.get('sampling_steps'),
        early_stopping_patience=sp_mse_cfg.get('early_stopping_patience')
    )