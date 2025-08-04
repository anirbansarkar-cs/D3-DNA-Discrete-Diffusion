"""
LentIMPRA-specific SP-MSE Validation Callback

This module implements the SP-MSE validation callback specifically for the LentIMPRA dataset,
inheriting from the base callback and providing LentIMPRA-specific oracle loading and prediction.
"""

import torch
import torch.nn.functional as F
from typing import Tuple
from utils.sp_mse_callback import BaseSPMSEValidationCallback


class LentIMPRASPMSECallback(BaseSPMSEValidationCallback):
    """SP-MSE validation callback specifically for LentIMPRA dataset."""
    
    def get_default_sampling_steps(self) -> int:
        """Get default sampling steps for LentIMPRA (sequence length)."""
        return 230
    
    def load_oracle_model(self):
        """Load LentIMPRA oracle model using existing LegNet infrastructure."""
        try:
            # Import existing LegNet components from mpralegnet
            from model_zoo.lentimpra.mpralegnet import load_model
            import os
            
            # Check if config file exists alongside checkpoint
            oracle_dir = os.path.dirname(self.oracle_path)
            config_path = os.path.join(oracle_dir, 'config.json')
            
            if os.path.exists(config_path):
                # Load using existing load_model function
                oracle, config = load_model(self.oracle_path, config_path)
                oracle.eval()
                return oracle
            else:
                # Fallback: create default config and load checkpoint
                print(f"Config file not found at {config_path}, using default config")
                from model_zoo.lentimpra.mpralegnet import get_default_config, LitModel
                
                config = get_default_config()
                oracle = LitModel(config)
                
                # Load checkpoint weights
                checkpoint = torch.load(self.oracle_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    oracle.load_state_dict(checkpoint['state_dict'])
                else:
                    oracle.load_state_dict(checkpoint)
                
                oracle.eval()
                return oracle
                
        except Exception as e:
            print(f"Failed to load LentIMPRA oracle model: {e}")
            return None
    
    def get_oracle_predictions(self, sequences: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Get LentIMPRA oracle predictions for sequences.
        
        Args:
            sequences: Input sequences (can be indices or one-hot)
            device: Device to run on
            
        Returns:
            Oracle predictions tensor
        """
        if self.oracle_model is None:
            raise RuntimeError("Oracle model not loaded")
        
        # Ensure oracle model is on the correct device
        self.oracle_model = self.oracle_model.to(device)
        
        # Convert to one-hot if needed
        if sequences.dtype == torch.long:
            sequences_one_hot = F.one_hot(sequences, num_classes=4).float()
        else:
            sequences_one_hot = sequences
        
        # LentIMPRA expects input as one-hot (batch_size, length, channels) for the oracle
        # The oracle model (LegNet) expects input as (batch_size, channels, length)
        if sequences_one_hot.shape[-1] == 4:  # (batch_size, length, 4)
            sequences_input = sequences_one_hot.permute(0, 2, 1).to(device)  # -> (batch_size, 4, length)
        else:  # Already (batch_size, 4, length)
            sequences_input = sequences_one_hot.to(device)
        
        # Get oracle predictions using the LegNet model's predict method
        with torch.no_grad():
            # Use the predict method from LitModel
            predictions = self.oracle_model.predict(sequences_input)
        
        return predictions
    
    def process_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process LentIMPRA batch data.
        
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


def create_lentimpra_sp_mse_callback(cfg, dataset_name: str = 'lentimpra'):
    """
    Create LentIMPRA SP-MSE callback from configuration.
    
    Args:
        cfg: Configuration object
        dataset_name: Dataset name (default: 'lentimpra')
        
    Returns:
        LentIMPRASPMSECallback instance or None if not enabled
    """
    if not hasattr(cfg, 'sp_mse_validation') or not cfg.sp_mse_validation.get('enabled', False):
        return None
    
    sp_mse_cfg = cfg.sp_mse_validation
    
    # Auto-resolve paths if not provided
    oracle_path = sp_mse_cfg.get('oracle_path')
    if oracle_path is None:
        if hasattr(cfg, 'paths') and hasattr(cfg.paths, 'oracle_model'):
            oracle_path = cfg.paths.oracle_model
        else:
            oracle_path = 'model_zoo/lentimpra/oracle_models/best_model-epoch=24-val_pearson=0.814.ckpt'
    
    data_path = sp_mse_cfg.get('data_path')
    if data_path is None:
        if hasattr(cfg, 'paths') and hasattr(cfg.paths, 'data_file'):
            data_path = cfg.paths.data_file
        else:
            data_path = 'model_zoo/lentimpra/lenti_MPRA_K562_data.h5'
    
    callback = LentIMPRASPMSECallback(
        oracle_path=oracle_path,
        data_path=data_path,
        validation_freq_epochs=sp_mse_cfg.get('validation_freq_epochs', 4),
        validation_samples=sp_mse_cfg.get('validation_samples', 1000),
        enabled=sp_mse_cfg.get('enabled', False),
        sampling_steps=sp_mse_cfg.get('sampling_steps', 230),
        early_stopping_patience=sp_mse_cfg.get('early_stopping_patience')
    )
    
    print(f"✓ SP-MSE callback configured to use LegNet oracle: {oracle_path}")
    
    return callback