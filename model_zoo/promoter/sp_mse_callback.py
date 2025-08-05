"""
Promoter-specific SP-MSE Validation Callback

This module implements the SP-MSE validation callback specifically for the Promoter dataset,
inheriting from the base callback and providing Promoter-specific oracle loading and prediction.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import re
from typing import Tuple
from utils.sp_mse_callback import BaseSPMSEValidationCallback
from model_zoo.promoter.sei import Sei, NonStrandSpecific


def upgrade_state_dict(state_dict, prefixes=["encoder.sentence_encoder.", "encoder."]):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict


class PromoterSPMSECallback(BaseSPMSEValidationCallback):
    """SP-MSE validation callback specifically for Promoter dataset."""
    
    def get_default_sampling_steps(self) -> int:
        """Get default sampling steps for Promoter (sequence length)."""
        return 1024
    
    def load_oracle_model(self):
        """Load Promoter SEI oracle model."""
        try:
            # Load SEI model with proper architecture
            sei_model = Sei(4096, 21907)  # 4096 seq length, 21907 features
            oracle = NonStrandSpecific(sei_model)
            
            # Load checkpoint if provided
            if self.oracle_path and self.oracle_path != 'null':
                checkpoint = torch.load(self.oracle_path, map_location='cpu')
                state_dict = upgrade_state_dict(checkpoint['state_dict'], prefixes=['module.'])
                oracle.load_state_dict(state_dict, strict=False)
            
            return oracle.eval()
        except Exception as e:
            print(f"Failed to load Promoter SEI oracle model: {e}")
            return None
    
    def get_oracle_predictions(self, sequences: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Get Promoter SEI oracle predictions for sequences using H3K4me3 filtering.
        
        Args:
            sequences: Input sequences (can be indices, one-hot, or promoter format)
            device: Device to run on
            
        Returns:
            H3K4me3 predictions tensor
        """
        if self.oracle_model is None:
            raise RuntimeError("Oracle model not loaded")
        
        # Load SEI features once if not already loaded
        if not hasattr(self, 'sei_features'):
            try:
                sei_features_path = getattr(self, 'sei_features_path', 'model_zoo/promoter/oracle_models/target.sei.names')
                self.sei_features = pd.read_csv(sei_features_path, sep='|', header=None)
            except:
                print("Warning: Could not load SEI features file, using all features")
                self.sei_features = None
        
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
        
        # Get SEI profile using the proper inference pattern
        predictions = self._get_sei_profile(sequences_one_hot, device)
        
        # Convert to tensor and ensure proper shape
        if isinstance(predictions, list):
            predictions = torch.tensor(predictions, device=device)
        else:
            predictions = torch.from_numpy(predictions).to(device)
        
        # Ensure predictions is 1D and add channel dimension
        if predictions.dim() == 0:
            predictions = predictions.unsqueeze(0)
        
        return predictions.unsqueeze(1)  # Add channel dim
    
    def _get_sei_profile(self, seq_one_hot, device):
        """
        Get SEI profile following the pattern from the example code.
        
        Args:
            seq_one_hot: One-hot encoded sequences (batch_size, seq_length, 4)
            device: Device to run on
            
        Returns:
            H3K4me3 predictions (batch_size,)
        """
        B, L, K = seq_one_hot.shape
        seq_one_hot = seq_one_hot.cpu()
        
        # Process in smaller batches to avoid GPU memory issues
        batch_size = min(32, B)  # Limit batch size to avoid OOM
        all_predictions = []
        
        for i in range(0, B, batch_size):
            end_idx = min(i + batch_size, B)
            batch_seq = seq_one_hot[i:end_idx]
            batch_B = batch_seq.shape[0]
            
            try:
                # Pad sequence to 4096 length as expected by SEI
                # Add 1536 bases on each side with uniform background (0.25 for each nucleotide)
                sei_inp = torch.cat([
                    torch.ones((batch_B, 4, 1536)) * 0.25,
                    batch_seq.transpose(1, 2),  # Convert to (batch, channels, length)
                    torch.ones((batch_B, 4, 1536)) * 0.25
                ], 2)  # batch_B x 4 x 4,096
                
                # Move to device and get predictions
                sei_inp = sei_inp.to(device)
                
                with torch.no_grad():
                    sei_out = self.oracle_model(sei_inp)
                    sei_out = sei_out.cpu().detach().numpy()  # batch_B x 21,907
                
                # Clean up GPU memory immediately
                del sei_inp
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Filter for H3K4me3 features if SEI features are available
                if self.sei_features is not None:
                    h3k4me3_mask = self.sei_features[1].str.strip().values == 'H3K4me3'
                    sei_out = sei_out[:, h3k4me3_mask]  # batch_B x 2,350 (H3K4me3 features)
                
                # Take mean across H3K4me3 features for this batch
                batch_pred = sei_out.mean(axis=1)  # batch_B
                all_predictions.append(batch_pred)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Clear cache and try with even smaller batch
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    print(f"GPU OOM, skipping batch {i}-{end_idx}")
                    # Return zeros for this batch as fallback
                    all_predictions.append(np.zeros(batch_B))
                else:
                    raise e
        
        # Concatenate all batch predictions
        predh3k4me3 = np.concatenate(all_predictions, axis=0)  # B
        
        return predh3k4me3
    
    def process_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process Promoter batch data.
        
        Args:
            batch: Raw batch data (promoter format or sequences/targets)
            
        Returns:
            Tuple of (sequences, targets)
        """
        # Handle case where batch might be a list or tuple
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            sequences, targets = batch
            return sequences, targets
        elif hasattr(batch, 'dim') and batch.dim() == 3 and batch.shape[-1] == 5:
            # Promoter format: (batch_size, seq_length, 5)
            sequences = batch  # Keep full format for oracle
            targets = batch[:, :, 4:5]  # Extract target part
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
    
    # Auto-resolve paths if not provided, use paths from config if available
    oracle_path = sp_mse_cfg.get('oracle_path')
    if oracle_path is None and hasattr(cfg, 'paths') and hasattr(cfg.paths, 'oracle_model'):
        oracle_path = cfg.paths.oracle_model
    elif oracle_path is None:
        oracle_path = 'model_zoo/promoter/oracle_models/best.sei.model.pth.tar'
    
    data_path = sp_mse_cfg.get('data_path')
    if data_path is None:
        data_path = None  # Promoter doesn't need a specific data file for SEI
    
    # Get SEI features path
    sei_features_path = 'model_zoo/promoter/oracle_models/target.sei.names'
    if hasattr(cfg, 'paths') and hasattr(cfg.paths, 'sei_features'):
        sei_features_path = cfg.paths.sei_features
    
    callback = PromoterSPMSECallback(
        oracle_path=oracle_path,
        data_path=data_path,
        validation_freq_epochs=sp_mse_cfg.get('validation_freq_epochs', 4),
        validation_samples=sp_mse_cfg.get('validation_samples', 1000),
        enabled=sp_mse_cfg.get('enabled', False),
        sampling_steps=sp_mse_cfg.get('sampling_steps'),
        early_stopping_patience=sp_mse_cfg.get('early_stopping_patience')
    )
    
    # Set SEI features path
    callback.sei_features_path = sei_features_path
    
    return callback