"""
LentIMPRA Dataset-Specific Models

This module provides LentIMPRA-specific wrappers around the core architectures.
These handle the dataset-specific signal preprocessing and model instantiation.
"""

import os
import torch
from omegaconf import DictConfig
from typing import Tuple, Any
from model.transformer import TransformerModel
from model.cnn import ConvolutionalModel


class LentIMPRATransformerModel(TransformerModel):
    """LentIMPRA-specific transformer model wrapper."""
    
    def __init__(self, config: DictConfig):
        # Ensure dataset-specific config is set
        if not hasattr(config, 'dataset'):
            config.dataset = {}
        config.dataset.name = 'lentimpra'
        config.dataset.signal_dim = 1  # Single regression target
        config.dataset.num_classes = 4
        config.dataset.sequence_length = 230
        
        super().__init__(config)


class LentIMPRAConvolutionalModel(ConvolutionalModel):
    """LentIMPRA-specific convolutional model wrapper."""
    
    def __init__(self, config: DictConfig):
        # Ensure dataset-specific config is set
        if not hasattr(config, 'dataset'):
            config.dataset = {}
        config.dataset.name = 'lentimpra'
        config.dataset.signal_dim = 1  # Single regression target
        config.dataset.num_classes = 4
        config.dataset.sequence_length = 230
        
        super().__init__(config)
    
    def forward(self, indices: torch.Tensor, labels: torch.Tensor, 
                train: bool, sigma: torch.Tensor) -> torch.Tensor:
        """
        LentIMPRA-specific forward pass with label preprocessing.
        
        LentIMPRA uses single regression target for regulatory activity.
        """
        # Convert indices to one-hot
        x = torch.nn.functional.one_hot(indices, num_classes=4).float()
        
        # LentIMPRA-specific label processing for conv architecture
        # Uses single regression value broadcasted to sequence length
        label_emb = torch.nn.Sequential(
            torch.nn.Linear(1, self.config.dataset.sequence_length),
            torch.nn.SiLU(),
            torch.nn.Linear(self.config.dataset.sequence_length, self.config.dataset.sequence_length),
        ).to(labels.device)
        
        # Ensure labels are proper shape for embedding
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)  # (N,) -> (N, 1)
        
        processed_labels = torch.unsqueeze(label_emb(labels), dim=2)
        
        # Concatenate sequence and processed labels
        x = torch.cat([x, processed_labels], dim=-1)
        x = x.permute(0, 2, 1)
        
        # Apply initial convolution
        out = self.act(self.linear(x))
        
        # Time conditioning
        c = torch.nn.functional.silu(self.sigma_map(sigma))
        
        # Apply conv blocks
        for block, dense, norm in zip(self.conv_blocks, self.denses, self.norms):
            h = self.act(block(norm(out + dense(c)[:, :, None])))
            if h.shape == out.shape:
                out = h + out
            else:
                out = h
        
        # Final output
        x = self.final(out)
        x = x.permute(0, 2, 1)
        
        return x


def create_model(config: DictConfig, architecture: str):
    """
    Factory function to create LentIMPRA models.
    
    Args:
        config: Configuration object
        architecture: 'transformer' or 'convolutional'
        
    Returns:
        Model instance
    """
    if architecture.lower() == 'transformer':
        return LentIMPRATransformerModel(config)
    elif architecture.lower() == 'convolutional':
        return LentIMPRAConvolutionalModel(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def load_trained_model(checkpoint_path: str, config: DictConfig, architecture: str, device: str = 'cuda') -> Tuple[torch.nn.Module, Any, Any]:
    """
    Load trained LentIMPRA model from specific checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration object
        architecture: 'transformer' or 'convolutional'
        device: Device to load model on
        
    Returns:
        Tuple of (model, graph, noise) ready for sampling/evaluation
    """
    # Import required components
    from model.ema import ExponentialMovingAverage
    from utils import graph_lib, noise_lib
    
    print(f"Loading LentIMPRA {architecture} model from {checkpoint_path}")
    
    # Validate checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create model
    if architecture.lower() == 'transformer':
        model = LentIMPRATransformerModel(config)
    elif architecture.lower() == 'convolutional':
        model = LentIMPRAConvolutionalModel(config)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    model.to(device)
    
    # Initialize graph and noise
    graph = graph_lib.get_graph(config, device)
    noise = noise_lib.get_noise(config).to(device)
    
    # Initialize EMA
    ema = ExponentialMovingAverage(model.parameters(), decay=config.training.ema)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if checkpoint_path.endswith('.ckpt'):
        # Lightning checkpoint
        if 'state_dict' in checkpoint:
            model_state = {}
            ema_state = {}
            
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('score_model.'):
                    model_key = key.replace('score_model.', '')
                    model_state[model_key] = value
                elif key.startswith('model.'):
                    model_key = key.replace('model.', '')
                    model_state[model_key] = value
                elif key.startswith('ema.'):
                    ema_key = key.replace('ema.', '')
                    ema_state[ema_key] = value
            
            # Load model weights
            if model_state:
                model.load_state_dict(model_state, strict=False)
                print("✓ Loaded model weights from Lightning checkpoint")
            
            # Load EMA weights
            if ema_state:
                ema.load_state_dict(ema_state)
                print("✓ Loaded EMA weights from Lightning checkpoint")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print("✓ Loaded model weights from Lightning checkpoint")
    else:
        # Original D3 format
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
            print("✓ Loaded model weights from original checkpoint")
        
        if 'ema' in checkpoint:
            ema.load_state_dict(checkpoint['ema'])
            print("✓ Loaded EMA weights from original checkpoint")
    
    # Apply EMA weights to model
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    
    print(f"✓ LentIMPRA {architecture} model loaded successfully")
    
    return model, graph, noise


# Legacy compatibility
def create_lentimpra_model(config: DictConfig, architecture: str = None):
    """Legacy function for backward compatibility."""
    if architecture is None:
        architecture = config.model.architecture
    return create_model(config, architecture)