"""
Promoter Dataset-Specific Models

This module provides Promoter-specific wrappers around the core architectures.
These handle the dataset-specific signal preprocessing and model instantiation.
"""

import os
import torch
import math
from omegaconf import DictConfig
from typing import Tuple, Any
from model.transformer import TransformerModel, create_transformer_model
from model.cnn import ConvolutionalModel, create_convolutional_model


class PromoterEmbeddingLayer(torch.nn.Module):
    """Promoter-specific embedding layer that matches the original implementation."""
    
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.empty((vocab_dim, dim)))
        self.signal_embedding = torch.nn.Linear(1, dim)  # Promoter uses signal_dim=1
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x, y):
        vocab_embed = self.embedding[x]
        signal_embed = self.signal_embedding(y)
        return torch.add(vocab_embed, signal_embed)


class PromoterTransformerModel(TransformerModel):
    """Promoter-specific transformer model wrapper."""
    
    def __init__(self, config: DictConfig):
        # Ensure dataset-specific config is set
        if not hasattr(config, 'dataset'):
            config.dataset = {}
        config.dataset.name = 'promoter'
        config.dataset.signal_dim = 1
        config.dataset.num_classes = 4
        config.dataset.sequence_length = 1024
        
        super().__init__(config)
        
        # Replace the vocab_embed with promoter-specific version
        vocab_size = config.tokens + (1 if getattr(config.graph, 'type', '') == 'absorb' else 0)
        self.vocab_embed = PromoterEmbeddingLayer(config.model.hidden_size, vocab_size)


class PromoterConvolutionalModel(ConvolutionalModel):
    """Promoter-specific convolutional model wrapper."""
    
    def __init__(self, config: DictConfig):
        # Ensure dataset-specific config is set
        if not hasattr(config, 'dataset'):
            config.dataset = {}
        config.dataset.name = 'promoter'
        config.dataset.signal_dim = 1
        config.dataset.num_classes = 4
        config.dataset.sequence_length = 1024
        
        super().__init__(config)
    
    def forward(self, indices: torch.Tensor, labels: torch.Tensor, 
                train: bool, sigma: torch.Tensor) -> torch.Tensor:
        """
        Promoter-specific forward pass with label preprocessing.
        
        Promoter uses labels directly without additional embedding.
        """
        # Convert indices to one-hot
        x = torch.nn.functional.one_hot(indices, num_classes=4).float()
        
        # Promoter uses labels directly - no additional processing needed
        # Concatenate sequence and labels directly
        x = torch.cat([x, labels], dim=-1)
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
    Factory function to create Promoter models.
    
    Args:
        config: Configuration object
        architecture: 'transformer' or 'convolutional'
        
    Returns:
        Model instance
    """
    if architecture.lower() == 'transformer':
        return PromoterTransformerModel(config)
    elif architecture.lower() == 'convolutional':
        return PromoterConvolutionalModel(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def load_trained_model(checkpoint_path: str, config: DictConfig, architecture: str, device: str = 'cuda') -> Tuple[torch.nn.Module, Any, Any]:
    """
    Load trained Promoter model from specific checkpoint file.
    
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
    
    print(f"Loading Promoter {architecture} model from {checkpoint_path}")
    
    # Validate checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create model
    if architecture.lower() == 'transformer':
        model = PromoterTransformerModel(config)
    elif architecture.lower() == 'convolutional':
        model = PromoterConvolutionalModel(config)
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
    
    print(f"✓ Promoter {architecture} model loaded successfully")
    
    return model, graph, noise


# Legacy compatibility
def create_promoter_model(config: DictConfig, architecture: str = None):
    """Legacy function for backward compatibility."""
    if architecture is None:
        architecture = config.model.architecture
    return create_model(config, architecture)