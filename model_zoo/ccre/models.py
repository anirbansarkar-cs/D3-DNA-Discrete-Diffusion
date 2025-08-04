"""
cCRE Dataset-Specific Models

This module provides cCRE-specific model factory functions that configure
the base architectures for unlabeled 512bp sequences.
"""

import os
import torch
from omegaconf import DictConfig
from typing import Tuple, Any
from model.transformer import TransformerModel, create_transformer_model
from model.cnn import ConvolutionalModel, create_convolutional_model


def create_ccre_transformer_model(config: DictConfig) -> TransformerModel:
    """Create a transformer model configured for cCRE dataset."""
    # Ensure dataset-specific config is set
    if not hasattr(config, 'dataset'):
        config.dataset = {}
    config.dataset.name = 'ccre'
    config.dataset.num_classes = 4
    config.dataset.sequence_length = 512
    config.dataset.signal_dim = 0  # No labels for unlabeled data
    
    # Use base transformer model - it now handles labels=None properly
    return TransformerModel(config)


def create_ccre_convolutional_model(config: DictConfig) -> ConvolutionalModel:
    """Create a convolutional model configured for cCRE dataset."""
    # Ensure dataset-specific config is set
    if not hasattr(config, 'dataset'):
        config.dataset = {}
    config.dataset.name = 'ccre'
    config.dataset.num_classes = 4
    config.dataset.sequence_length = 512
    config.dataset.signal_dim = 0  # No labels for unlabeled data
    
    # Use base convolutional model - it now handles labels=None properly
    return ConvolutionalModel(config)


def create_model(config: DictConfig, architecture: str):
    """
    Factory function to create cCRE models.
    
    Args:
        config: Configuration object
        architecture: 'transformer' or 'convolutional'
        
    Returns:
        Model instance
    """
    if architecture.lower() == 'transformer':
        return create_ccre_transformer_model(config)
    elif architecture.lower() == 'convolutional':
        return create_ccre_convolutional_model(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def load_trained_model(checkpoint_path: str, config: DictConfig, architecture: str, device: str = 'cuda') -> Tuple[torch.nn.Module, Any, Any]:
    """
    Load trained cCRE model from specific checkpoint file.
    
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
    
    print(f"Loading cCRE {architecture} model from {checkpoint_path}")
    
    # Validate checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create model
    model = create_model(config, architecture)
    
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
    
    print(f"✓ cCRE {architecture} model loaded successfully")
    
    return model, graph, noise


# Legacy compatibility
def create_ccre_model(config: DictConfig, architecture: str = None):
    """Legacy function for backward compatibility."""
    if architecture is None:
        architecture = config.model.architecture
    return create_model(config, architecture)