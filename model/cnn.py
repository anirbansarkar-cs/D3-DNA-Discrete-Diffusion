"""
Pure Convolutional Architecture for D3-DNA Discrete Diffusion

This module contains the core convolutional implementation that is completely
dataset-agnostic. All dataset-specific parameters are passed via configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from omegaconf import OmegaConf, DictConfig

from .layers import (
    TimestepEmbedder, GaussianFourierProjection, Dense, get_bias_dropout_scale
)


class ConvolutionalModel(nn.Module):
    """
    Pure Convolutional SEDD Model.
    
    This implementation is completely dataset-agnostic. All dataset-specific
    parameters (signal_dim, num_classes, sequence_length) are passed via config.
    """
    
    def __init__(self, config: DictConfig):
        super().__init__()
        
        if isinstance(config, dict):
            config = OmegaConf.create(config)
            
        self.config = config
        
        # Extract dataset-agnostic parameters from config
        self.absorb = config.graph.type == "absorb"
        vocab_size = config.tokens + (1 if self.absorb else 0)
        
        # These should be provided by dataset-specific config
        num_classes = config.dataset.num_classes
        sequence_length = config.dataset.sequence_length
        
        # Convolutional architecture parameters
        n_channels = getattr(config.model, 'conv_channels', 256)
        embed_dim = config.model.cond_dim
        
        # Core components
        self.sigma_map = TimestepEmbedder(embed_dim)
        
        # Gaussian Fourier projection for time embedding
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Initial convolution
        # Input: one-hot sequence + signal features
        input_channels = vocab_size + 1  # +1 for signal features
        self.linear = nn.Conv1d(input_channels, n_channels, kernel_size=9, padding=4)
        
        # Convolutional blocks with dilated convolutions
        self.conv_blocks = self._create_conv_blocks(n_channels)
        self.denses = nn.ModuleList([Dense(embed_dim, n_channels) for _ in range(len(self.conv_blocks))])
        self.norms = nn.ModuleList([nn.GroupNorm(1, n_channels) for _ in range(len(self.conv_blocks))])
        
        # Activation functions
        self.act = nn.SiLU()
        self.scale = nn.Parameter(torch.ones(1))
        
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(n_channels, vocab_size, kernel_size=1)
        )
        
        # Model configuration
        self.scale_by_sigma = getattr(config.model, 'scale_by_sigma', False)
    
    def _create_conv_blocks(self, n_channels: int) -> nn.ModuleList:
        """
        Create dilated convolutional blocks.
        
        Args:
            n_channels: Number of channels
            
        Returns:
            ModuleList of convolutional blocks
        """
        # Pattern of dilations: [1, 1, 4, 16, 64] repeated 4 times
        dilation_pattern = [1, 1, 4, 16, 64]
        blocks = []
        
        for _ in range(4):  # Repeat pattern 4 times
            for dilation in dilation_pattern:
                if dilation == 1:
                    padding = 4  # kernel_size=9, padding=4
                else:
                    padding = 4 * dilation  # kernel_size=9, dilation=dilation
                
                blocks.append(
                    nn.Conv1d(n_channels, n_channels, kernel_size=9, 
                             dilation=dilation, padding=padding)
                )
        
        return nn.ModuleList(blocks)
    
    def forward(self, indices: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                train: bool = True, sigma: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the convolutional model.
        
        Args:
            indices: Token indices (batch_size, seq_length)
            labels: Label/signal tensor - shape depends on dataset, or None for unconditional
            train: Training mode flag
            sigma: Noise level (batch_size,)
            
        Returns:
            Model output (batch_size, seq_length, vocab_size)
        """
        # Convert indices to one-hot representation
        num_classes = self.config.dataset.num_classes
        x = F.one_hot(indices, num_classes=num_classes).float()
        
        # Process labels based on dimensionality
        # This is where datasets provide their signal in different formats
        if labels is not None:
            if labels.dim() == 2:
                # Labels are (batch_size, signal_dim) - need to broadcast to sequence length
                if labels.shape[-1] == 1:
                    # Single signal value per sequence (e.g., promoter)
                    signal_features = labels.unsqueeze(-1).expand(-1, -1, x.shape[1])
                else:
                    # Multiple signal values that need to be processed into sequence-level features
                    # This should be handled by dataset-specific preprocessing
                    raise NotImplementedError(
                        "Multi-dimensional labels need dataset-specific preprocessing. "
                        "This should be handled in dataset-specific model wrappers."
                    )
            elif labels.dim() == 3:
                # Labels are already (batch_size, seq_length, signal_dim) or (batch_size, signal_dim, seq_length)
                if labels.shape[1] == x.shape[1]:  # (batch_size, seq_length, signal_dim)
                    signal_features = labels.transpose(1, 2)  # -> (batch_size, signal_dim, seq_length)
                elif labels.shape[2] == x.shape[1]:  # (batch_size, signal_dim, seq_length)
                    signal_features = labels
                else:
                    raise ValueError(f"Label dimensions {labels.shape} don't match sequence length {x.shape[1]}")
            else:
                raise ValueError(f"Unsupported label dimensions: {labels.shape}")
            
            # Concatenate sequence and signal features
            x = torch.cat([x, signal_features], dim=-1)  # (batch_size, seq_length, vocab_size + signal_dim)
        else:
            # For unconditional generation, create zero signal features
            signal_features = torch.zeros(x.shape[0], 1, x.shape[1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, signal_features], dim=-1)  # (batch_size, seq_length, vocab_size + 1)
        
        # Transpose for conv1d: (batch_size, features, seq_length)
        x = x.permute(0, 2, 1)
        
        # Initial convolution
        out = self.act(self.linear(x))
        
        # Time/noise conditioning
        c = F.silu(self.sigma_map(sigma))
        
        # Apply convolutional blocks with residual connections
        for block, dense, norm in zip(self.conv_blocks, self.denses, self.norms):
            # Add time conditioning and apply normalization
            conditioned = out + dense(c)[:, :, None]
            normalized = norm(conditioned)
            
            # Apply convolution with activation
            h = self.act(block(normalized))
            
            # Residual connection if shapes match
            if h.shape == out.shape:
                out = h + out
            else:
                out = h
        
        # Final output layer
        x = self.final(out)
        
        # Transpose back: (batch_size, seq_length, vocab_size)
        x = x.permute(0, 2, 1)
        
        return x


def create_convolutional_model(config: DictConfig) -> ConvolutionalModel:
    """
    Factory function to create a convolutional model.
    
    Args:
        config: Configuration containing model and dataset parameters
        
    Returns:
        Convolutional model instance
    """
    return ConvolutionalModel(config)