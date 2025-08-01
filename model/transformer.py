"""
Pure Transformer Architecture for D3-DNA Discrete Diffusion

This module contains the core transformer implementation that is completely
dataset-agnostic. All dataset-specific parameters are passed via configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from omegaconf import OmegaConf, DictConfig
import math

from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

from . import rotary
from .layers import (
    LayerNorm, TimestepEmbedder, LabelEmbedder, #EmbeddingLayer,
    modulate_fused, get_bias_dropout_scale
)


class DDiTBlock(nn.Module):
    """
    Diffusion Transformer Block with attention and feed-forward layers.
    """
    
    def __init__(self, dim: int, n_heads: int, cond_dim: int, 
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout

        # Attention layers
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward layers
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        # Adaptive layer normalization
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return get_bias_dropout_scale()(self.training)

    def forward(self, x: torch.Tensor, rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor], 
                c: torch.Tensor, seqlens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_length, dim)
            rotary_cos_sin: Rotary position encoding (cos, sin)
            c: Conditioning tensor (batch_size, cond_dim)
            seqlens: Optional sequence lengths for variable-length attention
            
        Returns:
            Output tensor (batch_size, seq_length, dim)
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # Adaptive layer normalization modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # Multi-head self-attention
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        
        # Apply rotary position embedding
        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        
        # Prepare sequence lengths for flash attention
        if seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=x.device
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)
            
        # Flash attention
        x = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, seq_len, 0., causal=False)
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

        # Apply attention output projection with gating
        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # Feed-forward network with gating
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), 
            None, gate_mlp, x, self.dropout
        )
        
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim, signal_dim=2):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors, 
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        #remove if label embedding is used
        self.signal_embedding = nn.Linear(signal_dim, dim)
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x, y):
        vocab_embed = self.embedding[x] #return only this if label embedding is used
        if y is not None:
            signal_embed = self.signal_embedding(y.to(torch.float32))
            return torch.add(vocab_embed, signal_embed[:, None, :]) #[:, None, :] extra for deepstarr
        else:
            # For unconditional generation, return only vocab embedding
            return vocab_embed


class DDitFinalLayer(nn.Module):
    """Final output layer for the transformer."""
    
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TransformerModel(nn.Module):
    """
    Pure Transformer SEDD Model.
    
    This implementation is completely dataset-agnostic. All dataset-specific
    parameters (num_classes, sequence_length) are passed via config.
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
        signal_dim = config.dataset.signal_dim
        class_dropout_prob = getattr(config.model, 'class_dropout_prob', 0.1)
        
        # Core components
        self.vocab_embed = EmbeddingLayer(
            dim=config.model.hidden_size, 
            vocab_dim=vocab_size,
            signal_dim=config.dataset.signal_dim,
        )

        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.label_embed = LabelEmbedder(num_classes, config.model.cond_dim, class_dropout_prob)
        self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DDiTBlock(
                dim=config.model.hidden_size, 
                n_heads=config.model.n_heads, 
                cond_dim=config.model.cond_dim,
                dropout=config.model.dropout
            ) 
            for _ in range(config.model.n_blocks)
        ])
        
        # Output layer
        self.output_layer = DDitFinalLayer(
            hidden_size=config.model.hidden_size, 
            out_channels=vocab_size, 
            cond_dim=config.model.cond_dim
        )
        
        # Model configuration
        self.scale_by_sigma = getattr(config.model, 'scale_by_sigma', False)

    def forward(self, indices: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                train: bool = True, sigma: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            indices: Token indices (batch_size, seq_length)
            labels: Label/signal tensor (batch_size, signal_dim) or None for unconditional
            train: Training mode flag
            sigma: Noise level (batch_size,)
            
        Returns:
            Model output (batch_size, seq_length, vocab_size)
        """
        # Embedding
        x = self.vocab_embed(indices, labels)
        
        # Conditioning
        c = F.silu(self.sigma_map(sigma))
        # Note: label_embed could be added here if needed: + self.label_embed(labels, train)
        
        # Rotary position encoding
        rotary_cos_sin = self.rotary_emb(x)

        # Forward through transformer blocks
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c, seqlens=None)
            x = self.output_layer(x, c)

        # Mask out the input tokens (standard diffusion technique)
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
        
        return x


def create_transformer_model(config: DictConfig) -> TransformerModel:
    """
    Factory function to create a transformer model.
    
    Args:
        config: Configuration containing model and dataset parameters
        
    Returns:
        Transformer model instance
    """
    return TransformerModel(config)