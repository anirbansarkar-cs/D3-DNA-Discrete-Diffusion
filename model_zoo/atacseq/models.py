"""
ATACSeq Dataset-Specific Models

This module provides ATACSeq-specific wrappers around the core architectures.
These handle the dataset-specific signal preprocessing and model instantiation.
"""

import torch
from typing import Optional
from omegaconf import DictConfig
from model.transformer import TransformerModel, create_transformer_model
from model.cnn import ConvolutionalModel, create_convolutional_model


class ATACSeqTransformerModel(TransformerModel):
    """ATACSeq-specific transformer model wrapper."""
    
    def __init__(self, config: DictConfig):
        # Ensure dataset-specific config is set
        if not hasattr(config, 'dataset'):
            config.dataset = {}
        config.dataset.name = 'atacseq'
        config.dataset.signal_dim = 18
        config.dataset.num_classes = 4
        config.dataset.sequence_length = 1001
        
        # Convert to OmegaConf if needed
        if isinstance(config, dict):
            from omegaconf import OmegaConf
            config = OmegaConf.create(config)
        
        super().__init__(config)


# NOTE: test this conv implementation
class ATACSeqConvolutionalModel(ConvolutionalModel):
    """ATACSeq-specific convolutional model wrapper."""
    
    def __init__(self, config: DictConfig):
        # Ensure dataset-specific config is set
        if not hasattr(config, 'dataset'):
            config.dataset = {}
        config.dataset.name = 'atacseq'
        config.dataset.signal_dim = 18
        config.dataset.num_classes = 4
        config.dataset.sequence_length = 1001
        
        # Convert to OmegaConf if needed
        if isinstance(config, dict):
            from omegaconf import OmegaConf
            config = OmegaConf.create(config)
        
        super().__init__(config)
    
    def forward(self, indices: torch.Tensor, labels: torch.Tensor, 
                train: bool, sigma: torch.Tensor) -> torch.Tensor:
        """
        ATACSeq-specific forward pass with label preprocessing.
        
        ATACSeq uses a special label embedding layer for convolutional architecture.
        """
        # Convert indices to one-hot
        x = torch.nn.functional.one_hot(indices, num_classes=4).float()
        
        # ATACSeq-specific label processing for conv architecture
        # Uses a label embedding layer that broadcasts to sequence length
        label_emb = torch.nn.Sequential(
            torch.nn.Linear(18, self.config.dataset.sequence_length),
            torch.nn.SiLU(),
            torch.nn.Linear(self.config.dataset.sequence_length, self.config.dataset.sequence_length),
        ).to(labels.device)
        
        processed_labels = torch.unsqueeze(label_emb(labels), dim=18)
        
        # Concatenate sequence and processed labels
        x = torch.cat([x, processed_labels], dim=-1)
        x = x.permute(0, 18, 1)
        
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
        x = x.permute(0, 18, 1)
        
        return x


def create_model(config: DictConfig, architecture: str):
    """
    Factory function to create ATACSeq models.
    
    Args:
        config: Configuration object
        architecture: 'transformer' or 'convolutional'
        
    Returns:
        Model instance
    """
    if architecture.lower() == 'transformer':
        return ATACSeqTransformerModel(config)
    elif architecture.lower() == 'convolutional':
        return ATACSeqConvolutionalModel(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# Legacy compatibility
def create_atacseq_model(config: DictConfig, architecture: Optional[str] = None):
    """Legacy function for backward compatibility."""
    if architecture is None:
        architecture = config.model.architecture
    return create_model(config, architecture)