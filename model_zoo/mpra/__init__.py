"""
MPRA Dataset Support for D3-DNA Discrete Diffusion

This module provides complete support for MPRA (Massively Parallel Reporter Assay)
datasets in the D3 framework, including data loading, model training, evaluation,
and sampling capabilities.

The MPRA dataset contains DNA sequences with regulatory activity measurements
from three cell lines: K562, HepG2, and SK-N-SH.
"""

from .data import (
    MPRADataset,
    get_mpra_datasets,
    get_mpra_dataloaders,
    get_mpra_dataloaders_with_cycle
)

from .models import (
    MPRATransformerModel,
    MPRAConvolutionalModel,
    create_model,
    load_trained_model,
    create_mpra_model  # Legacy compatibility
)

from .sp_mse_callback import (
    MPRASPMSECallback,
    create_mpra_sp_mse_callback
)

# Import oracle model for evaluation
from .mpra import PL_MPRA, MPRA

__all__ = [
    # Data loading
    'MPRADataset',
    'get_mpra_datasets', 
    'get_mpra_dataloaders',
    'get_mpra_dataloaders_with_cycle',
    
    # Models
    'MPRATransformerModel',
    'MPRAConvolutionalModel', 
    'create_model',
    'load_trained_model',
    'create_mpra_model',
    
    # Oracle models
    'PL_MPRA',
    'MPRA',
    
    # Callbacks
    'MPRASPMSECallback',
    'create_mpra_sp_mse_callback',
]

# Dataset metadata
DATASET_INFO = {
    'name': 'mpra',
    'sequence_length': 200,
    'num_classes': 4,
    'signal_dim': 3,  # K562, HepG2, SK-N-SH cell lines
    'cell_lines': ['K562', 'HepG2', 'SK-N-SH'],
    'data_format': 'h5',
    'input_keys': ['x_train', 'x_valid', 'x_test'],
    'target_keys': ['y_train', 'y_valid', 'y_test'],
    'description': 'Massively Parallel Reporter Assay data with regulatory activity measurements from three cell lines'
}