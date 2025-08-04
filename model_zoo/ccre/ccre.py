"""
cCRE Dataset Module

This module provides the main interface for the cCRE (candidate Cis-Regulatory Elements) dataset.
Unlike labeled datasets, cCRE focuses on unlabeled 512bp DNA sequences for unconditional generation
and variant effect prediction.

Since cCRE data has no labels, there is no oracle model to define here.
The dataset is used primarily for:
1. Unconditional sequence generation
2. Variant effect prediction using the diffusion model itself
3. TraitGym benchmarking
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Dataset information
DATASET_NAME = "cCRE"
SEQUENCE_LENGTH = 512
NUM_CLASSES = 4  # A, C, G, T
HAS_LABELS = False  # Unlabeled dataset

# Default data file path
DEFAULT_DATA_FILE = "/grid/koo/home/yiyu/scratch/cCRE_Processing/output_h5s/ccres_512bp_no_N.h5"

def get_dataset_info():
    """Get basic information about the cCRE dataset."""
    return {
        'name': DATASET_NAME,
        'sequence_length': SEQUENCE_LENGTH,
        'num_classes': NUM_CLASSES,
        'has_labels': HAS_LABELS,
        'description': 'Candidate Cis-Regulatory Elements (cCRE) - 512bp unlabeled DNA sequences',
        'use_cases': [
            'Unconditional sequence generation',
            'Variant effect prediction',
            'TraitGym benchmarking'
        ],
        'default_data_file': DEFAULT_DATA_FILE
    }

def print_dataset_info():
    """Print information about the cCRE dataset."""
    info = get_dataset_info()
    
    print("=" * 50)
    print(f"Dataset: {info['name']}")
    print("=" * 50)
    print(f"Sequence Length: {info['sequence_length']} bp")
    print(f"Number of Classes: {info['num_classes']} (A, C, G, T)")
    print(f"Has Labels: {info['has_labels']}")
    print(f"Description: {info['description']}")
    print("\nUse Cases:")
    for use_case in info['use_cases']:
        print(f"  - {use_case}")
    print(f"\nDefault Data File: {info['default_data_file']}")
    print("=" * 50)

if __name__ == '__main__':
    print_dataset_info()