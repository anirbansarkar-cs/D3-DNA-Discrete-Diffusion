"""
Inpainting utilities for DNA sequence generation.

Provides functionality to constrain specific positions during sampling
based on motif positions from CSV files.
"""

import pandas as pd
import torch
from typing import List, Union
from pathlib import Path


def load_motif_positions(
    csv_path: Union[str, Path],
    sequence_length: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Load motif positions from CSV and create a binary mask.

    Expected CSV format:
        motif_name,start,end,strand
        pattern_0,10,25,+
        pattern_1,50,65,-
        ...

    Args:
        csv_path: Path to CSV file with motif positions
        sequence_length: Length of sequences (e.g., 230 for LentiMPRA)
        device: Device for tensors

    Returns:
        Binary mask tensor of shape (sequence_length,) where 1 indicates motif position
    """
    df = pd.read_csv(csv_path)

    # Create binary mask
    mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)

    for _, row in df.iterrows():
        start = int(row['start'])
        end = int(row['end'])
        # Mark motif positions
        mask[start:end] = True

    return mask


def create_inpainting_projection(
    initial_x: torch.Tensor,
    motif_mask: torch.Tensor,
    mode: str
):
    """
    Create a projection function that fixes positions during sampling.

    Args:
        initial_x: Initial sequence tensor of shape (batch_size, seq_len)
        motif_mask: Binary mask of shape (seq_len,) where True indicates motif positions
        mode: Either 'motif' or 'not_motif'
            - 'motif': Fix motif positions to initial values (evolve background)
            - 'not_motif': Fix non-motif positions to initial values (evolve motifs)

    Returns:
        Projection function that takes current x and returns constrained x
    """
    if mode not in ['motif', 'not_motif']:
        raise ValueError(f"Mode must be 'motif' or 'not_motif', got: {mode}")

    # Determine which positions to fix
    if mode == 'motif':
        fix_mask = motif_mask  # Fix motif positions
    else:  # mode == 'not_motif'
        fix_mask = ~motif_mask  # Fix non-motif positions

    def proj_fun(x):
        """Apply inpainting constraints: copy initial values at fixed positions."""
        # x shape: (batch_size, seq_len)
        # initial_x shape: (batch_size, seq_len)
        # fix_mask shape: (seq_len,) - broadcasts to (batch_size, seq_len)

        # Create output by copying x
        result = x.clone()

        # Overwrite fixed positions with initial values
        result[:, fix_mask] = initial_x[:, fix_mask]

        return result

    return proj_fun
