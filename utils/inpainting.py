"""
Inpainting utilities for constrained sequence generation.
"""
import torch
import numpy as np
import h5py
from pathlib import Path


class InpaintingManager:
    """
    Manages inpainting constraints for sequence generation.

    Two modes:
    - inpaint_motifs: Fix positions OUTSIDE motif regions
    - inpaint_not_motifs: Fix positions INSIDE motif regions
    """

    def __init__(self, mode, data_file, device='cpu', seed=None, expected_signal_dim=None):
        """
        Initialize inpainting manager.

        Args:
            mode: Either 'inpaint_motifs' or 'inpaint_not_motifs'
            data_file: Path to all_hits_combined.h5
            device: torch device
            seed: Random seed for choosing between dev/hk when both available
            expected_signal_dim: Expected dimension of Y_target (e.g., 2 for DeepSTARR)
        """
        if mode not in ['inpaint_motifs', 'inpaint_not_motifs']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'inpaint_motifs' or 'inpaint_not_motifs'")

        self.mode = mode
        self.device = device
        self.rng = np.random.RandomState(seed)

        # Load data
        with h5py.File(data_file, 'r') as f:
            self.X = torch.from_numpy(f['X'][:]).float().to(device)  # (n, 249, 4)
            self.start_dev = f['start_dev'][:]
            self.end_dev = f['end_dev'][:]
            self.start_hk = f['start_hk'][:]
            self.end_hk = f['end_hk'][:]
            self.Y_target = torch.from_numpy(f['Y_target'][:]).float().to(device)  # (n, signal_dim)

        self.num_samples = len(self.X)

        # Validate Y_target dimensions if expected_signal_dim is provided
        if expected_signal_dim is not None:
            actual_dim = self.Y_target.shape[1]
            if actual_dim != expected_signal_dim:
                raise ValueError(
                    f"Y_target dimension mismatch: expected {expected_signal_dim}, "
                    f"got {actual_dim}. The inpainting data file may not match this dataset."
                )

        # Precompute masks for each sample
        self.masks = self._precompute_masks()

    def _choose_positions(self, idx):
        """
        Choose which positions (dev or hk) to use for sample idx.

        Returns:
            (start, end) tuple, or (None, None) if both are NaN
        """
        start_dev = self.start_dev[idx]
        end_dev = self.end_dev[idx]
        start_hk = self.start_hk[idx]
        end_hk = self.end_hk[idx]

        # Check which positions are available
        dev_available = not (np.isnan(start_dev) or np.isnan(end_dev))
        hk_available = not (np.isnan(start_hk) or np.isnan(end_hk))

        if not dev_available and not hk_available:
            return None, None
        elif dev_available and not hk_available:
            return int(start_dev), int(end_dev)
        elif hk_available and not dev_available:
            return int(start_hk), int(end_hk)
        else:
            # Both available, randomly choose
            if self.rng.rand() < 0.5:
                return int(start_dev), int(end_dev)
            else:
                return int(start_hk), int(end_hk)

    def _precompute_masks(self):
        """
        Precompute masks for all samples.

        Returns:
            List of masks, one per sample. Each mask is a boolean tensor of shape (seq_len,)
            where True means "fix this position" and False means "generate freely"
        """
        masks = []
        seq_len = self.X.shape[1]

        for idx in range(self.num_samples):
            start, end = self._choose_positions(idx)

            if start is None:
                # No position data, don't fix anything
                mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
            else:
                mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)

                if self.mode == 'inpaint_motifs':
                    # Fix positions OUTSIDE [start, end]
                    mask[:start] = True
                    mask[end+1:] = True
                elif self.mode == 'inpaint_not_motifs':
                    # Fix positions INSIDE [start, end] (inclusive)
                    mask[start:end+1] = True

            masks.append(mask)

        return masks

    def get_projection_fn(self, sample_indices):
        """
        Get projection function for a batch of samples.

        Args:
            sample_indices: List or tensor of sample indices to use for inpainting

        Returns:
            proj_fun: Function that takes x and returns x with fixed positions
        """
        if isinstance(sample_indices, torch.Tensor):
            sample_indices = sample_indices.cpu().numpy()

        # Get reference sequences and masks for this batch
        reference_seqs = self.X[sample_indices]  # (batch, seq_len, 4)
        masks = torch.stack([self.masks[idx] for idx in sample_indices])  # (batch, seq_len)

        def proj_fun(x):
            """
            Project x to satisfy inpainting constraints.

            Args:
                x: Tensor of shape (batch, seq_len) with token indices

            Returns:
                x with fixed positions replaced by reference values
            """
            # x is token indices, reference_seqs is one-hot
            # Convert reference to token indices
            reference_tokens = reference_seqs.argmax(dim=-1)  # (batch, seq_len)

            # Apply mask: where mask is True, use reference; where False, use generated
            x_projected = torch.where(masks, reference_tokens, x)

            return x_projected

        return proj_fun

    def get_identity_projection(self):
        """
        Get identity projection function (no constraints).

        Returns:
            proj_fun: Function that returns input unchanged
        """
        def proj_fun(x):
            return x
        return proj_fun


def create_inpainting_manager(mode=None, data_file=None, device='cpu', seed=None, expected_signal_dim=None):
    """
    Create inpainting manager if mode is specified, otherwise return None.

    Args:
        mode: Either 'inpaint_motifs', 'inpaint_not_motifs', or None
        data_file: Path to all_hits_combined.h5
        device: torch device
        seed: Random seed
        expected_signal_dim: Expected dimension of Y_target for validation

    Returns:
        InpaintingManager or None
    """
    if mode is None or mode == 'none':
        return None

    if data_file is None:
        raise ValueError("data_file must be provided when using inpainting mode")

    return InpaintingManager(mode, data_file, device, seed, expected_signal_dim)
