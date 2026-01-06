"""
Inpainting utilities for DNA sequence generation.

Provides functionality to constrain specific positions during sampling
based on pattern matches from CSV files.
"""

import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


class InpaintingManager:
    """
    Manages inpainting constraints for batch sampling.

    Loads pattern hit data from CSV files and creates projection functions
    that fix specific positions to specified token IDs during sampling.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        pattern_to_sequence: Optional[Dict[str, str]] = None,
        device: str = "cuda"
    ):
        """
        Args:
            csv_path: Path to CSV file with pattern hits (e.g., Dev_high_hits.csv)
            pattern_to_sequence: Dict mapping pattern names to DNA sequences.
                                 If None, patterns will be loaded later via load_pattern_sequences().
            device: Device for tensors
        """
        self.device = device
        self.csv_path = Path(csv_path)
        self.pattern_to_sequence = pattern_to_sequence or {}

        # DNA to token ID mapping: A=0, C=1, G=2, T=3
        self.nucleotide_to_id = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        # Load and index the CSV data
        self.df = pd.read_csv(csv_path)
        self._build_sequence_index()

    def _build_sequence_index(self):
        """Build index mapping sequence_name to list of pattern regions."""
        self.sequence_to_patterns: Dict[int, List[Dict]] = {}

        for _, row in self.df.iterrows():
            seq_name = int(row['sequence_name'])
            pattern_info = {
                'motif_name': row['motif_name'],
                'start': int(row['start']),
                'end': int(row['end']),
                'strand': row['strand'],
                'test_idx': int(row['test_idx'])
            }

            if seq_name not in self.sequence_to_patterns:
                self.sequence_to_patterns[seq_name] = []
            self.sequence_to_patterns[seq_name].append(pattern_info)

    def load_pattern_sequences(self, pattern_csv_path: Union[str, Path]):
        """
        Load pattern name to DNA sequence mapping from CSV.

        Expected CSV format:
            motif_name,sequence
            pos_patterns.pattern_0,ACGTACGT...
            pos_patterns.pattern_1,TGCATGCA...

        Args:
            pattern_csv_path: Path to CSV with pattern sequences
        """
        df = pd.read_csv(pattern_csv_path)
        self.pattern_to_sequence = dict(zip(df['motif_name'], df['sequence']))

    def sequence_to_ids(self, sequence: str) -> List[int]:
        """Convert DNA sequence string to list of token IDs."""
        return [self.nucleotide_to_id[nt.upper()] for nt in sequence]

    def get_inpainting_constraints(
        self,
        sequence_names: List[int]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Get inpainting constraints for a batch of sequences.

        Args:
            sequence_names: List of sequence indices (one per batch element)

        Returns:
            Tuple of (input_locs_batch, input_ids_batch) where each is a list
            of lists. Each inner list contains the positions/IDs for one sample.
        """
        input_locs_batch = []
        input_ids_batch = []

        for seq_name in sequence_names:
            locs = []
            ids = []

            if seq_name in self.sequence_to_patterns:
                for pattern_info in self.sequence_to_patterns[seq_name]:
                    motif_name = pattern_info['motif_name']
                    start = pattern_info['start']
                    end = pattern_info['end']
                    strand = pattern_info['strand']

                    # Get positions for this pattern
                    positions = list(range(start, end))

                    # Get token IDs if pattern sequence is available
                    if motif_name in self.pattern_to_sequence:
                        seq = self.pattern_to_sequence[motif_name]
                        # Handle strand orientation
                        if strand == '-':
                            seq = self._reverse_complement(seq)
                        # Truncate or pad sequence to match positions
                        seq = seq[:len(positions)]
                        token_ids = self.sequence_to_ids(seq)
                    else:
                        # Pattern sequence not loaded yet, use placeholder
                        token_ids = [0] * len(positions)  # Placeholder

                    locs.extend(positions)
                    ids.extend(token_ids)

            input_locs_batch.append(locs)
            input_ids_batch.append(ids)

        return input_locs_batch, input_ids_batch

    def _reverse_complement(self, seq: str) -> str:
        """Get reverse complement of DNA sequence."""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
                      'a': 't', 't': 'a', 'c': 'g', 'g': 'c'}
        return ''.join(complement.get(nt, nt) for nt in reversed(seq))

    def create_projection_function(
        self,
        sequence_names: List[int],
        batch_size: int
    ):
        """
        Create a projection function for use in sampling.

        Args:
            sequence_names: List of sequence indices for the batch
            batch_size: Batch size (should match len(sequence_names))

        Returns:
            Projection function that fixes constrained positions
        """
        input_locs_batch, input_ids_batch = self.get_inpainting_constraints(sequence_names)

        # Pre-compute tensors for each batch element
        locs_tensors = []
        ids_tensors = []

        for locs, ids in zip(input_locs_batch, input_ids_batch):
            if locs:
                locs_tensors.append(torch.tensor(locs, device=self.device, dtype=torch.long))
                ids_tensors.append(torch.tensor(ids, device=self.device, dtype=torch.long))
            else:
                locs_tensors.append(None)
                ids_tensors.append(None)

        def proj_fun(x):
            """Apply inpainting constraints to batch."""
            for batch_idx in range(min(batch_size, x.shape[0])):
                if locs_tensors[batch_idx] is not None:
                    x[batch_idx, locs_tensors[batch_idx]] = ids_tensors[batch_idx]
            return x

        return proj_fun

    def get_unique_sequence_names(self) -> List[int]:
        """Get all unique sequence names in the dataset."""
        return sorted(self.sequence_to_patterns.keys())

    def get_test_idx_mapping(self) -> Dict[int, int]:
        """Get mapping from sequence_name to test_idx."""
        mapping = {}
        for seq_name, patterns in self.sequence_to_patterns.items():
            if patterns:
                mapping[seq_name] = patterns[0]['test_idx']
        return mapping


def load_inpainting_data(
    csv_path: Union[str, Path],
    pattern_csv_path: Optional[Union[str, Path]] = None,
    device: str = "cuda"
) -> InpaintingManager:
    """
    Convenience function to load inpainting data.

    Args:
        csv_path: Path to pattern hits CSV
        pattern_csv_path: Optional path to pattern sequences CSV
        device: Device for tensors

    Returns:
        Configured InpaintingManager
    """
    manager = InpaintingManager(csv_path, device=device)
    if pattern_csv_path:
        manager.load_pattern_sequences(pattern_csv_path)
    return manager
