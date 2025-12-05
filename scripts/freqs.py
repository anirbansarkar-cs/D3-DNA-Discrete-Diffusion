#!/usr/bin/env python3
"""
Plot nucleotide and dinucleotide frequencies over sampling timesteps.

Analyzes sequences saved from sampling (shape: N, L, T, 4) and plots:
- Nucleotide frequencies (A, C, G, T) over time
- Dinucleotide frequencies (16 combinations) over time
Both with mean and standard deviation across samples.
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_sequences(h5_path):
    """Load sequences from H5 file."""
    with h5py.File(h5_path, 'r') as f:
        sequences = f['sequence'][:]  # (N, L, T, 4)
    print(f"Loaded sequences with shape: {sequences.shape}")
    print(f"  N={sequences.shape[0]} samples")
    print(f"  L={sequences.shape[1]} sequence length")
    print(f"  T={sequences.shape[2]} timesteps")
    return sequences


def compute_nucleotide_frequencies(sequences):
    """
    Compute nucleotide frequencies over time.

    Args:
        sequences: (N, L, T, 4) array

    Returns:
        mean_freqs: (T, 4) array of mean frequencies
        std_freqs: (T, 4) array of std frequencies
    """
    N, L, T, _ = sequences.shape

    # Sum across sequence length to get counts per sample per timestep
    counts = sequences.sum(axis=1)  # (N, T, 4)

    # Convert to frequencies (divide by sequence length)
    freqs = counts / L  # (N, T, 4)

    # Compute mean and std across samples
    mean_freqs = freqs.mean(axis=0)  # (T, 4)
    std_freqs = freqs.std(axis=0)  # (T, 4)

    return mean_freqs, std_freqs


def compute_dinucleotide_frequencies(sequences):
    """
    Compute dinucleotide frequencies over time.

    Args:
        sequences: (N, L, T, 4) array (one-hot encoded)

    Returns:
        mean_freqs: (T, 16) array of mean dinuc frequencies
        std_freqs: (T, 16) array of std dinuc frequencies
        dinuc_labels: list of 16 dinucleotide labels
    """
    N, L, T, _ = sequences.shape
    nucleotides = ['A', 'C', 'G', 'T']

    # Create all 16 dinucleotide labels
    dinuc_labels = [n1 + n2 for n1 in nucleotides for n2 in nucleotides]

    # Convert one-hot to indices: (N, L, T, 4) -> (N, L, T)
    seq_indices = np.argmax(sequences, axis=-1)  # (N, L, T)

    # Initialize dinucleotide counts
    dinuc_counts = np.zeros((N, T, 16))  # 16 dinucleotides

    # Count dinucleotides for each sample and timestep
    for t in range(T):
        for n in range(N):
            seq = seq_indices[n, :, t]  # (L,)
            # Get consecutive pairs
            for i in range(L - 1):
                first_nuc = seq[i]
                second_nuc = seq[i + 1]
                dinuc_idx = first_nuc * 4 + second_nuc
                dinuc_counts[n, t, dinuc_idx] += 1

    # Convert to frequencies (divide by number of dinucleotides: L-1)
    dinuc_freqs = dinuc_counts / (L - 1)  # (N, T, 16)

    # Compute mean and std across samples
    mean_freqs = dinuc_freqs.mean(axis=0)  # (T, 16)
    std_freqs = dinuc_freqs.std(axis=0)  # (T, 16)

    return mean_freqs, std_freqs, dinuc_labels


def plot_nucleotide_frequencies(mean_freqs, std_freqs, output_path):
    """Plot nucleotide frequencies over time."""
    T = mean_freqs.shape[0]
    timesteps = np.arange(T)
    nucleotides = ['A', 'C', 'G', 'T']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (nuc, color) in enumerate(zip(nucleotides, colors)):
        ax.plot(timesteps, mean_freqs[:, i], label=nuc, color=color, linewidth=2)
        ax.fill_between(timesteps,
                        mean_freqs[:, i] - std_freqs[:, i],
                        mean_freqs[:, i] + std_freqs[:, i],
                        alpha=0.2, color=color)

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Nucleotide Frequencies Over Sampling Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 0.5])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved nucleotide frequency plot to: {output_path}")
    plt.close()


def plot_dinucleotide_frequencies(mean_freqs, std_freqs, dinuc_labels, output_path):
    """Plot dinucleotide frequencies over time."""
    T = mean_freqs.shape[0]
    timesteps = np.arange(T)

    # Create a 4x4 subplot grid for the 16 dinucleotides
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle('Dinucleotide Frequencies Over Sampling Time', fontsize=16, fontweight='bold')

    for idx, (dinuc, ax) in enumerate(zip(dinuc_labels, axes.flat)):
        ax.plot(timesteps, mean_freqs[:, idx], linewidth=2, color='#1f77b4')
        ax.fill_between(timesteps,
                        mean_freqs[:, idx] - std_freqs[:, idx],
                        mean_freqs[:, idx] + std_freqs[:, idx],
                        alpha=0.3, color='#1f77b4')
        ax.set_title(dinuc, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim([0, max(0.15, mean_freqs[:, idx].max() * 1.1)])

        # Only show x-label on bottom row
        if idx >= 12:
            ax.set_xlabel('Timestep', fontsize=10)
        # Only show y-label on left column
        if idx % 4 == 0:
            ax.set_ylabel('Frequency', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved dinucleotide frequency plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot nucleotide and dinucleotide frequencies over time')
    parser.add_argument('--sequences', required=True, help='Path to H5 file with saved sequences')
    parser.add_argument('--output_dir', default=None, help='Output directory for plots (default: same as input file)')
    parser.add_argument('--prefix', default='', help='Prefix for output filenames')
    args = parser.parse_args()

    # Load sequences
    sequences = load_sequences(args.sequences)

    # Determine output directory
    if args.output_dir is None:
        output_dir = Path(args.sequences).parent
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add prefix if provided
    prefix = args.prefix + '_' if args.prefix else ''

    # Compute and plot nucleotide frequencies
    print("\nComputing nucleotide frequencies...")
    mean_nuc, std_nuc = compute_nucleotide_frequencies(sequences)
    nuc_output = output_dir / f"{prefix}nucleotide_frequencies.png"
    plot_nucleotide_frequencies(mean_nuc, std_nuc, nuc_output)

    # Compute and plot dinucleotide frequencies
    print("\nComputing dinucleotide frequencies...")
    mean_dinuc, std_dinuc, dinuc_labels = compute_dinucleotide_frequencies(sequences)
    dinuc_output = output_dir / f"{prefix}dinucleotide_frequencies.png"
    plot_dinucleotide_frequencies(mean_dinuc, std_dinuc, dinuc_labels, dinuc_output)

    print("\nDone!")


if __name__ == '__main__':
    main()
