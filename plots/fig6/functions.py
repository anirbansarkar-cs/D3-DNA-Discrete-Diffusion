"""
Figure 6B: Global sampling dynamics across time - plotting functions

Standalone functions for plotting different metrics during D3 reverse diffusion:
- Mutation rate
- Score magnitude (L2 norm)
- Predicted activities
- Attribution magnitude
- Motif proxy
- Motif scores
- Q_rev (reverse move probability)
- Staggered score
"""
from typing import Tuple, Optional
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.6,
    'axes.grid': False,
    'grid.linestyle': '-',
    'grid.color': '#dddddd',
    'legend.frameon': False,
    'axes.prop_cycle': mpl.cycler(color=['#E24A33','#348ABD','#988ED5','#777777','#FBC15E','#8EBA42','#FFB5B8']),
})


# ============================================================================
# Utility Functions
# ============================================================================

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Apply moving average smoothing to a 1D array."""
    if w <= 1:
        return x
    w = int(max(1, w))
    c = np.convolve(x, np.ones(w)/w, mode='same')
    return c


def compute_bands(per_sample: np.ndarray, q_lo: float = 0.1, q_hi: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (low, median, high) percentile tracks over samples for each step.
    
    Args:
        per_sample: (S, N) array where S=steps, N=samples
        q_lo: Lower quantile (default 0.1)
        q_hi: Upper quantile (default 0.9)
    
    Returns:
        Tuple of (low, median, high) arrays, each of shape (S,)
    """
    low = np.nanpercentile(per_sample, q_lo*100.0, axis=1)
    med = np.nanpercentile(per_sample, 50.0, axis=1)
    high = np.nanpercentile(per_sample, q_hi*100.0, axis=1)
    return low, med, high


def infer_stage_bounds(rate: np.ndarray, thr1: float = 0.6, thr2: float = 0.25) -> Tuple[int, int]:
    """Return (t1, t2) indices for stage boundaries given a smoothed rate.
    
    Args:
        rate: 1D array of smoothed rate values
        thr1: Stage1→2 threshold as fraction of max rate
        thr2: Stage2→3 threshold as fraction of max rate
    
    Returns:
        Tuple of (t1, t2) stage boundary indices
    """
    if rate.size < 3:
        return max(1, rate.size//3), max(2, 2*rate.size//3)
    m = np.nanmax(rate)
    if not np.isfinite(m) or m <= 0:
        S = rate.size
        return S//3, (2*S)//3
    t1 = next((i for i, v in enumerate(rate) if v <= thr1*m), None)
    t2 = next((i for i, v in enumerate(rate) if v <= thr2*m), None)
    S = rate.size
    if t1 is None or t2 is None or t1 >= t2-1:
        return S//3, (2*S)//3
    return int(t1), int(t2)


def enforce_saturation_stage(flips: np.ndarray, t1: int, t2: int) -> Tuple[int, int]:
    """Ensure stage 3 is the saturation (no flips) regime if present.
    
    Args:
        flips: (S, N) integer counts per step per sample
        t1, t2: Current stage boundaries
    
    Returns:
        Adjusted (t1, t2) boundaries
    """
    S = flips.shape[0]
    zero_all = np.all(flips == 0, axis=1)  # (S,)
    s0 = None
    if zero_all[-1]:
        j = S - 1
        while j >= 0 and zero_all[j]:
            j -= 1
        s0 = j + 1
    if s0 is not None and s0 > 0:
        t2_new = max(t1 + 1, min(t2, s0))
        t1_new = max(1, min(t1, t2_new - 1))
        return t1_new, t2_new
    return t1, t2


def draw_stage_lines(ax: plt.Axes, t1: int, t2: int):
    """Draw vertical dashed lines at stage boundaries."""
    for tx in (t1, t2):
        ax.axvline(tx, color='#333333', linestyle='--', linewidth=1.5, alpha=0.8)


def annotate_stage_labels(ax: plt.Axes, t1: int, t2: int, num_steps: int):
    """Place stage labels at midpoints."""
    mids = [(0 + t1)/2.0, (t1 + t2)/2.0, (t2 + (num_steps-1))/2.0]
    for i, m in enumerate(mids, start=1):
        ax.text(m, 0.98, f'Stage {i}', transform=ax.get_xaxis_transform(), 
                ha='center', va='top', fontsize=9, color='#333333')


def _plot_bands_and_traces(ax: plt.Axes, xs: np.ndarray, per_sample: np.ndarray, 
                           low: np.ndarray, med: np.ndarray, high: np.ndarray,
                           color: str, label: str, q_lo: float, q_hi: float,
                           plot_individual: bool = True, n_traces: int = 50, 
                           trace_alpha: float = 0.12):
    """Helper to plot percentile bands and individual traces."""
    ax.fill_between(xs, low, high, color=color, alpha=0.15, linewidth=0)
    ax.plot(xs, med, color=color, linewidth=2.2, 
            label=f"{label} (median, {int(q_lo*100)}–{int(q_hi*100)}%)")
    
    if plot_individual and per_sample is not None and per_sample.ndim == 2 and per_sample.shape[1] > 0:
        nplot = min(per_sample.shape[1], max(0, n_traces))
        if nplot > 0:
            pick = np.linspace(0, per_sample.shape[1]-1, nplot, dtype=int)
            ax.plot(xs, per_sample[:, pick], color=color, alpha=trace_alpha, linewidth=0.8)


# ============================================================================
# Data Loading Helper Functions
# ============================================================================

def load_mutation_rate(flips: np.ndarray, seq_len: int) -> np.ndarray:
    """Compute mutation rate per sample: (S, N)."""
    return flips / max(1, seq_len)


def meanpos_l2_per_sample(backward: h5py.File, chunk_n: int = 64) -> np.ndarray:
    """Compute per-sample mean per-position L2 across bases: (S,N)."""
    logits = backward['backward/logits']  # (S,N,L,4)
    S, N, L, _ = logits.shape
    out = np.zeros((S, N), dtype=np.float32)
    for t in range(S):
        for i in range(0, N, chunk_n):
            x = logits[t, i:i+chunk_n]          # (n,L,4)
            perpos = np.linalg.norm(x, axis=-1) # (n,L)  L2 over bases
            out[t, i:i+chunk_n] = perpos.mean(axis=-1)
    return out


def attr_per_sample(h5: h5py.File, key_channels: str = 'attr_channels', 
                    key_attr: str = 'attr', chunk_n: int = 64) -> np.ndarray:
    """Compute per-sample mean |attr| over positions/(channels): returns (S,N).
    Uses chunking + lazy h5 loading to avoid memory issues."""
    if key_channels in h5:
        d = h5[key_channels]
        S, N, L, C = d.shape
        out = np.zeros((S, N), dtype=np.float32)
        for t in range(S):
            for i in range(0, N, chunk_n):
                x = d[t, i:i+chunk_n]  # (n,L,4)  # this actually loads samples into memory
                vals = np.mean(np.abs(x), axis=(1,2))  # (n,)
                out[t, i:i+chunk_n] = vals
        return out
    elif key_attr in h5:
        d = h5[key_attr]
        S, N, L = d.shape
        out = np.zeros((S, N), dtype=np.float32)
        for t in range(S):
            for i in range(0, N, chunk_n):
                x = d[t, i:i+chunk_n]  # (n,L)
                vals = np.mean(np.abs(x), axis=1)  # (n,)
                out[t, i:i+chunk_n] = vals
        return out  # shape (timesteps, samples) = (S, N)
    else:
        raise KeyError('No attr_channels or attr dataset found')


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax along specified axis."""
    x = x - np.nanmax(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.clip(np.sum(ex, axis=axis, keepdims=True), 1e-12, None)


def motif_margin_per_sample(backward: h5py.File, max_n: int = 256, 
                            max_l: int = 128, chunk_n: int = 32) -> np.ndarray:
    """Per-sample motif proxy margin (top1 - top2 prob) averaged over subsampled positions.
    Returns (S,N_sel) with N_sel<=max_n.
    """
    logits = backward['backward/logits']  # (S,N,L,4)
    S, N, L, C = logits.shape
    sel_n = min(N, max_n)
    sel_l = min(L, max_l)
    n_idx = np.linspace(0, N-1, sel_n, dtype=int)
    l_idx = np.linspace(0, L-1, sel_l, dtype=int)
    out = np.zeros((S, sel_n), dtype=np.float32)
    for t in range(S):
        for bi in range(0, sel_n, chunk_n):
            idxn = n_idx[bi:bi+chunk_n]
            x = logits[t, idxn][:, l_idx]  # (n,l,4)
            p = softmax(x, axis=-1)
            p_sorted = np.sort(p, axis=-1)
            margin = p_sorted[..., -1] - p_sorted[..., -2]  # (n,l)
            out[t, bi:bi+chunk_n] = np.mean(margin, axis=1)
    return out


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_staggered_score(stag_per: np.ndarray, ax: Optional[plt.Axes] = None,
                        q_lo: float = 0.1, q_hi: float = 0.9,
                        color: str = '#2ca02c', t1: Optional[int] = None,
                        t2: Optional[int] = None, plot_individual: bool = True,
                        n_traces: int = 50, trace_alpha: float = 0.12) -> plt.Axes:
    """Plot staggered score over timesteps.

    Args:
        stag_per: (S, N) array of staggered score values per sample
        ax: Matplotlib axes (creates new if None)
        q_lo, q_hi: Quantiles for bands
        color: Plot color
        t1, t2: Stage boundaries (optional)
        plot_individual: Whether to overlay individual traces
        n_traces: Number of traces to overlay
        trace_alpha: Alpha for individual traces

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 3))

    low, med, high = compute_bands(stag_per, q_lo, q_hi)
    xs = np.arange(stag_per.shape[0])

    _plot_bands_and_traces(ax, xs, stag_per, low, med, high, color,
                           'stagger score', q_lo, q_hi, plot_individual,
                           n_traces, trace_alpha)

    if t1 is not None and t2 is not None:
        draw_stage_lines(ax, t1, t2)
        annotate_stage_labels(ax, t1, t2, len(xs))

    ax.set_ylabel('stagger score')
    ax.set_xlabel('timestep')
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.grid(False)

    return ax


def plot_dsigma(dsigma: np.ndarray, ax: Optional[plt.Axes] = None,
                q_lo: float = 0.1, q_hi: float = 0.9, color: str = 'tab:cyan',
                t1: Optional[int] = None, t2: Optional[int] = None,
                plot_individual: bool = False, n_traces: int = 50,
                trace_alpha: float = 0.12) -> plt.Axes:
    """Plot dsigma (step size) over timesteps.

    Args:
        dsigma: (S,) or (S, N) array of step sizes
        ax: Matplotlib axes (creates new if None)
        q_lo, q_hi: Quantiles for bands (only used if dsigma is (S, N))
        color: Plot color
        t1, t2: Stage boundaries (optional)
        plot_individual: Whether to overlay individual traces
        n_traces: Number of traces to overlay
        trace_alpha: Alpha for individual traces

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 3))

    xs = np.arange(dsigma.shape[0])

    if dsigma.ndim == 1:
        # Single trace
        ax.plot(xs, dsigma, color=color, linewidth=2.2, label='dsigma')
    else:
        # Multiple samples
        low, med, high = compute_bands(dsigma, q_lo, q_hi)
        _plot_bands_and_traces(ax, xs, dsigma, low, med, high, color,
                               'dsigma', q_lo, q_hi, plot_individual,
                               n_traces, trace_alpha)

    if t1 is not None and t2 is not None:
        draw_stage_lines(ax, t1, t2)
        annotate_stage_labels(ax, t1, t2, len(xs))

    ax.set_ylabel('dsigma (step size)')
    ax.set_xlabel('timestep')
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.grid(False)

    return ax


def plot_staggered_score_by_activity(stag_scores: np.ndarray, activities: np.ndarray,
                                     ax: Optional[plt.Axes] = None,
                                     t1: Optional[int] = None, t2: Optional[int] = None,
                                     log_scale: bool = True,
                                     percentile_groups: list = None) -> plt.Axes:
    """Plot log staggered scores grouped by activity percentiles.

    Args:
        stag_scores: (S, N) array of staggered score values per sample
        activities: (S, N) array of predicted activities per sample
        ax: Matplotlib axes (creates new if None)
        t1, t2: Stage boundaries (optional)
        log_scale: Apply log10(1+.) transformation to scores
        percentile_groups: List of tuples [(plo1, phi1, label1), ...] for grouping
                          Default: [(0, 33, 'low'), (33, 67, 'med'), (67, 100, 'high')]

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 3))

    if percentile_groups is None:
        percentile_groups = [(0, 33, 'low'), (33, 67, 'med'), (67, 100, 'high')]

    S, N = stag_scores.shape
    xs = np.arange(S)
    colors = ['#E24A33', '#988ED5', '#348ABD']  # low, med, high

    # Apply log transform if requested
    scores_to_plot = stag_scores.copy()
    if log_scale:
        scores_to_plot = np.log10(1.0 + np.clip(scores_to_plot, 0, None))

    for i, (plo, phi, label) in enumerate(percentile_groups):
        # For each timestep, group samples by their activity percentile
        group_means = np.zeros(S)

        for t in range(S):
            # Get activity percentiles at this timestep
            act_t = activities[t, :]
            p_lo = np.percentile(act_t, plo)
            p_hi = np.percentile(act_t, phi)

            # Select samples in this percentile range
            mask = (act_t >= p_lo) & (act_t <= p_hi)
            if np.any(mask):
                group_means[t] = np.nanmean(scores_to_plot[t, mask])
            else:
                group_means[t] = np.nan

        color = colors[i % len(colors)]
        ax.plot(xs, group_means, color=color, linewidth=2.2,
                label=f'{label} activity ({plo}-{phi}%)')

    if t1 is not None and t2 is not None:
        draw_stage_lines(ax, t1, t2)
        annotate_stage_labels(ax, t1, t2, len(xs))

    ylabel = 'log10(1+stagger score)' if log_scale else 'stagger score'
    ax.set_ylabel(f'{ylabel} by activity')
    ax.set_xlabel('timestep')
    ax.legend(frameon=False, fontsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.grid(False)

    return ax


def plot_qrev_stay_vs_move(qrev: np.ndarray, ax: Optional[plt.Axes] = None,
                          q_lo: float = 0.1, q_hi: float = 0.9,
                          t1: Optional[int] = None, t2: Optional[int] = None,
                          plot_individual: bool = False, n_traces: int = 50,
                          trace_alpha: float = 0.12) -> plt.Axes:
    """Plot Q_rev stay vs move probabilities over timesteps.

    Computes the average probability of staying in the same state (diagonal)
    vs moving to a different state (off-diagonal) from the reverse transition matrix.

    Args:
        qrev: (S, N, K, K) array of reverse transition matrices, or
              (S, N, 2) array with pre-computed [stay_prob, move_prob] per sample
        ax: Matplotlib axes (creates new if None)
        q_lo, q_hi: Quantiles for bands
        t1, t2: Stage boundaries (optional)
        plot_individual: Whether to overlay individual traces
        n_traces: Number of traces to overlay
        trace_alpha: Alpha for individual traces

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 3))

    if qrev.ndim == 4:
        # Compute stay vs move from full matrix
        S, N, K, _ = qrev.shape
        stay_probs = np.zeros((S, N))
        move_probs = np.zeros((S, N))

        for t in range(S):
            for n in range(N):
                # Diagonal elements are stay probabilities
                diag = np.diag(qrev[t, n])
                stay_probs[t, n] = np.mean(diag)

                # Off-diagonal elements are move probabilities
                off_diag = qrev[t, n].copy()
                np.fill_diagonal(off_diag, 0)
                move_probs[t, n] = np.mean(off_diag[off_diag != 0]) if np.any(off_diag != 0) else 0
    elif qrev.ndim == 3 and qrev.shape[2] == 2:
        # Pre-computed stay/move probabilities
        stay_probs = qrev[:, :, 0]
        move_probs = qrev[:, :, 1]
    else:
        raise ValueError(f"qrev must be (S, N, K, K) or (S, N, 2), got {qrev.shape}")

    xs = np.arange(stay_probs.shape[0])

    # Plot stay probabilities
    low_stay, med_stay, high_stay = compute_bands(stay_probs, q_lo, q_hi)
    _plot_bands_and_traces(ax, xs, stay_probs, low_stay, med_stay, high_stay,
                           '#348ABD', 'stay', q_lo, q_hi, plot_individual,
                           n_traces, trace_alpha)

    # Plot move probabilities
    low_move, med_move, high_move = compute_bands(move_probs, q_lo, q_hi)
    _plot_bands_and_traces(ax, xs, move_probs, low_move, med_move, high_move,
                           '#E24A33', 'move', q_lo, q_hi, plot_individual,
                           n_traces, trace_alpha)

    if t1 is not None and t2 is not None:
        draw_stage_lines(ax, t1, t2)
        annotate_stage_labels(ax, t1, t2, len(xs))

    ax.set_ylabel('Q_rev probability')
    ax.set_xlabel('timestep')
    ax.legend(frameon=False, fontsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.grid(False)

    return ax


def plot_mutation_rate_avg(mutation_rates: np.ndarray, ax: Optional[plt.Axes] = None,
                           q_lo: float = 0.1, q_hi: float = 0.9,
                           color: str = 'tab:olive', t1: Optional[int] = None,
                           t2: Optional[int] = None, plot_individual: bool = True,
                           n_traces: int = 50, trace_alpha: float = 0.12) -> plt.Axes:
    """Plot average mutation rate from the mutation matrix over time.

    For uniform graph, the theoretical rate is constant, but empirical rates may vary.

    Args:
        mutation_rates: (S, N) array of mutation rates per sample, or
                       (S, N, K, K) array of full rate matrices
        ax: Matplotlib axes (creates new if None)
        q_lo, q_hi: Quantiles for bands
        color: Plot color
        t1, t2: Stage boundaries (optional)
        plot_individual: Whether to overlay individual traces
        n_traces: Number of traces to overlay
        trace_alpha: Alpha for individual traces

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 3))

    if mutation_rates.ndim == 4:
        # Compute average off-diagonal rate from full matrix
        S, N, K, _ = mutation_rates.shape
        avg_rates = np.zeros((S, N))

        for t in range(S):
            for n in range(N):
                # Off-diagonal elements are mutation rates
                rate_matrix = mutation_rates[t, n]
                off_diag = rate_matrix.copy()
                np.fill_diagonal(off_diag, 0)
                avg_rates[t, n] = np.mean(np.abs(off_diag[off_diag != 0])) if np.any(off_diag != 0) else 0
    else:
        avg_rates = mutation_rates

    low, med, high = compute_bands(avg_rates, q_lo, q_hi)
    xs = np.arange(avg_rates.shape[0])

    _plot_bands_and_traces(ax, xs, avg_rates, low, med, high, color,
                           'mutation rate', q_lo, q_hi, plot_individual,
                           n_traces, trace_alpha)

    if t1 is not None and t2 is not None:
        draw_stage_lines(ax, t1, t2)
        annotate_stage_labels(ax, t1, t2, len(xs))

    ax.set_ylabel('avg mutation rate')
    ax.set_xlabel('timestep')
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.grid(False)

    return ax
