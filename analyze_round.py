#!/usr/bin/env python3
"""
Generic Round Analysis Script — Rounded Pipeline DPS Experiments.

Produces a comprehensive text analysis for any experimental round.
Uses the full 5-filter bio-plausibility from bio_plausibility.py
(GC, entropy, CpG, strand symmetry, homopolymer).

Sections:
  0.  Header
  1.  Seed pool baseline (oracle, GC, 5-filter BP, adapter balance)
  2.  Bio-plausibility landscape (Q1 — at oracle thresholds and bins)
  3.  Adapter balance by GC bin × oracle level (Q2)
  4.  Per-condition stats table (Q3 partial — top 30 by mean oracle and BP count)
  5.  Winning combinations (Q3 — top 5 by multiple metrics)
  6.  Factor-level marginal effects (w, eta, nf, act)
  7.  Round N+1 grid recommendation (Q4 — seed pool sizing)
  8.  Archive analysis (re-checked with 5-filter BP)
  9.  Entropy & GC control

Usage:
    python analyze_round.py --round_num 2
    python analyze_round.py --round_num 1 \\
        --output results/rounded_pipeline/round1_analysis.txt
    python analyze_round.py --round_num 2 \\
        --results_dir results/rounded_pipeline/round2_dps_push \\
        --seed_pool results/rounded_pipeline/seed_pool_round2/seeds.h5 \\
        --output results/rounded_pipeline/round2_analysis.txt
"""

import argparse
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import h5py
import numpy as np

# Full 5-filter bio-plausibility (must be in same directory or PYTHONPATH)
from bio_plausibility import is_bio_plausible as _is_bio_plausible_5filter
from bio_plausibility import _compute_entropy


# ── Constants ──────────────────────────────────────────────────────────────

ORACLE_RANGES = [
    ("<1.0",    None, 1.0),
    ("1.0-2.0", 1.0,  2.0),
    ("2.0-3.0", 2.0,  3.0),
    ("3.0-4.0", 3.0,  4.0),
    ("4.0-5.0", 4.0,  5.0),
    ("5.0-6.0", 5.0,  6.0),
    ("6.0-7.0", 6.0,  7.0),
    ("7.0-8.0", 7.0,  8.0),
    (">8.0",    8.0,  None),
]

ORACLE_THRESHOLDS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]

GC_BINS = [
    ("40-45%", 40.0, 45.0),
    ("45-50%", 45.0, 50.0),
    ("50-55%", 50.0, 55.0),
    ("55-60%", 55.0, 60.0),
    ("60-65%", 60.0, 65.0),
    ("65-70%", 65.0, 70.0),
    ("other",  None, None),
]

# 5-filter thresholds (matching bio_plausibility.py defaults)
BP_GC_RANGE      = (0.40, 0.70)  # fractions, not percentages — matches collect_round.py gc_bins
BP_ENTROPY_THRESH = 1.9
BP_CPG_MAX        = 0.072
BP_SYM_MAX        = 0.15
BP_HOMO_MAX       = 12

# Adapter sequences (checked at 5′ end, positions 0–14)
ADAPTER_FWD = "AGGACCGGATCAACT"
ADAPTER_RC  = "TCGGTTCACGCAATG"
_NUC        = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
ADAPTER_FWD_IDX = np.array([_NUC[c] for c in ADAPTER_FWD], dtype=np.int64)
ADAPTER_RC_IDX  = np.array([_NUC[c] for c in ADAPTER_RC],  dtype=np.int64)

W = 120  # output width


# ── Tee output ─────────────────────────────────────────────────────────────

class Tee:
    """Write to both a file and the real stdout simultaneously."""

    def __init__(self, filename):
        self._file   = open(filename, 'w')
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


# ── Adapter detection ──────────────────────────────────────────────────────

def detect_adapter(indices):
    """Classify sequences: 0=forward, 1=rc, 2=neither.

    Args:
        indices: (N, L) int array of base indices

    Returns:
        (N,) int8 array
    """
    N, L = indices.shape
    labels = np.full(N, 2, dtype=np.int8)
    flen = min(len(ADAPTER_FWD_IDX), L)
    rlen = min(len(ADAPTER_RC_IDX),  L)
    fwd_match = np.all(indices[:, :flen] == ADAPTER_FWD_IDX[:flen], axis=1)
    rc_match  = np.all(indices[:, :rlen] == ADAPTER_RC_IDX[:rlen],  axis=1)
    labels[fwd_match] = 0
    labels[rc_match]  = 1
    return labels


def adapter_str(n_fwd, n_rc, total):
    if total == 0:
        return "N/A"
    nei = total - n_fwd - n_rc
    return (f"fwd={n_fwd}({100*n_fwd/total:.1f}%) "
            f"rc={n_rc}({100*n_rc/total:.1f}%) "
            f"neither={nei}({100*nei/total:.1f}%)")


# ── Formatting ─────────────────────────────────────────────────────────────

def section_header(title):
    print("\n" + "=" * W)
    print(f"  {title}")
    print("=" * W)


def subheader(title):
    print(f"\n--- {title} ---")


def fmt_table(rows, headers, widths=None):
    """Print fixed-width table, truncated to W chars."""
    if not rows:
        print("  (no data)")
        return
    if widths is None:
        widths = [
            max(len(str(h)), max(len(str(r[i])) for r in rows), 0) + 2
            for i, h in enumerate(headers)
        ]
    hdr = " ".join(str(h).rjust(w) for h, w in zip(headers, widths))
    sep = " ".join("-" * w for w in widths)
    print(hdr[:W])
    print(sep[:W])
    for row in rows:
        parts = []
        for v, w in zip(row, widths):
            if isinstance(v, float):
                s = f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}"
            else:
                s = str(v)
            parts.append(s.rjust(w))
        print(" ".join(parts)[:W])


# ── Bio-plausibility wrapper ───────────────────────────────────────────────

def apply_5filter_bp(indices, gc_frac_unit=None, entropy=None):
    """Apply full 5-filter bio-plausibility check.

    Args:
        indices:       (N, L) int array of base indices {0,1,2,3}
        gc_frac_unit:  (N,) GC fractions in [0,1], or None to compute
        entropy:       (N,) Shannon entropy, or None to compute

    Returns:
        mask:      (N,) bool
        breakdown: dict with per-filter pass counts
    """
    return _is_bio_plausible_5filter(
        indices,
        gc_fractions=gc_frac_unit,
        entropy=entropy,
        gc_range=BP_GC_RANGE,
        entropy_thresh=BP_ENTROPY_THRESH,
        cpg_max=BP_CPG_MAX,
        strand_sym_max=BP_SYM_MAX,
        max_homo_run=BP_HOMO_MAX,
    )


# ── Filename parsing ───────────────────────────────────────────────────────

def parse_filename(fname):
    m = re.match(
        r"samples_w([\d.]+)_eta([\d.]+)_nf([\d.]+)_act([\d.]+)\.h5",
        fname,
    )
    if not m:
        return None
    return {
        "w":   float(m.group(1)),
        "eta": float(m.group(2)),
        "nf":  float(m.group(3)),
        "act": float(m.group(4)),
    }


# ── Mask helpers ───────────────────────────────────────────────────────────

def oracle_mask(oracle, lo, hi):
    mask = np.ones(len(oracle), dtype=bool)
    if lo is not None:
        mask &= oracle >= lo
    if hi is not None:
        mask &= oracle < hi
    return mask


def gc_bin_mask(gc_pct, lo, hi):
    if lo is None and hi is None:
        # "other": not in any defined bin
        defined = np.zeros(len(gc_pct), dtype=bool)
        for _, glo, ghi in GC_BINS[:-1]:
            defined |= (gc_pct >= glo) & (gc_pct < ghi)
        return ~defined
    mask = np.ones(len(gc_pct), dtype=bool)
    if lo is not None:
        mask &= gc_pct >= lo
    if hi is not None:
        mask &= gc_pct < hi
    return mask


# ── Data loading ───────────────────────────────────────────────────────────

def load_seed_pool(seed_path):
    """Load seed pool h5. Returns dict with oracle, gc_pct, entropy, bp_mask, etc."""
    with h5py.File(seed_path, "r") as f:
        seqs     = f["arr_0"][:]
        oracle   = f["oracle_preds"][()].flatten()
        gc_frac  = f["gc_fractions"][()].flatten()

    # Normalise to fractions and percentages
    gc_frac_unit = gc_frac if gc_frac.max() <= 1.0 else gc_frac / 100.0
    gc_pct       = gc_frac_unit * 100.0

    indices = seqs.argmax(axis=1)          # (N, 4, L) → (N, L)
    entropy = _compute_entropy(indices)

    bp_mask, bp_bd = apply_5filter_bp(indices, gc_frac_unit, entropy)
    adapter = detect_adapter(indices)

    return {
        "n":            len(oracle),
        "oracle":       oracle,
        "gc_pct":       gc_pct,
        "entropy":      entropy,
        "bp_mask":      bp_mask,
        "bp_breakdown": bp_bd,
        "adapter":      adapter,     # 0=fwd, 1=rc, 2=neither
    }


def load_all_conditions(results_dir, verbose=True):
    """Load all sample h5 files from act*/ subdirs.

    Memory-efficient: per-condition summary stats stored in dicts; global
    per-sequence scalars (oracle, gc_pct, bp_mask, adapter) accumulated as
    lightweight arrays (~135 MB for 13.5 M sequences).

    Returns:
        conditions: list of per-condition stat dicts
        global_agg: dict with concatenated global arrays + global breakdown
    """
    results_dir = Path(results_dir)

    h5_files = []
    for act_dir in sorted(results_dir.iterdir()):
        if not act_dir.is_dir():
            continue
        for h5p in sorted(act_dir.glob("samples_*.h5")):
            h5_files.append(h5p)

    if verbose:
        print(f"  Found {len(h5_files)} sample files")

    conditions = []

    # Per-sequence accumulators (lightweight scalars / bools only)
    g_oracle  = []
    g_gc_pct  = []
    g_bp_mask = []
    g_adapter = []

    # Global per-filter breakdown
    g_total   = 0
    g_breakdown = defaultdict(int)

    for i, h5p in enumerate(h5_files):
        if verbose and i % 100 == 0:
            print(f"  [{i}/{len(h5_files)}] loading...", end="\r", flush=True)

        params = parse_filename(h5p.name)
        if params is None:
            print(f"\n  WARNING: could not parse {h5p.name}", file=sys.stderr)
            continue

        with h5py.File(h5p, "r") as f:
            seqs    = f["arr_0"][:]
            oracle  = f["oracle_preds"][()].flatten()
            gc_frac = f["gc_fractions"][()].flatten()

        gc_frac_unit = gc_frac if gc_frac.max() <= 1.0 else gc_frac / 100.0
        gc_pct       = gc_frac_unit * 100.0
        indices      = seqs.argmax(axis=1)   # (N, L)
        entropy      = _compute_entropy(indices)

        bp_mask, bp_bd = apply_5filter_bp(indices, gc_frac_unit, entropy)
        adapter = detect_adapter(indices)

        N      = len(oracle)
        n_bp   = int(bp_mask.sum())
        n_fwd  = int((adapter == 0).sum())
        n_rc   = int((adapter == 1).sum())

        bp_oracle      = oracle[bp_mask]
        bp_mean_oracle = float(np.mean(bp_oracle)) if n_bp > 0 else 0.0
        unique_count   = len({row.tobytes() for row in indices})

        cond = {
            **params,
            "path":           str(h5p),
            "n_total":        N,
            "n_unique":       unique_count,
            "mean_oracle":    float(np.mean(oracle)),
            "median_oracle":  float(np.median(oracle)),
            "max_oracle":     float(np.max(oracle)),
            "std_oracle":     float(np.std(oracle)),
            "mean_gc":        float(np.mean(gc_pct)),
            "mean_entropy":   float(np.mean(entropy)),
            "n_bp":           n_bp,
            "bp_pct":         100.0 * n_bp / N,
            "bp_mean_oracle": bp_mean_oracle,
            "gc_pass_pct":    100.0 * bp_bd["gc_pass"]          / N,
            "ent_pass_pct":   100.0 * bp_bd["entropy_pass"]     / N,
            "cpg_pass_pct":   100.0 * bp_bd["cpg_pass"]         / N,
            "sym_pass_pct":   100.0 * bp_bd["strand_sym_pass"]  / N,
            "homo_pass_pct":  100.0 * bp_bd["homopolymer_pass"] / N,
            "n_fwd":          n_fwd,
            "n_rc":           n_rc,
            "fwd_pct":        100.0 * n_fwd / N,
            "pct_above2":     float(100.0 * np.mean(oracle > 2)),
            "pct_above3":     float(100.0 * np.mean(oracle > 3)),
            "pct_above4":     float(100.0 * np.mean(oracle > 4)),
        }
        conditions.append(cond)

        # Global accumulation
        g_oracle.append(oracle.astype(np.float32))
        g_gc_pct.append(gc_pct.astype(np.float32))
        g_bp_mask.append(bp_mask)
        g_adapter.append(adapter)
        g_total += N
        for k, v in bp_bd.items():
            if k != "total":
                g_breakdown[k] += int(v)

    if verbose:
        print(f"\n  Loaded {len(conditions)} conditions, {g_total:,} sequences")

    global_agg = {
        "oracle":    np.concatenate(g_oracle),
        "gc_pct":    np.concatenate(g_gc_pct),
        "bp_mask":   np.concatenate(g_bp_mask),
        "adapter":   np.concatenate(g_adapter),
        "total":     g_total,
        "breakdown": dict(g_breakdown),
    }
    return conditions, global_agg


def load_archives(results_dir):
    """Load per-act archive h5 files. Returns dict keyed by (act_dir_name, archive_type)."""
    results_dir = Path(results_dir)
    archives = {}
    names = [
        "archive_top1000.h5",
        "archive_best_bio_plausible.h5",
        "archive_best_above1.0.h5",
    ]
    for act_dir in sorted(results_dir.iterdir()):
        if not act_dir.is_dir():
            continue
        for aname in names:
            path = act_dir / aname
            if not path.exists():
                continue
            with h5py.File(path, "r") as f:
                seqs    = f["arr_0"][:]
                oracle  = f["oracle_preds"][()].flatten()
                gc_frac = f["gc_fractions"][()].flatten()
            gc_frac_unit = gc_frac if gc_frac.max() <= 1.0 else gc_frac / 100.0
            gc_pct       = gc_frac_unit * 100.0
            indices      = seqs.argmax(axis=1)
            entropy      = _compute_entropy(indices)
            bp_mask, bp_bd = apply_5filter_bp(indices, gc_frac_unit, entropy)
            adapter = detect_adapter(indices)
            key = aname.replace(".h5", "")
            archives[(act_dir.name, key)] = {
                "oracle":      oracle,
                "gc_pct":      gc_pct,
                "bp_mask":     bp_mask,
                "bp_breakdown": bp_bd,
                "adapter":     adapter,
                "n":           len(oracle),
            }
    return archives


# ── Analysis sections ──────────────────────────────────────────────────────

def section0_header(round_num, results_dir, seed_pool_path, n_conditions, n_total):
    section_header(f"ROUND {round_num} ANALYSIS — ROUNDED PIPELINE DPS EXPERIMENTS")
    print(f"  Round:            {round_num}")
    print(f"  Results dir:      {results_dir}")
    print(f"  Seed pool:        {seed_pool_path}")
    print(f"  Conditions found: {n_conditions:,}")
    print(f"  Total sequences:  {n_total:,}")
    print(f"  Date:             {date.today().isoformat()}")
    print(f"  Bio-plausibility: 5 filters: GC [40-70%], entropy>1.9, "
          f"CpG<=7.2%, strand_sym<0.15, homopolymer<=12bp")


def section1_seed_baseline(seed):
    section_header("SECTION 1: SEED POOL BASELINE")

    oracle  = seed["oracle"]
    gc_pct  = seed["gc_pct"]
    entropy = seed["entropy"]
    bp_mask = seed["bp_mask"]
    bp_bd   = seed["bp_breakdown"]
    adapter = seed["adapter"]
    n       = seed["n"]

    n_fwd = int((adapter == 0).sum())
    n_rc  = int((adapter == 1).sum())

    print(f"\n  Pool: {n:,} sequences")
    print(f"  Oracle:  mean={np.mean(oracle):.4f}  median={np.median(oracle):.4f}  "
          f"max={np.max(oracle):.4f}  min={np.min(oracle):.4f}  std={np.std(oracle):.4f}")
    print(f"  GC%:     mean={np.mean(gc_pct):.2f}%  std={np.std(gc_pct):.2f}%  "
          f"in [40-60%]: {int(((gc_pct>=40)&(gc_pct<=60)).sum())}/{n} "
          f"({100*np.mean((gc_pct>=40)&(gc_pct<=60)):.1f}%)")
    print(f"  Entropy: mean={np.mean(entropy):.4f}  std={np.std(entropy):.4f}")
    print(f"  Adapter: {adapter_str(n_fwd, n_rc, n)}")

    subheader("5-Filter Bio-Plausibility")
    print(f"  {'Filter':<30} {'Pass':>8} {'Rate':>8}")
    print(f"  {'-'*48}")
    for key, label in [
        ("gc_pass",           "GC [40-70%]"),
        ("entropy_pass",      "Entropy > 1.9"),
        ("cpg_pass",          "CpG <= 7.2%"),
        ("strand_sym_pass",   "Strand symmetry < 0.15"),
        ("homopolymer_pass",  "Homopolymer <= 12bp"),
        ("all_pass",          "ALL PASS"),
    ]:
        cnt = bp_bd[key]
        print(f"  {label:<30} {cnt:>8,} {100*cnt/n:>7.1f}%")

    subheader("Oracle distribution (with adapter and BP split)")
    headers = ["Range",  "Count", "Pct%", "MeanOr", "Fwd", "RC", "Neither", "Fwd%", "BP"]
    widths  = [10,        8,       7,      9,         7,    6,    8,          7,      7]
    rows = []
    for label, lo, hi in ORACLE_RANGES:
        m   = oracle_mask(oracle, lo, hi)
        cnt = int(m.sum())
        if cnt > 0:
            a_sub = adapter[m]
            fwd_c = int((a_sub == 0).sum())
            rc_c  = int((a_sub == 1).sum())
            nei_c = cnt - fwd_c - rc_c
            bp_c  = int(bp_mask[m].sum())
            rows.append([label, cnt, f"{100*cnt/n:.1f}",
                         float(np.mean(oracle[m])),
                         fwd_c, rc_c, nei_c, f"{100*fwd_c/cnt:.1f}", bp_c])
        else:
            rows.append([label, 0, "0.0", 0.0, 0, 0, 0, "0.0", 0])
    fmt_table(rows, headers, widths)


def section2_bio_landscape(g, seed):
    """Section 2: Bio-Plausibility Landscape (Q1)."""
    section_header("SECTION 2: BIO-PLAUSIBILITY LANDSCAPE")

    oracle  = g["oracle"]
    bp_mask = g["bp_mask"]
    adapter = g["adapter"]
    bd      = g["breakdown"]
    N       = g["total"]

    subheader("Per-filter pass rates — all generated sequences")
    print(f"  Total sequences: {N:,}")
    print(f"  {'Filter':<30} {'Pass':>12} {'Rate':>8}")
    print(f"  {'-'*52}")
    for key, label in [
        ("gc_pass",           "GC [40-70%]"),
        ("entropy_pass",      "Entropy > 1.9"),
        ("cpg_pass",          "CpG <= 7.2%"),
        ("strand_sym_pass",   "Strand symmetry < 0.15"),
        ("homopolymer_pass",  "Homopolymer <= 12bp"),
        ("all_pass",          "ALL PASS"),
    ]:
        cnt = bd[key]
        print(f"  {label:<30} {cnt:>12,} {100*cnt/N:>7.1f}%")

    seed_bp = int(seed["bp_mask"].sum())
    seed_n  = seed["n"]
    print(f"\n  Seed BP pass rate:      {seed_bp:,}/{seed_n:,} ({100*seed_bp/seed_n:.1f}%)")
    print(f"  Generated BP pass rate: {bd['all_pass']:,}/{N:,} ({100*bd['all_pass']/N:.1f}%)")

    subheader("Bio-plausible counts at oracle thresholds (cumulative >=)")
    headers = ["oracle>=", "total_seqs", "bp_count", "bp_pct%",
               "fwd_count", "rc_count", "fwd_pct%"]
    widths  = [9,           11,           10,         9,
               11,           10,           9]
    rows = []
    for thresh in ORACLE_THRESHOLDS:
        above   = oracle >= thresh
        total   = int(above.sum())
        bp_a    = above & bp_mask
        bp_cnt  = int(bp_a.sum())
        fwd_cnt = int((bp_a & (adapter == 0)).sum())
        rc_cnt  = int((bp_a & (adapter == 1)).sum())
        bp_pct  = f"{100*bp_cnt/total:.1f}"  if total  > 0 else "0.0"
        fwd_pct = f"{100*fwd_cnt/bp_cnt:.1f}" if bp_cnt > 0 else "0.0"
        rows.append([f">={thresh:.1f}", total, bp_cnt, bp_pct, fwd_cnt, rc_cnt, fwd_pct])
    fmt_table(rows, headers, widths)

    subheader("Bio-plausible per oracle bin (with adapter split)")
    headers2 = ["Range",  "Total",  "BP",    "BP%",   "Fwd_BP", "RC_BP", "Fwd%_BP", "MeanOr"]
    widths2  = [10,        10,       9,       7,       9,        9,       10,         9]
    rows2 = []
    for label, lo, hi in ORACLE_RANGES:
        m      = oracle_mask(oracle, lo, hi)
        total  = int(m.sum())
        bp_m   = m & bp_mask
        bp     = int(bp_m.sum())
        fwd_bp = int((bp_m & (adapter == 0)).sum())
        rc_bp  = int((bp_m & (adapter == 1)).sum())
        bp_pct = f"{100*bp/total:.1f}"  if total > 0 else "0.0"
        fwd_bp_pct = f"{100*fwd_bp/bp:.1f}" if bp > 0 else "0.0"
        mean_or = float(np.mean(oracle[m])) if total > 0 else 0.0
        rows2.append([label, total, bp, bp_pct, fwd_bp, rc_bp, fwd_bp_pct, mean_or])
    fmt_table(rows2, headers2, widths2)

    seed_max = float(np.max(seed["oracle"]))
    above_sm = oracle > seed_max
    n_above     = int(above_sm.sum())
    n_above_bp  = int((above_sm & bp_mask).sum())
    print(f"\n  Seed max oracle: {seed_max:.4f}")
    print(f"  Sequences above seed max: {n_above:,} ({100*n_above/N:.2f}%)")
    print(f"  Bio-plausible above seed max: {n_above_bp:,}")
    if n_above > 0:
        print(f"  Max oracle achieved: {float(np.max(oracle)):.4f}")


def section3_adapter_gc_oracle(g):
    """Section 3: Adapter Balance by GC Bin × Oracle Level (Q2)."""
    section_header("SECTION 3: ADAPTER BALANCE BY GC BIN × ORACLE LEVEL")

    oracle  = g["oracle"]
    gc_pct  = g["gc_pct"]
    bp_mask = g["bp_mask"]
    adapter = g["adapter"]
    N       = g["total"]

    def gc_oracle_table(title, seq_mask):
        subheader(title)
        headers = ["GC Bin",  "OracleRange", "Total", "BP",  "Fwd",  "RC",  "Fwd%",  "MeanOr"]
        widths  = [8,          12,             8,       7,     7,      7,     6,        8]
        rows = []
        for gc_label, gc_lo, gc_hi in GC_BINS:
            gc_m = gc_bin_mask(gc_pct, gc_lo, gc_hi) & seq_mask
            for or_label, or_lo, or_hi in ORACLE_RANGES:
                cell = gc_m & oracle_mask(oracle, or_lo, or_hi)
                total = int(cell.sum())
                if total == 0:
                    continue
                bp  = int((cell & bp_mask).sum())
                fwd = int((cell & (adapter == 0)).sum())
                rc  = int((cell & (adapter == 1)).sum())
                mean_or = float(np.mean(oracle[cell]))
                rows.append([gc_label, or_label, total, bp, fwd, rc,
                             f"{100*fwd/total:.1f}", mean_or])
        fmt_table(rows, headers, widths)

    gc_oracle_table("All sequences: GC bin × oracle range",
                    np.ones(N, dtype=bool))
    gc_oracle_table("Bio-plausible only: GC bin × oracle range",
                    bp_mask)

    subheader("GC bin marginals (all sequences)")
    headers = ["GC Bin", "Total", "Pct%", "BP",  "BP%",  "Fwd%", "MeanOr", "MaxOr"]
    widths  = [8,         9,       6,      9,     6,      6,      8,        8]
    rows = []
    for gc_label, gc_lo, gc_hi in GC_BINS:
        m = gc_bin_mask(gc_pct, gc_lo, gc_hi)
        total = int(m.sum())
        if total == 0:
            continue
        bp  = int((m & bp_mask).sum())
        fwd = int((m & (adapter == 0)).sum())
        rows.append([gc_label, total, f"{100*total/N:.1f}", bp,
                     f"{100*bp/total:.1f}", f"{100*fwd/total:.1f}",
                     float(np.mean(oracle[m])), float(np.max(oracle[m]))])
    fmt_table(rows, headers, widths)


def section4_condition_table(conditions):
    """Section 4: Per-Condition Stats Table (Q3 partial)."""
    section_header("SECTION 4: PER-CONDITION STATS TABLE")

    headers = ["w",  "eta", "nf",  "act",
               "total", "unique",
               "mean_or", "med_or", "max_or", "GC%",
               "bp_cnt", "bp%",
               "gc%", "ent%", "cpg%", "sym%", "homo%",
               "fwd%", ">2%", ">3%", ">4%"]
    widths  = [5,    7,     6,     5,
               7,      7,
               8,        8,       8,       6,
               7,       6,
               5,    5,     5,     5,      6,
               6,     5,    5,    5]

    def print_top30(conds, sort_key, label):
        subheader(f"Top 30 by {label}")
        sorted_c = sorted(conds, key=lambda c: c[sort_key], reverse=True)[:30]
        rows = []
        for c in sorted_c:
            rows.append([
                c["w"], c["eta"], c["nf"], c["act"],
                c["n_total"], c["n_unique"],
                c["mean_oracle"], c["median_oracle"], c["max_oracle"],
                f"{c['mean_gc']:.1f}",
                c["n_bp"], f"{c['bp_pct']:.1f}",
                f"{c['gc_pass_pct']:.0f}",  f"{c['ent_pass_pct']:.0f}",
                f"{c['cpg_pass_pct']:.0f}",  f"{c['sym_pass_pct']:.0f}",
                f"{c['homo_pass_pct']:.0f}",
                f"{c['fwd_pct']:.1f}",
                f"{c['pct_above2']:.1f}", f"{c['pct_above3']:.1f}", f"{c['pct_above4']:.1f}",
            ])
        fmt_table(rows, headers, widths)

    print_top30(conditions, "mean_oracle",  "mean oracle")
    print_top30(conditions, "n_bp",         "bio-plausible count")


def section5_winning_combos(conditions):
    """Section 5: Winning Combinations (Q3)."""
    section_header("SECTION 5: WINNING COMBINATIONS")

    headers = ["#", "w",  "eta", "nf",  "act",
               "mean_or", "max_or", "bp_cnt",
               "bp%", "fwd%", ">2%", ">3%", ">4%"]
    widths  = [3,   5,    7,     6,     5,
               8,        8,       7,
               5,     5,     5,    5,    5]

    metrics = [
        ("Mean Oracle",                "mean_oracle",    True),
        ("Max Oracle",                 "max_oracle",     True),
        ("Bio-Plausible Count",        "n_bp",           True),
        ("Bio-Plausible Mean Oracle",  "bp_mean_oracle", True),
        ("Bio-Plausible Fraction %",   "bp_pct",         True),
        ("Adapter Balance (|fwd%-50| smallest)", None,   None),  # ascending special
    ]

    for label, key, descending in metrics:
        subheader(f"Top 5 by {label}")
        if key is not None:
            sorted_c = sorted(conditions, key=lambda c: c[key], reverse=descending)
        else:
            sorted_c = sorted(conditions, key=lambda c: abs(c["fwd_pct"] - 50))
        rows = []
        for i, c in enumerate(sorted_c[:5], 1):
            rows.append([
                i, c["w"], c["eta"], c["nf"], c["act"],
                c["mean_oracle"], c["max_oracle"], c["n_bp"],
                f"{c['bp_pct']:.1f}", f"{c['fwd_pct']:.1f}",
                f"{c['pct_above2']:.1f}", f"{c['pct_above3']:.1f}", f"{c['pct_above4']:.1f}",
            ])
        fmt_table(rows, headers, widths)


def section6_factor_effects(conditions):
    """Section 6: Factor-Level Marginal Effects."""
    section_header("SECTION 6: FACTOR-LEVEL MARGINAL EFFECTS")

    effect_sizes = {}

    for factor in ["w", "eta", "nf", "act"]:
        g_mean  = defaultdict(list)
        g_med   = defaultdict(list)
        g_bp    = defaultdict(list)
        g_bpfrc = defaultdict(list)
        g_fwd   = defaultdict(list)

        for c in conditions:
            v = c[factor]
            g_mean[v].append(c["mean_oracle"])
            g_med[v].append(c["median_oracle"])
            g_bp[v].append(c["n_bp"])
            g_bpfrc[v].append(c["bp_pct"])
            g_fwd[v].append(c["fwd_pct"])

        subheader(f"Marginal effects: {factor}")
        headers = ["Value", "MeanOr", "StdOr", "MedOr",
                   "Mean_BP", "BPFrac%", "MeanFwd%", "N"]
        widths  = [8,       8,        7,       7,
                   9,         9,         9,         5]
        rows = []
        for val in sorted(g_mean.keys()):
            rows.append([
                val,
                np.mean(g_mean[val]),  np.std(g_mean[val]),  np.mean(g_med[val]),
                np.mean(g_bp[val]),    np.mean(g_bpfrc[val]), np.mean(g_fwd[val]),
                len(g_mean[val]),
            ])
        fmt_table(rows, headers, widths)

        means     = [np.mean(v) for v in g_mean.values()]
        effect    = max(means) - min(means)
        effect_sizes[factor] = effect
        sorted_items = sorted(g_mean.items(), key=lambda x: np.mean(x[1]), reverse=True)
        trend = " > ".join(f"{k}({np.mean(v):.4f})" for k, v in sorted_items)
        print(f"  Effect size (max-min mean oracle): {effect:.4f}")
        print(f"  Trend: {trend}")

    subheader("Effect size ranking (most → least influential)")
    ranked = sorted(effect_sizes.items(), key=lambda x: x[1], reverse=True)
    headers2 = ["Factor", "EffectSize"]
    widths2  = [8,         12]
    fmt_table([[f, f"{e:.4f}"] for f, e in ranked], headers2, widths2)


def section7_round3_recommendation(conditions, g, round_num=None):
    """Section 7: Round N+1 Grid Recommendation (Q4)."""
    next_round = (round_num + 1) if round_num else "N+1"
    section_header(f"SECTION 7: ROUND {next_round} GRID RECOMMENDATION")

    oracle  = g["oracle"]
    bp_mask = g["bp_mask"]
    adapter = g["adapter"]
    N       = g["total"]
    n_conds = len(conditions)

    # Condition quality histogram
    subheader("Condition quality distribution by bio-plausible count")
    bins = [(0, 1), (1, 10), (10, 50), (50, 200), (200, 500), (500, 10**9)]
    labels_b = ["=0", "1-9", "10-49", "50-199", "200-499", "500+"]
    print(f"  {'BP Count Bin':<15} {'Conditions':>12} {'Pct%':>7}")
    print(f"  {'-'*35}")
    for (lo, hi), lab in zip(bins, labels_b):
        cnt = sum(1 for c in conditions if lo <= c["n_bp"] < hi)
        print(f"  {lab:<15} {cnt:>12} {100*cnt/n_conds:>7.1f}%")

    # Worth-keeping threshold
    subheader("Conditions surviving bio-plausible count cutoffs")
    headers = ["BP Cutoff", "N Surviving", "Pct%"]
    widths  = [10,           12,            7]
    rows = []
    for cutoff in [1, 10, 50, 100, 200, 500]:
        surv = sum(1 for c in conditions if c["n_bp"] >= cutoff)
        rows.append([f">={cutoff}", surv, f"{100*surv/n_conds:.1f}"])
    fmt_table(rows, headers, widths)

    # Cross-tabulate by factor (cutoff = 50)
    CUTOFF = 50
    surviving = [c for c in conditions if c["n_bp"] >= CUTOFF]
    subheader(f"Surviving conditions (BP >= {CUTOFF}) cross-tabulated by factor value")
    for factor in ["w", "eta", "nf", "act"]:
        val_total   = defaultdict(int)
        val_survive = defaultdict(int)
        for c in conditions:
            val_total[c[factor]] += 1
        for c in surviving:
            val_survive[c[factor]] += 1
        parts = []
        for val in sorted(val_total.keys()):
            tot  = val_total[val]
            surv = val_survive.get(val, 0)
            parts.append(f"{val}:{surv}/{tot}({100*surv/tot:.0f}%)")
        print(f"  {factor}: {' | '.join(parts)}")

    # Seed pool sizing at oracle thresholds
    subheader("Seed pool sizing: bio-plausible sequences at oracle thresholds")
    headers3 = ["oracle>=", "AllBP", "FwdBP", "RCBP", "Fwd%BP"]
    widths3  = [10,          10,      8,       8,      8]
    rows3 = []
    for thresh in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]:
        m        = (oracle >= thresh) & bp_mask
        total_bp = int(m.sum())
        fwd_bp   = int((m & (adapter == 0)).sum())
        rc_bp    = int((m & (adapter == 1)).sum())
        fwd_pct_s = f"{100*fwd_bp/total_bp:.1f}" if total_bp > 0 else "0.0"
        rows3.append([f">={thresh:.1f}", total_bp, fwd_bp, rc_bp, fwd_pct_s])
    fmt_table(rows3, headers3, widths3)

    # Recommendation block
    subheader("RECOMMENDATION")
    bp_at_2   = int(((oracle >= 2.0) & bp_mask).sum())
    bp_at_2_5 = int(((oracle >= 2.5) & bp_mask).sum())
    bp_at_3   = int(((oracle >= 3.0) & bp_mask).sum())
    print(f"  Bio-plausible >= 2.0: {bp_at_2:,} ({100*bp_at_2/N:.2f}% of all, "
          f"{bp_at_2/n_conds:.1f} per condition)")
    print(f"  Bio-plausible >= 2.5: {bp_at_2_5:,} ({100*bp_at_2_5/N:.3f}% of all, "
          f"{bp_at_2_5/n_conds:.1f} per condition)")
    print(f"  Bio-plausible >= 3.0: {bp_at_3:,} ({100*bp_at_3/N:.3f}% of all, "
          f"{bp_at_3/n_conds:.1f} per condition)")

    for target in [5_000, 10_000, 20_000, 50_000]:
        if bp_at_2 > 0:
            top_k = max(1, int(np.ceil(target / n_conds)))
            avail = min(bp_at_2, target)
            print(f"  For seed pool = {target:,} (oracle>=2.0 BP): "
                  f"top_k ~{top_k} per cond (available: {avail:,} / {bp_at_2:,})")

    # Factor monotonicity
    print(f"\n  Factor effects (for grid narrowing):")
    for factor in ["w", "eta", "nf", "act"]:
        g_means = defaultdict(list)
        for c in conditions:
            g_means[c[factor]].append(c["mean_oracle"])
        means = {k: np.mean(v) for k, v in g_means.items()}
        effect   = max(means.values()) - min(means.values())
        best_val = max(means, key=means.get)
        sorted_means = [m for _, m in sorted(means.items())]
        if all(sorted_means[i] >= sorted_means[i+1] for i in range(len(sorted_means)-1)):
            trend = "monotonic (lower=better)"
        elif all(sorted_means[i] <= sorted_means[i+1] for i in range(len(sorted_means)-1)):
            trend = "monotonic (higher=better)"
        else:
            trend = "non-monotonic"
        print(f"    {factor}: effect={effect:.4f}, best={best_val}, {trend}")


def section8_archive_analysis(archives):
    """Section 8: Archive Analysis (re-checked with 5-filter BP)."""
    section_header("SECTION 8: ARCHIVE ANALYSIS")

    if not archives:
        print("\n  No archive files found.")
        return

    archive_types = sorted({at for _, at in archives.keys()})

    for atype in archive_types:
        subheader(atype)
        headers = ["act_dir", "N",  "MeanOr", "MaxOr", "MeanGC%",
                   "BP_5f", "BP_5f%", "Fwd", "RC", "Fwd%"]
        widths  = [14,        6,   8,        8,       9,
                   7,       8,         7,    7,    6]
        rows = []
        all_parts = {"oracle": [], "gc_pct": [], "bp_mask": [], "adapter": []}

        for (act_dir, at), data in sorted(archives.items()):
            if at != atype:
                continue
            oracle  = data["oracle"]
            gc_pct  = data["gc_pct"]
            bp_mask = data["bp_mask"]
            adapter = data["adapter"]
            n       = data["n"]

            n_bp  = int(bp_mask.sum())
            n_fwd = int((adapter == 0).sum())
            n_rc  = int((adapter == 1).sum())
            rows.append([
                act_dir, n, np.mean(oracle), np.max(oracle), np.mean(gc_pct),
                n_bp, f"{100*n_bp/n:.1f}",
                n_fwd, n_rc, f"{100*n_fwd/n:.1f}",
            ])
            for k in all_parts:
                all_parts[k].append(data[k])

        fmt_table(rows, headers, widths)

        if all_parts["oracle"]:
            all_oracle  = np.concatenate(all_parts["oracle"])
            all_gc      = np.concatenate(all_parts["gc_pct"])
            all_bp      = np.concatenate(all_parts["bp_mask"])
            all_adapter = np.concatenate(all_parts["adapter"])
            tn   = len(all_oracle)
            tbp  = int(all_bp.sum())
            tfwd = int((all_adapter == 0).sum())
            trc  = int((all_adapter == 1).sum())
            print(f"\n  Cross-total: {tn} seqs, mean={np.mean(all_oracle):.4f}, "
                  f"max={np.max(all_oracle):.4f}, GC={np.mean(all_gc):.1f}%, "
                  f"BP_5f={tbp}({100*tbp/tn:.1f}%), "
                  f"adapter={adapter_str(tfwd, trc, tn)}")

    print("\n  NOTE: 'BP_5f' uses all 5 filters (GC, entropy, CpG, strand_sym, homopolymer).")
    print("  archive_best_bio_plausible.h5 was collected with GC 45-55% + oracle>1 only;")
    print("  the 5-filter pass rate above may differ.")


def section9_entropy_gc(conditions, g, seed):
    """Section 9: Entropy & GC Control."""
    section_header("SECTION 9: ENTROPY & GC CONTROL")

    cond_ent    = [c["mean_entropy"] for c in conditions]
    cond_oracle = [c["mean_oracle"]  for c in conditions]
    seed_ent    = float(np.mean(seed["entropy"]))
    gen_ent     = float(np.mean(cond_ent))
    collapse    = 100.0 * (seed_ent - gen_ent) / seed_ent if seed_ent > 0 else 0.0

    subheader("Entropy Analysis")
    print(f"  Seed mean entropy:    {seed_ent:.4f}")
    print(f"  Generated mean:       {gen_ent:.4f}  (collapse: {collapse:.1f}%)")
    print(f"  Entropy range:        [{min(cond_ent):.4f}, {max(cond_ent):.4f}]")
    corr = np.corrcoef(cond_oracle, cond_ent)[0, 1]
    print(f"  r(mean_oracle, entropy) across conditions: {corr:.3f}")

    for factor in ["w", "eta", "nf", "act"]:
        grp = defaultdict(list)
        for c in conditions:
            grp[c[factor]].append(c["mean_entropy"])
        vals = sorted(grp.keys())
        print(f"  Entropy by {factor}: " +
              " | ".join(f"{v}={np.mean(grp[v]):.4f}" for v in vals))

    subheader("GC Control Analysis")
    gc_vals  = [c["mean_gc"] for c in conditions]
    seed_gc  = float(np.mean(seed["gc_pct"]))
    gen_gc   = float(np.mean(gc_vals))
    in_range = sum(1 for g_val in gc_vals if 40 <= g_val <= 60)

    print(f"  Seed mean GC%:        {seed_gc:.2f}%")
    print(f"  Generated mean GC%:   {gen_gc:.2f}%")
    print(f"  GC% range (cond):     [{min(gc_vals):.2f}%, {max(gc_vals):.2f}%]")
    print(f"  Conds with mean GC in [40-60%]: {in_range}/{len(gc_vals)} "
          f"({100*in_range/len(gc_vals):.1f}%)")

    for factor in ["w", "eta", "nf", "act"]:
        grp = defaultdict(list)
        for c in conditions:
            grp[c[factor]].append(c["mean_gc"])
        vals = sorted(grp.keys())
        print(f"  GC% by {factor}: " +
              " | ".join(f"{v}={np.mean(grp[v]):.2f}" for v in vals))


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generic round analysis for Rounded Pipeline DPS experiments"
    )
    parser.add_argument("--round_num", type=int, required=True,
                        help="Round number (e.g., 1 or 2)")
    parser.add_argument("--results_dir", default=None,
                        help="Root dir with act*/ subdirs "
                             "(default: results/rounded_pipeline/round{N}_dps_push)")
    parser.add_argument("--seed_pool", default=None,
                        help="Seed pool h5 file "
                             "(default: inferred from round_num)")
    parser.add_argument("--output", default=None,
                        help="Output text file "
                             "(default: results/rounded_pipeline/round{N}_analysis.txt)")
    args = parser.parse_args()

    R    = args.round_num
    base = Path("results/rounded_pipeline")

    if args.results_dir is None:
        args.results_dir = str(base / f"round{R}_dps_push")
    if args.seed_pool is None:
        if R == 1:
            args.seed_pool = str(base / "seed_pool_round1" / "true_seeds.h5")
        else:
            args.seed_pool = str(base / f"seed_pool_round{R}" / "seeds.h5")
    if args.output is None:
        args.output = str(base / f"round{R}_analysis.txt")

    # Redirect stdout through Tee
    tee = Tee(args.output)
    sys.stdout = tee

    try:
        print(f"Round {R} analysis → {args.output}")

        print("\nLoading seed pool ...")
        seed = load_seed_pool(args.seed_pool)
        n_fwd_s = int((seed["adapter"] == 0).sum())
        n_rc_s  = int((seed["adapter"] == 1).sum())
        print(f"  {seed['n']:,} seqs, mean oracle {np.mean(seed['oracle']):.4f}, "
              f"adapter: {adapter_str(n_fwd_s, n_rc_s, seed['n'])}")

        print("\nLoading conditions (computing 5-filter BP per file, may take a few minutes) ...")
        conditions, g_agg = load_all_conditions(args.results_dir)

        print("\nLoading archives ...")
        archives = load_archives(args.results_dir)
        print(f"  Found {len(archives)} archive files")

        # ── Sections ────────────────────────────────────────────────────
        section0_header(R, args.results_dir, args.seed_pool,
                        len(conditions), g_agg["total"])
        section1_seed_baseline(seed)
        section2_bio_landscape(g_agg, seed)
        section3_adapter_gc_oracle(g_agg)
        section4_condition_table(conditions)
        section5_winning_combos(conditions)
        section6_factor_effects(conditions)
        section7_round3_recommendation(conditions, g_agg, round_num=R)
        section8_archive_analysis(archives)
        section9_entropy_gc(conditions, g_agg, seed)

        print("\n" + "=" * W)
        print("  ANALYSIS COMPLETE")
        print("=" * W)

    finally:
        sys.stdout = tee._stdout
        tee.close()

    print(f"Analysis written to: {args.output}")


if __name__ == "__main__":
    main()
