"""
galaxy_ks.py
------------
KS-test based analysis to find the faint magnitude limit that most
efficiently distinguishes the fiducial from the stochastic model.

For a fixed bright_key (and implicitly a fixed redshift, since d1s arrays
are computed per redshift), we sweep over all faint_keys and ask:
"How many bright galaxy pointings are needed before the KS test
consistently rejects the null hypothesis that both models are the same?"

The answer is estimated via bootstrap: for each trial we draw i samples
with replacement from each model's d1s array and compute the KS p-value.
We find the smallest i at which the p-value drops and stays below the
significance threshold. Repeating this n_trials times gives a distribution
of critical sample sizes — the faint limit with the smallest median (and
tightest spread) is the most observationally efficient.

Usage
-----
    from galaxy_d1s import load_d1s
    from galaxy_ks import KSConfig, run_ks_analysis, plot_ks_results, summarise_ks

    from galaxy_neighbors import AnalysisConfig
    cfg = AnalysisConfig(...)

    d1s_fid  = load_d1s('cache/z10.5/d1s_fiducial_real10.npz',  cfg)
    d1s_stoc = load_d1s('cache/z10.5/d1s_stochastic_real10.npz', cfg)

    ks_cfg = KSConfig()
    results = run_ks_analysis(d1s_fid, d1s_stoc, cfg, ks_cfg, bright_key='M21.5')

    fig = plot_ks_results(results, ks_cfg)
    print(summarise_ks(results))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KSConfig:
    """Parameters for the KS test analysis.

    Parameters
    ----------
    n_trials : int
        Number of bootstrap trials per faint limit. Default: 1000.
    max_sample : int
        Maximum sample size to test. If the critical size exceeds this,
        the trial is recorded as None (inconclusive). Default: 100.
    significance : float
        P-value threshold below which the KS test is considered to reject
        the null hypothesis. Default: 0.05.
    summary_percentile : float
        Upper percentile reported alongside the median as a spread measure.
        Default: 90.
    """
    n_trials: int = 1000
    max_sample: int = 100
    significance: float = 0.05
    summary_percentile: float = 90.0


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _find_critical_sample(pvalues: np.ndarray, threshold: float) -> int | None:
    """Find the first index where p-value drops and stays below threshold.

    Uses a suffix-maximum approach: the critical index is the first position
    where all subsequent p-values are also below the threshold.

    Parameters
    ----------
    pvalues : np.ndarray
        Array of p-values indexed by sample size.
    threshold : float

    Returns
    -------
    int or None
        Critical sample size, or None if never reached.
    """
    suffix_max = np.maximum.accumulate(pvalues[::-1])[::-1]
    # Shift left so the suffix excludes the current element
    suffix_max = np.r_[suffix_max[1:], -np.inf]
    idx = np.where(suffix_max < threshold)[0]
    return int(idx[0]) if len(idx) else None


def _ks_trial(
    arr_fid: np.ndarray,
    arr_stoc: np.ndarray,
    max_sample: int,
    significance: float,
    rng: np.random.Generator,
) -> int | None:
    """Run a single bootstrap KS trial.

    Draws i samples with replacement from each array for i in [3, max_sample],
    computes the KS p-value at each i, then finds the critical sample size.

    Parameters
    ----------
    arr_fid, arr_stoc : np.ndarray
        Full d1s arrays for fiducial and stochastic models.
    max_sample : int
    significance : float
    rng : np.random.Generator

    Returns
    -------
    int or None
    """
    pvalues = np.zeros(max_sample)
    for i in range(3, max_sample):
        sample_fid  = rng.choice(arr_fid,  size=i, replace=True)
        sample_stoc = rng.choice(arr_stoc, size=i, replace=True)
        pvalues[i]  = kstest(sample_fid, sample_stoc).pvalue

    return _find_critical_sample(pvalues, significance)


def run_ks_analysis(
    d1s_fid: dict[str, dict[str, np.ndarray]],
    d1s_stoc: dict[str, dict[str, np.ndarray]],
    cfg,                        # AnalysisConfig
    ks_cfg: Optional[KSConfig] = None,
    bright_key: str = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Run the full KS bootstrap analysis for all faint limits.

    For a fixed bright_key, sweeps over all faint_keys and estimates the
    distribution of critical sample sizes via bootstrap.

    Parameters
    ----------
    d1s_fid, d1s_stoc : dicts
        Output of compute_d1s() or load_d1s().
    cfg : AnalysisConfig
    ks_cfg : KSConfig, optional
    bright_key : str, optional
        Which bright threshold to analyse. Defaults to cfg.bright_names[0].
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        results[faint_key] = np.ndarray of shape (n_trials,) containing
        critical sample sizes. None trials are stored as np.nan.
    """
    if ks_cfg is None:
        ks_cfg = KSConfig()
    if bright_key is None:
        bright_key = cfg.bright_names[0]

    rng = np.random.default_rng(seed)
    results: dict[str, np.ndarray] = {}

    for fkey in cfg.faint_names:
        arr_fid  = d1s_fid[bright_key][fkey]
        arr_stoc = d1s_stoc[bright_key][fkey]

        if len(arr_fid) < ks_cfg.max_sample or len(arr_stoc) < ks_cfg.max_sample:
            print(
                f"  Warning: {fkey} has fewer entries than max_sample "
                f"({len(arr_fid)} fid, {len(arr_stoc)} stoc) — skipping."
            )
            results[fkey] = np.full(ks_cfg.n_trials, np.nan)
            continue

        critical_sizes = np.full(ks_cfg.n_trials, np.nan)
        for trial in range(ks_cfg.n_trials):
            c = _ks_trial(arr_fid, arr_stoc, ks_cfg.max_sample, ks_cfg.significance, rng)
            if c is not None:
                critical_sizes[trial] = c

        results[fkey] = critical_sizes
        print(f"  {bright_key} | {fkey}: "
              f"median={np.nanmedian(critical_sizes):.1f}  "
              f"p{ks_cfg.summary_percentile:.0f}={np.nanpercentile(critical_sizes, ks_cfg.summary_percentile):.1f}  "
              f"inconclusive={np.isnan(critical_sizes).sum()}/{ks_cfg.n_trials}")

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise_ks(
    results: dict[str, np.ndarray],
    ks_cfg: Optional[KSConfig] = None,
) -> str:
    """Return a formatted summary table of median and spread per faint limit.

    The 'score' column is the 90th percentile — a single number capturing
    both typical cost (median) and worst-case reliability (upper tail).
    Lower is better.
    """
    if ks_cfg is None:
        ks_cfg = KSConfig()

    pct = ks_cfg.summary_percentile
    lines = [
        f"{'faint_key':<12}  {'median':>8}  {'p'+str(int(pct)):>8}  {'inconclusive':>12}",
        "-" * 48,
    ]
    # Sort by median so the best limit floats to the top
    sorted_keys = sorted(results, key=lambda k: np.nanmedian(results[k]))
    for fkey in sorted_keys:
        arr = results[fkey]
        lines.append(
            f"{fkey:<12}  "
            f"{np.nanmedian(arr):>8.1f}  "
            f"{np.nanpercentile(arr, pct):>8.1f}  "
            f"{np.isnan(arr).sum():>12d}"
        )
    lines.append("-" * 48)
    lines.append(f"Best limit (lowest median): {sorted_keys[0]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ks_results(
    results: dict[str, np.ndarray],
    ks_cfg: Optional[KSConfig] = None,
    bright_key: str = "",
    redshift_label: float = None,
    figsize: tuple = (9, 5),
    n_bins: int = 30,
) -> plt.Figure:
    """Plot overlapping histograms of critical sample size distributions.

    Parameters
    ----------
    results : dict
        Output of run_ks_analysis().
    ks_cfg : KSConfig, optional
    bright_key : str
        Used in the plot title.
    redshift_label : float, optional
        Redshift shown in the title if provided.
    figsize : tuple
    n_bins : int

    Returns
    -------
    fig : matplotlib.Figure
    """
    if ks_cfg is None:
        ks_cfg = KSConfig()

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by median so legend order matches "best to worst"
    sorted_keys = sorted(results, key=lambda k: np.nanmedian(results[k]))

    for fkey in sorted_keys:
        arr = results[fkey]
        valid = arr[~np.isnan(arr)]
        median = np.nanmedian(arr)
        ax.hist(
            valid, bins=n_bins, alpha=0.5,
            label=rf"$M_{{UV}}<{fkey.replace('M','-')}$  (med={median:.0f})",
        )

    ax.set_xlabel("Number of bright galaxy pointings needed", fontsize=14)
    ax.set_ylabel("Trial count", fontsize=14)

    title = "KS test: pointings needed to distinguish models at "
    title += rf"{int((1-ks_cfg.significance)*100)}% confidence"
    if bright_key or redshift_label is not None:
        subtitle_parts = []
        if bright_key:
            subtitle_parts.append(rf"$M_{{UV,0}}<-{bright_key.replace('M','')}$")
        if redshift_label is not None:
            subtitle_parts.append(rf"$z={redshift_label}$")
        title += "\n" + ",  ".join(subtitle_parts)

    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    fig.tight_layout()
    return fig


def plot_ks_summary_bars(
    results: dict[str, np.ndarray],
    ks_cfg: Optional[KSConfig] = None,
    bright_key: str = "",
    redshift_label: float = None,
    figsize: tuple = (9, 5),
) -> plt.Figure:
    """Bar chart of median critical sample size with 90th percentile error bars.

    A compact alternative to the histogram plot — useful when comparing
    many faint limits at once or across redshifts.
    """
    if ks_cfg is None:
        ks_cfg = KSConfig()

    sorted_keys = sorted(results, key=lambda k: np.nanmedian(results[k]))
    medians = [np.nanmedian(results[k]) for k in sorted_keys]
    p90s    = [np.nanpercentile(results[k], ks_cfg.summary_percentile) for k in sorted_keys]
    errors  = [p - m for p, m in zip(p90s, medians)]
    labels  = [k.replace("M", "-") for k in sorted_keys]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, medians, yerr=errors, capsize=5, alpha=0.8,
           color="steelblue", error_kw={"ecolor": "black", "lw": 1.5})
    ax.set_xlabel(r"Faint limit $M_{\rm UV}$", fontsize=14)
    ax.set_ylabel("Median pointings needed", fontsize=14)

    title = rf"KS efficiency by faint limit"
    if bright_key or redshift_label is not None:
        parts = []
        if bright_key:
            parts.append(rf"$M_{{UV,0}}<-{bright_key.replace('M','')}$")
        if redshift_label is not None:
            parts.append(rf"$z={redshift_label}$")
        title += "\n" + ",  ".join(parts)

    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig
