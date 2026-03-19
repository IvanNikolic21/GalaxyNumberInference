"""
galaxy_lr.py
------------
Directional model selection via sequential log-likelihood ratio test.

Unlike the KS test (which is symmetric), this asks:
  "Given that model A is the null hypothesis, how many samples from model B
   are needed before we can prefer model B over model A at a given confidence?"

The log-likelihood ratio is:
    log LR = sum_i [ log p_B(x_i) - log p_A(x_i) ]

where x_i are samples drawn from model B, p_A and p_B are KDEs fit to the
full fiducial and stochastic d1s arrays respectively.

The significance threshold is calibrated by bootstrapping under the null
(drawing from model A itself), giving the 95th percentile of the null LR
distribution. The critical sample size is the first N where the observed
log-LR permanently exceeds this threshold.

Usage
-----
    from galaxy_lr import LRConfig, run_lr_analysis, summarise_lr, \
                          plot_lr_results, plot_lr_summary_bars
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LRConfig:
    """Parameters for the likelihood ratio bootstrap analysis.

    Parameters
    ----------
    n_trials : int
        Number of bootstrap trials per faint limit. Default: 2000.
    max_sample : int
        Maximum sample size to test per trial. Default: 100.
    significance : float
        False positive rate for threshold calibration. Default: 0.05.
    summary_percentile : float
        Upper percentile reported alongside median. Default: 90.
    n_null_bootstrap : int
        Number of null samples used to calibrate the LR threshold. Default: 1000.
    bw_method : str or float
        Bandwidth method passed to gaussian_kde. Default: 'scott'.
    """
    n_trials:          int   = 2000
    max_sample:        int   = 100
    significance:      float = 0.05
    summary_percentile: float = 90.0
    n_null_bootstrap:  int   = 1000
    bw_method:         str   = 'scott'


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _find_critical_sample(values: np.ndarray, thresholds: np.ndarray) -> int | None:
    """Find the first index where values permanently exceed threshold.

    Uses suffix-minimum: critical index is the first position where all
    subsequent values also exceed the threshold.
    """
    idx = np.where(values < thresholds)[0]
    return int(idx[0]) if len(idx) else None

def _calibrate_threshold(
    kde_A: gaussian_kde,
    max_sample: int,
    significance: float,
    n_null_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Calibrate LR threshold at each sample size under the null (model A).

    Draws samples from model A and computes the log-LR as if they were
    model B observations. Returns the (1-significance) percentile at each
    sample size — this is the threshold that must be exceeded to reject A.

    Returns
    -------
    thresholds : np.ndarray, shape (max_sample,)
    """
    # Store log-LR for each null trial at each sample size
    null_lrs = np.zeros((n_null_bootstrap, max_sample))

    for trial in range(n_null_bootstrap):
        # Sample from model A (null)
        null_samples = kde_A.resample(max_sample, seed=rng).flatten()
        log_ll_increments = np.log(np.clip(kde_A(null_samples), 1e-300, None))
        null_lrs[trial] = np.cumsum(log_ll_increments)

    # Threshold = (1-significance) percentile of null distribution at each N
    return np.percentile(null_lrs, significance*100, axis=0) #+ np.log(significance) #alterations

def _lr_trial(
    kde_A: gaussian_kde,
    arr_stoc: np.ndarray,
    thresholds: np.ndarray,
    max_sample: int,
    rng: np.random.Generator,
) -> int | None:
    """Run a single bootstrap trial for the LR test.

    Draws samples from model B, computes cumulative log-LR, finds critical N.
    """
    sample_B = rng.choice(arr_stoc, size=max_sample, replace=True)
    log_ll = np.cumsum(np.log(np.clip(kde_A(sample_B), 1e-300, None)))
    return _find_critical_sample(log_ll, thresholds)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_lr_analysis(
    d1s_fid: dict,
    d1s_stoc: dict,
    cfg,
    lr_cfg: Optional[LRConfig] = None,
    bright_key: str = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Run LR bootstrap analysis for all faint limits.

    Parameters
    ----------
    d1s_fid, d1s_stoc : dicts
        Output of compute_d1s() or load_d1s().
    cfg : AnalysisConfig
    lr_cfg : LRConfig, optional
    bright_key : str, optional
        Defaults to cfg.bright_names[0].
    seed : int

    Returns
    -------
    results : dict
        results[faint_key] = np.ndarray of shape (n_trials,)
        NaN entries indicate inconclusive trials.
    """
    if lr_cfg is None:
        lr_cfg = LRConfig()
    if bright_key is None:
        bright_key = cfg.bright_names[0]

    rng = np.random.default_rng(seed)
    results: dict[str, np.ndarray] = {}

    for fkey in cfg.faint_names:
        arr_fid  = d1s_fid[bright_key][fkey]
        arr_stoc = d1s_stoc[bright_key][fkey]

        if len(arr_fid) < lr_cfg.max_sample or len(arr_stoc) < lr_cfg.max_sample:
            print(f"  Warning: {fkey} has too few entries "
                  f"({len(arr_fid)} fid, {len(arr_stoc)} stoc) — skipping.")
            results[fkey] = np.full(lr_cfg.n_trials, np.nan)
            continue

        # Fit KDEs once per (bkey, fkey)
        try:
            kde_A = gaussian_kde(arr_fid,  bw_method=lr_cfg.bw_method)
            #kde_B = gaussian_kde(arr_stoc, bw_method=lr_cfg.bw_method)
        except Exception as e:
            print(f"  Warning: KDE failed for {fkey}: {e} — skipping.")
            results[fkey] = np.full(lr_cfg.n_trials, np.nan)
            continue

        # Calibrate threshold under null
        thresholds = _calibrate_threshold(
            kde_A, #kde_B,
            lr_cfg.max_sample, lr_cfg.significance,
            lr_cfg.n_null_bootstrap, rng,
        )

        # Bootstrap trials
        critical_ns = np.full(lr_cfg.n_trials, np.nan)
        for trial in range(lr_cfg.n_trials):
            c = _lr_trial(kde_A, arr_stoc, thresholds, lr_cfg.max_sample, rng)
            if c is not None:
                critical_ns[trial] = c

        results[fkey] = critical_ns

        print(f"  {bright_key} | {fkey}:  "
              f"median={np.nanmedian(critical_ns):.1f}  "
              f"p{lr_cfg.summary_percentile:.0f}={np.nanpercentile(critical_ns, lr_cfg.summary_percentile):.1f}  "
              f"inconclusive={np.isnan(critical_ns).sum()}/{lr_cfg.n_trials}")

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise_lr(results: dict, lr_cfg: Optional[LRConfig] = None) -> str:
    """Formatted summary table, ordered by faint_limit as given."""
    if lr_cfg is None:
        lr_cfg = LRConfig()

    pct = lr_cfg.summary_percentile
    lines = [
        f"{'faint_key':<12}  {'median':>8}  {'p'+str(int(pct)):>8}  {'n_inconclusive':>14}",
        "-" * 48,
    ]
    for fkey, arr in results.items():
        lines.append(
            f"{fkey:<12}  "
            f"{np.nanmedian(arr):>8.1f}  "
            f"{np.nanpercentile(arr, pct):>8.1f}  "
            f"{np.isnan(arr).sum():>14d}"
        )
    lines.append("-" * 48)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_lr_results(
    results: dict,
    lr_cfg: Optional[LRConfig] = None,
    bright_key: str = "",
    redshift_label: float = None,
    figsize: tuple = (8, 5),
    n_bins: int = 30,
) -> plt.Figure:
    """Histogram of LR critical sample sizes, ordered by faint_limit."""
    if lr_cfg is None:
        lr_cfg = LRConfig()

    fig, ax = plt.subplots(figsize=figsize)

    for fkey, arr in results.items():
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            ax.hist(valid, bins=n_bins, alpha=0.4, label=fkey.replace("M", "-"))

    ax.set_xlabel("Pointings needed", fontsize=13)
    ax.set_ylabel("Trial count", fontsize=13)

    title = "Likelihood Ratio test"
    parts = []
    if bright_key:
        parts.append(rf"$M_{{UV,0}}<-{bright_key.replace('M','')}$")
    if redshift_label is not None:
        parts.append(rf"$z={redshift_label}$")
    if parts:
        title += "\n" + ",  ".join(parts)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    return fig


def plot_lr_summary_bars(
    results: dict,
    lr_cfg: Optional[LRConfig] = None,
    bright_key: str = "",
    redshift_label: float = None,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Bar chart of median LR critical sample sizes, ordered by faint_limit."""
    if lr_cfg is None:
        lr_cfg = LRConfig()

    fkeys  = list(results.keys())
    labels = [k.replace("M", "-") for k in fkeys]

    medians = [np.nanmedian(results[k]) for k in fkeys]
    p90s    = [np.nanpercentile(results[k], lr_cfg.summary_percentile) for k in fkeys]
    errors  = [p - m for p, m in zip(p90s, medians)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, medians, yerr=errors, capsize=4, alpha=0.8,
           color="steelblue", error_kw={"ecolor": "black", "lw": 1.5})
    ax.set_xlabel(r"Faint limit $M_{\rm UV}$", fontsize=13)
    ax.set_ylabel("Median pointings needed", fontsize=13)
    ax.tick_params(axis="x", rotation=45)

    title = "Likelihood Ratio test"
    parts = []
    if bright_key:
        parts.append(rf"$M_{{UV,0}}<-{bright_key.replace('M','')}$")
    if redshift_label is not None:
        parts.append(rf"$z={redshift_label}$")
    if parts:
        title += "\n" + ",  ".join(parts)
    ax.set_title(title, fontsize=12)

    fig.tight_layout()
    return fig