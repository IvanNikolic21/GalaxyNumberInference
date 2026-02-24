"""
galaxy_ks.py
------------
KS and Anderson-Darling test analysis to find the faint magnitude limit
that most efficiently distinguishes fiducial from stochastic models.

For a fixed bright_key, sweeps over all faint_keys and estimates via
bootstrap the minimum number of pointings needed to distinguish the two
d1 distributions at a given significance level.

Both KS and AD tests are run in parallel:
- KS is most sensitive to differences near the distribution center
- AD weights the tails more heavily, catching differences KS may miss

Usage
-----
    from galaxy_ks import KSConfig, run_ks_analysis, plot_ks_results, \
                          plot_ks_summary_bars, summarise_ks

    ks_cfg  = KSConfig()
    results = run_ks_analysis(d1s_fid, d1s_stoc, cfg, ks_cfg, bright_key='M21.5')

    print(summarise_ks(results, ks_cfg))
    fig = plot_ks_results(results, ks_cfg, bright_key='M21.5', redshift_label=10.5)
    fig = plot_ks_summary_bars(results, ks_cfg, bright_key='M21.5', redshift_label=10.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, anderson_ksamp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KSConfig:
    """Parameters for the KS/AD bootstrap analysis.

    Parameters
    ----------
    n_trials : int
        Number of bootstrap trials per faint limit. Default: 2000.
    max_sample : int
        Maximum sample size to test per trial. Default: 100.
    significance : float
        P-value threshold for rejecting the null hypothesis. Default: 0.05.
    summary_percentile : float
        Upper percentile reported alongside median as spread measure. Default: 90.
    """
    n_trials: int = 2000
    max_sample: int = 100
    significance: float = 0.05
    summary_percentile: float = 90.0


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _find_critical_sample(pvalues: np.ndarray, threshold: float) -> int | None:
    """Find the first index where p-value drops and stays below threshold.

    Uses suffix-maximum: critical index is the first position where all
    subsequent p-values are also below the threshold.
    """
    suffix_max = np.maximum.accumulate(pvalues[::-1])[::-1]
    suffix_max = np.r_[suffix_max[1:], -np.inf]
    idx = np.where(suffix_max < threshold)[0]
    return int(idx[0]) if len(idx) else None


def _bootstrap_trial(
    arr_fid: np.ndarray,
    arr_stoc: np.ndarray,
    max_sample: int,
    significance: float,
    rng: np.random.Generator,
) -> tuple[int | None, int | None]:
    """Run a single bootstrap trial for both KS and AD tests.

    Draws i samples with replacement from each array for i in [3, max_sample].

    Returns
    -------
    critical_ks, critical_ad : int or None
        Critical sample sizes for KS and AD respectively.
    """
    pvalues_ks = np.zeros(max_sample)
    pvalues_ad = np.zeros(max_sample)

    for i in range(3, max_sample):
        sample_fid  = rng.choice(arr_fid,  size=i, replace=True)
        sample_stoc = rng.choice(arr_stoc, size=i, replace=True)

        pvalues_ks[i] = kstest(sample_fid, sample_stoc).pvalue

        # AD p-value is capped at [0.001, 0.25] by scipy
        try:
            pvalues_ad[i] = anderson_ksamp([sample_fid, sample_stoc]).pvalue
        except Exception:
            pvalues_ad[i] = 1.0

    return (
        _find_critical_sample(pvalues_ks, significance),
        _find_critical_sample(pvalues_ad, significance),
    )


def run_ks_analysis(
    d1s_fid: dict,
    d1s_stoc: dict,
    cfg,
    ks_cfg: Optional[KSConfig] = None,
    bright_key: str = None,
    seed: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """Run KS and AD bootstrap analysis for all faint limits.

    Parameters
    ----------
    d1s_fid, d1s_stoc : dicts
        Output of compute_d1s() or load_d1s().
    cfg : AnalysisConfig
    ks_cfg : KSConfig, optional
    bright_key : str, optional
        Defaults to cfg.bright_names[0].
    seed : int

    Returns
    -------
    results : dict
        results[faint_key]['ks'] = np.ndarray of shape (n_trials,)
        results[faint_key]['ad'] = np.ndarray of shape (n_trials,)
        NaN entries indicate inconclusive trials.
    """
    if ks_cfg is None:
        ks_cfg = KSConfig()
    if bright_key is None:
        bright_key = cfg.bright_names[0]

    rng = np.random.default_rng(seed)
    results: dict[str, dict[str, np.ndarray]] = {}

    for fkey in cfg.faint_names:
        arr_fid  = d1s_fid[bright_key][fkey]
        arr_stoc = d1s_stoc[bright_key][fkey]

        if len(arr_fid) < ks_cfg.max_sample or len(arr_stoc) < ks_cfg.max_sample:
            print(f"  Warning: {fkey} has too few entries "
                  f"({len(arr_fid)} fid, {len(arr_stoc)} stoc) — skipping.")
            results[fkey] = {
                'ks': np.full(ks_cfg.n_trials, np.nan),
                'ad': np.full(ks_cfg.n_trials, np.nan),
            }
            continue

        critical_ks = np.full(ks_cfg.n_trials, np.nan)
        critical_ad = np.full(ks_cfg.n_trials, np.nan)

        for trial in range(ks_cfg.n_trials):
            c_ks, c_ad = _bootstrap_trial(
                arr_fid, arr_stoc, ks_cfg.max_sample, ks_cfg.significance, rng
            )
            if c_ks is not None:
                critical_ks[trial] = c_ks
            if c_ad is not None:
                critical_ad[trial] = c_ad

        results[fkey] = {'ks': critical_ks, 'ad': critical_ad}

        print(f"  {bright_key} | {fkey}:  "
              f"KS median={np.nanmedian(critical_ks):.1f}  "
              f"p{ks_cfg.summary_percentile:.0f}={np.nanpercentile(critical_ks, ks_cfg.summary_percentile):.1f}  "
              f"AD median={np.nanmedian(critical_ad):.1f}  "
              f"p{ks_cfg.summary_percentile:.0f}={np.nanpercentile(critical_ad, ks_cfg.summary_percentile):.1f}  "
              f"inconclusive KS={np.isnan(critical_ks).sum()} AD={np.isnan(critical_ad).sum()}"
              f"/{ks_cfg.n_trials}")

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise_ks(
    results: dict,
    ks_cfg: Optional[KSConfig] = None,
) -> str:
    """Formatted summary table of median and spread for both KS and AD.

    Ordered by faint_limit as given (no sorting by statistic).
    """
    if ks_cfg is None:
        ks_cfg = KSConfig()

    pct = ks_cfg.summary_percentile
    lines = [
        f"{'faint_key':<12}  {'KS med':>8}  {'KS p'+str(int(pct)):>8}  "
        f"{'AD med':>8}  {'AD p'+str(int(pct)):>8}  {'n_inconclusive':>14}",
        "-" * 68,
    ]
    for fkey, res in results.items():
        ks, ad = res['ks'], res['ad']
        lines.append(
            f"{fkey:<12}  "
            f"{np.nanmedian(ks):>8.1f}  "
            f"{np.nanpercentile(ks, pct):>8.1f}  "
            f"{np.nanmedian(ad):>8.1f}  "
            f"{np.nanpercentile(ad, pct):>8.1f}  "
            f"{np.isnan(ks).sum():>7d} / {np.isnan(ad).sum():<7d}"
        )
    lines.append("-" * 68)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ks_results(
    results: dict,
    ks_cfg: Optional[KSConfig] = None,
    bright_key: str = "",
    redshift_label: float = None,
    figsize: tuple = (12, 5),
    n_bins: int = 30,
) -> plt.Figure:
    """Side-by-side histogram panels for KS and AD critical sample sizes.

    Ordered by faint_limit as given in config — no sorting.
    """
    if ks_cfg is None:
        ks_cfg = KSConfig()

    fig, (ax_ks, ax_ad) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for fkey, res in results.items():
        label = fkey.replace("M", "-")
        for ax, test in [(ax_ks, 'ks'), (ax_ad, 'ad')]:
            valid = res[test][~np.isnan(res[test])]
            if len(valid) > 0:
                ax.hist(valid, bins=n_bins, alpha=0.4, label=label)

    for ax, title in [(ax_ks, 'KS test'), (ax_ad, 'Anderson-Darling')]:
        ax.set_xlabel("Pointings needed", fontsize=13)
        ax.set_ylabel("Trial count", fontsize=13)
        subtitle = title
        if bright_key or redshift_label is not None:
            parts = []
            if bright_key:
                parts.append(rf"$M_{{UV,0}}<-{bright_key.replace('M','')}$")
            if redshift_label is not None:
                parts.append(rf"$z={redshift_label}$")
            subtitle += "\n" + ",  ".join(parts)
        ax.set_title(subtitle, fontsize=12)
        ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    return fig


def plot_ks_summary_bars(
    results: dict,
    ks_cfg: Optional[KSConfig] = None,
    bright_key: str = "",
    redshift_label: float = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Side-by-side bar charts for KS and AD, ordered by faint_limit.

    Error bars run from median to the summary_percentile.
    """
    if ks_cfg is None:
        ks_cfg = KSConfig()

    # Preserve config order — no sorting
    fkeys  = list(results.keys())
    labels = [k.replace("M", "-") for k in fkeys]

    fig, (ax_ks, ax_ad) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, test, title in [
        (ax_ks, 'ks', 'KS test'),
        (ax_ad, 'ad', 'Anderson-Darling'),
    ]:
        medians = [np.nanmedian(results[k][test]) for k in fkeys]
        p90s    = [np.nanpercentile(results[k][test], ks_cfg.summary_percentile) for k in fkeys]
        errors  = [p - m for p, m in zip(p90s, medians)]

        ax.bar(labels, medians, yerr=errors, capsize=4, alpha=0.8,
               color="steelblue", error_kw={"ecolor": "black", "lw": 1.5})
        ax.set_xlabel(r"Faint limit $M_{\rm UV}$", fontsize=13)
        ax.set_ylabel("Median pointings needed", fontsize=13)
        ax.tick_params(axis="x", rotation=45)

        subtitle = title
        if bright_key or redshift_label is not None:
            parts = []
            if bright_key:
                parts.append(rf"$M_{{UV,0}}<-{bright_key.replace('M','')}$")
            if redshift_label is not None:
                parts.append(rf"$z={redshift_label}$")
            subtitle += "\n" + ",  ".join(parts)
        ax.set_title(subtitle, fontsize=12)

    fig.tight_layout()
    return fig
