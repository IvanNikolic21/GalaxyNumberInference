#!/usr/bin/env python
"""
forecast_count_significance.py
-------------------------------
Forecast, for a fixed-aperture observing proposal, how many "sigma" the
fiducial (high-luminosity) and stochastic (high-stochasticity) models are
separated by, as a function of the number of independent bright-galaxy
pointings N, using raw neighbor COUNTS (not d1) as the summary statistic.

Unlike the KS-test forecast used for d1 in Paper I, this uses a
log-likelihood-ratio / Wilks'-theorem approach: since both models give a
fully specified, simulatable count distribution p(n | model) per pointing
(no free parameters), the LR test is the Neyman-Pearson-optimal test for
distinguishing them -- strictly more powerful than a generic nonparametric
test like KS. This is also the same convention already used in Paper I for
the single-pointing GN-z11 result (Delta ln L ~ 1.97 -> sigma ~ 2.0, via
sigma = sqrt(2 * |Delta ln L|), Wilks 1938).

For each of --n-trials bootstrap trials and each N in --n-values:
    draw N counts (with replacement) from the TRUE model's raw count array
    Delta ln L = sum_i [ln p_fid(n_i) - ln p_stoc(n_i)]
    sigma_N    = sqrt(2 * |Delta ln L|)
Repeated once assuming fiducial is the true model, once assuming
stochastic is -- a real proposal should be defensible either way.

All physical/observational choices are CLI flags -- nothing is hardcoded,
since none of these numbers (aperture, magnitudes, redshift) are settled.

NOTE on aperture convention: AnalysisConfig.survey_area_arcmin2 feeds
search_box_mpc(), which computes side = sqrt(area) and passes it straight
through as `half_side` to the box search (extends +/-half_side in each
dimension) -- so the box actually searched has full side 2*sqrt(area),
i.e. an effective area of 4x the nominal --area-arcmin2 value. This is an
existing convention already baked into every Paper I number, not
introduced here. If you want an exact real-instrument FOV searched, pass
area/4 instead of the raw FOV value.

Usage
-----
    python forecast_count_significance.py \\
        --redshift 12.0 --area-arcmin2 4.84 --muv0 -21.5 --muvlim -18.0 \\
        --n-values 2 3 4 5 --n-realizations 100 --n-trials 2000 \\
        --output-dir /groups/astro/ivannik/projects/Neighbors/count_forecast
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from galaxy_neighbors import AnalysisConfig, run_neighbor_analysis, _mag_to_key
from run_ks import REDSHIFT_CONFIGS, N_REALIZATIONS  # reuse the same path registry as run_ks.py


def available_realizations(path: Path) -> int:
    """Actual number of realizations stored in a muv catalog's 'data' dataset.

    run_neighbor_analysis indexes f["data"][idx] directly per realization
    (not a bounds-safe slice), so requesting more than this raises an
    IndexError deep inside galaxy_neighbors.py -- check first instead.
    """
    with h5py.File(path, "r") as f:
        return f["data"].shape[0]

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def build_count_pmf(counts: np.ndarray, max_count: int, alpha: float = 1.0) -> np.ndarray:
    """Laplace-smoothed empirical PMF over n = 0..max_count.

    alpha : additive (Laplace) smoothing pseudo-count per bin -- avoids
        ln(0) when a count value seen under one model has zero raw
        frequency under the other model's histogram.
    """
    hist = np.bincount(counts, minlength=max_count + 1)[: max_count + 1].astype(float)
    hist += alpha
    return hist / hist.sum()


def forecast_sigma_vs_n(
    counts_fid: np.ndarray,
    counts_stoc: np.ndarray,
    n_values: list[int],
    n_trials: int,
    rng: np.random.Generator,
) -> dict:
    """Bootstrap sigma_N for both true-model assumptions.

    Returns
    -------
    dict with keys 'fid_true' and 'stoc_true', each mapping N -> array of
    shape (n_trials,) of sigma values from that trial.
    """
    max_count = int(max(counts_fid.max(), counts_stoc.max()))
    pmf_fid  = build_count_pmf(counts_fid,  max_count)
    pmf_stoc = build_count_pmf(counts_stoc, max_count)
    ln_fid, ln_stoc = np.log(pmf_fid), np.log(pmf_stoc)

    out = {"fid_true": {}, "stoc_true": {}}
    for true_label, true_counts in [("fid_true", counts_fid), ("stoc_true", counts_stoc)]:
        for N in n_values:
            sigmas = np.empty(n_trials)
            for t in range(n_trials):
                draw = rng.choice(true_counts, size=N, replace=True)
                dlnL = np.sum(ln_fid[draw] - ln_stoc[draw])
                sigmas[t] = np.sqrt(2.0 * abs(dlnL))
            out[true_label][N] = sigmas
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--redshift", type=float, required=True, choices=sorted(REDSHIFT_CONFIGS.keys()))
    p.add_argument("--area-arcmin2", type=float, required=True,
                   help="Passed straight through to AnalysisConfig.survey_area_arcmin2 -- "
                        "see the aperture-convention note in this script's docstring.")
    p.add_argument("--muv0", type=float, required=True,
                   help="Bright-galaxy magnitude threshold (galaxy is 'bright' if M_UV < muv0).")
    p.add_argument("--muvlim", type=float, required=True,
                   help="Faint neighbor magnitude limit.")
    p.add_argument("--n-realizations", type=int, default=None,
                   help="How many mock catalog realizations to load per model when building the "
                        "count PMFs. Defaults to run_ks.py's N_REALIZATIONS for --redshift.")
    p.add_argument("--n-values", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                   help="Number of independent pointings to forecast sigma at.")
    p.add_argument("--n-trials", type=int, default=2000, help="Bootstrap trials per N.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    z_cfg = REDSHIFT_CONFIGS[args.redshift]
    n_realizations = args.n_realizations or N_REALIZATIONS[args.redshift]

    n_avail_fid  = available_realizations(z_cfg.muv_fiducial_path)
    n_avail_stoc = available_realizations(z_cfg.muv_stochastic_path)
    n_avail = min(n_avail_fid, n_avail_stoc)
    if n_realizations > n_avail:
        log.warning(f"Requested n_realizations={n_realizations}, but fiducial catalog has "
                    f"{n_avail_fid} and stochastic has {n_avail_stoc} available -- "
                    f"clipping to {n_avail}.")
        n_realizations = n_avail

    cfg = AnalysisConfig(
        bright_limits=[args.muv0],
        faint_limits=[args.muvlim],
        preselect_faint_limit=args.muvlim,
        survey_area_arcmin2=args.area_arcmin2,
    )
    bright_key, faint_key = _mag_to_key(args.muv0), _mag_to_key(args.muvlim)

    log.info(f"z={args.redshift}  area={args.area_arcmin2} arcmin^2  "
             f"M_UV,0={args.muv0}  M_UV,lim={args.muvlim}  n_realizations={n_realizations}")
    log.info("Running neighbor search (this builds the count PMFs from scratch) ...")
    results_fid, results_stoc = run_neighbor_analysis(
        redshift_cfg=z_cfg, analysis_cfg=cfg, n_realizations=n_realizations,
    )
    counts_fid  = np.array([r.n_neighbors for r in results_fid[bright_key][faint_key]])
    counts_stoc = np.array([r.n_neighbors for r in results_stoc[bright_key][faint_key]])
    log.info(f"  fiducial:   {len(counts_fid)} bright galaxies, "
             f"mean count={counts_fid.mean():.2f}, max={counts_fid.max()}")
    log.info(f"  stochastic: {len(counts_stoc)} bright galaxies, "
             f"mean count={counts_stoc.mean():.2f}, max={counts_stoc.max()}")

    # ------------------------------------------------------------------
    # Expected number of FAINT neighbors (M_UV < muvlim) within the FoV
    # aperture per pointing, for both models -- this is just a clearer
    # summary of counts_fid/counts_stoc themselves (already the per-pointing
    # neighbor counts the whole forecast is built from), not a separate
    # calculation.
    # ------------------------------------------------------------------
    mean_fid,  std_fid  = counts_fid.mean(),  counts_fid.std()
    mean_stoc, std_stoc = counts_stoc.mean(), counts_stoc.std()
    p16_fid,  p50_fid,  p84_fid  = np.percentile(counts_fid,  [2.5, 50, 97.5])
    p16_stoc, p50_stoc, p84_stoc = np.percentile(counts_stoc, [2.5, 50, 97.5])
    frac_zero_fid  = float((counts_fid  == 0).mean())
    frac_zero_stoc = float((counts_stoc == 0).mean())

    print(f"\nExpected number of faint (M_UV < {args.muvlim}) neighbors within the "
          f"{args.area_arcmin2} arcmin^2 FoV, per bright-galaxy pointing:")
    print(f"  {'model':<12}  {'mean +/- std':>16}  {'median [2.5,97.5]':>20}  {'P(n=0)':>8}")
    print(f"  {'-'*62}")
    print(f"  {'fiducial':<12}  {mean_fid:>6.2f} +/- {std_fid:<6.2f}  "
          f"{p50_fid:>6.1f} [{p16_fid:.1f},{p84_fid:.1f}]  {frac_zero_fid:>8.3f}")
    print(f"  {'stochastic':<12}  {mean_stoc:>6.2f} +/- {std_stoc:<6.2f}  "
          f"{p50_stoc:>6.1f} [{p16_stoc:.1f},{p84_stoc:.1f}]  {frac_zero_stoc:>8.3f}")

    log.info(f"Bootstrapping sigma(N) for N={args.n_values}, {args.n_trials} trials each ...")
    result = forecast_sigma_vs_n(counts_fid, counts_stoc, args.n_values, args.n_trials, rng)

    # ------------------------------------------------------------------
    # Summary table + npz
    # ------------------------------------------------------------------
    print(f"\n{'N':>4}  {'sigma (fid true)':>26}  {'sigma (stoc true)':>26}")
    print("-" * 60)
    summary = {}
    for N in args.n_values:
        s_fid  = result["fid_true"][N]
        s_stoc = result["stoc_true"][N]
        lo_f, med_f, hi_f = np.percentile(s_fid, [2.5, 50, 97.5])
        lo_s, med_s, hi_s = np.percentile(s_stoc, [2.5, 50, 97.5])
        print(f"{N:>4}  {med_f:>8.2f} [{lo_f:.2f},{hi_f:.2f}]      "
              f"{med_s:>8.2f} [{lo_s:.2f},{hi_s:.2f}]")
        summary[N] = dict(fid_true=(lo_f, med_f, hi_f), stoc_true=(lo_s, med_s, hi_s))

    np.savez(
        args.output_dir / f"count_sigma_z{args.redshift}_M{bright_key}_lim{faint_key}.npz",
        n_values=np.array(args.n_values),
        **{f"fid_true_N{N}": result["fid_true"][N] for N in args.n_values},
        **{f"stoc_true_N{N}": result["stoc_true"][N] for N in args.n_values},
        counts_fid=counts_fid, counts_stoc=counts_stoc,
        mean_fid=mean_fid, std_fid=std_fid, mean_stoc=mean_stoc, std_stoc=std_stoc,
        frac_zero_fid=frac_zero_fid, frac_zero_stoc=frac_zero_stoc,
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-ticks")
    plt.rcParams.update({"font.size": 14, "xtick.top": True, "ytick.right": True,
                         "xtick.direction": "in", "ytick.direction": "in"})
    fig, ax = plt.subplots(figsize=(6, 5))
    for true_label, color, label in [("fid_true", "#d94701", "high luminosity true"),
                                      ("stoc_true", "#2171b5", "high stochasticity true")]:
        meds = [np.percentile(result[true_label][N], 50) for N in args.n_values]
        los  = [np.percentile(result[true_label][N], 2.5) for N in args.n_values]
        his  = [np.percentile(result[true_label][N], 97.5) for N in args.n_values]
        ax.plot(args.n_values, meds, "o-", color=color, label=label)
        ax.fill_between(args.n_values, los, his, color=color, alpha=0.2)
    ax.set_xlabel("N (independent pointings)")
    ax.set_ylabel(r"significance [$\sigma$]")
    ax.set_title(f"z={args.redshift}, area={args.area_arcmin2} arcmin$^2$\n"
                 rf"$M_{{\rm UV,0}}={args.muv0}$, $M_{{\rm UV,lim}}={args.muvlim}$"
                 "\n" rf"$\langle n\rangle$ per pointing: fid={mean_fid:.2f}, stoc={mean_stoc:.2f}",
                 fontsize=11)
    ax.legend(fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"count_sigma_z{args.redshift}_M{bright_key}_lim{faint_key}.pdf")
    log.info(f"Saved plot + npz to {args.output_dir}")


if __name__ == "__main__":
    main()
