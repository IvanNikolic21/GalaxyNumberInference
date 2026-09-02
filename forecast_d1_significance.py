#!/usr/bin/env python
"""
forecast_d1_significance.py
-----------------------------
Companion to forecast_count_significance.py, for the JWST proposal
,instead of raw neighbor counts, forecast the MEAN distance to the
nearest neighbor (d1) as a function of the number of pointings N, 
with bootstrap error bars. Design target - the plot this script makes is
exactly what's needed to check that by eye.

Reuses galaxy_d1s.compute_d1s directly (same min_neighbors filtering
convention as run_analysis.py/run_ks.py) rather than re-deriving d1 from
scratch, so this is methodologically consistent with the rest of Paper I,
not a parallel reimplementation.

For each of --n-trials bootstrap trials and each N in --n-values:
    draw N d1 values (with replacement) from the model's pooled, filtered
    d1 array, take their mean.
This gives, per model per N, a distribution of "mean d1 at that N" --
its spread IS the bootstrap error. Also reports a
derived separation-in-sigma = |mean_fid - mean_stoc| / sqrt(std_fid^2 +
std_stoc^2), for direct comparison with forecast_count_significance.py's
log-likelihood-ratio sigma (not the primary deliverable here, but useful
context since both scripts should tell a consistent story).

All physical/observational choices are CLI flags. See the aperture-
convention note in forecast_count_significance.py's docstring -- the same
4x-area caveat applies here (both scripts route through
AnalysisConfig.survey_area_arcmin2 -> search_box_mpc identically).

Usage
-----
    python forecast_d1_significance.py \\
        --redshift 14.0 --area-arcmin2 4.84 --muv0 -20.5 --muvlim -18.0 \\
        --n-values 1 2 5 --n-realizations 100 --n-trials 2000 \\
        --output-dir /groups/astro/ivannik/projects/Neighbors/d1_forecast
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from galaxy_neighbors import AnalysisConfig, run_neighbor_analysis, _mag_to_key
from galaxy_d1s import D1sConfig, compute_d1s
from run_ks import REDSHIFT_CONFIGS, N_REALIZATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def available_realizations(path: Path) -> int:
    """Actual number of realizations stored in a muv catalog's 'data' dataset --
    same guard as forecast_count_significance.py, see that script for why."""
    import h5py
    with h5py.File(path, "r") as f:
        return f["data"].shape[0]


def bootstrap_mean_vs_n(d1_arr: np.ndarray, n_values: list[int], n_trials: int,
                         rng: np.random.Generator) -> dict[int, np.ndarray]:
    """For each N, return an array of shape (n_trials,) of bootstrap sample means."""
    out = {}
    for N in n_values:
        means = np.empty(n_trials)
        for t in range(n_trials):
            draw = rng.choice(d1_arr, size=N, replace=True)
            means[t] = draw.mean()
        out[N] = means
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--redshift", type=float, required=True, choices=sorted(REDSHIFT_CONFIGS.keys()))
    p.add_argument("--area-arcmin2", type=float, required=True)
    p.add_argument("--muv0", type=float, required=True)
    p.add_argument("--muvlim", type=float, required=True)
    p.add_argument("--min-neighbors", type=int, default=1,
                   help="Passed to D1sConfig -- environments with fewer detected neighbors than "
                        "this are excluded from the d1 distribution entirely (same convention as "
                        "run_analysis.py). Default 1: at least one neighbor required to have a d1.")
    p.add_argument("--n-realizations", type=int, default=None)
    p.add_argument("--n-values", type=int, nargs="+", default=[1, 2, 5])
    p.add_argument("--n-trials", type=int, default=2000)
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
    d1s_cfg = D1sConfig(min_neighbors=args.min_neighbors)
    bright_key, faint_key = _mag_to_key(args.muv0), _mag_to_key(args.muvlim)

    log.info(f"z={args.redshift}  area={args.area_arcmin2} arcmin^2  "
             f"M_UV,0={args.muv0}  M_UV,lim={args.muvlim}  n_realizations={n_realizations}")
    log.info("Running neighbor search ...")
    results_fid, results_stoc = run_neighbor_analysis(
        redshift_cfg=z_cfg, analysis_cfg=cfg, n_realizations=n_realizations,
    )
    d1s_fid  = compute_d1s(results_fid,  cfg, z_cfg, d1s_cfg)
    d1s_stoc = compute_d1s(results_stoc, cfg, z_cfg, d1s_cfg)
    d1_fid  = d1s_fid[bright_key][faint_key]
    d1_stoc = d1s_stoc[bright_key][faint_key]
    log.info(f"  fiducial:   {len(d1_fid)} usable environments (>= {args.min_neighbors} neighbor(s)), "
             f"mean d1={d1_fid.mean():.3f} cMpc, std={d1_fid.std():.3f}")
    log.info(f"  stochastic: {len(d1_stoc)} usable environments (>= {args.min_neighbors} neighbor(s)), "
             f"mean d1={d1_stoc.mean():.3f} cMpc, std={d1_stoc.std():.3f}")

    log.info(f"Bootstrapping mean d1 vs N for N={args.n_values}, {args.n_trials} trials each ...")
    boot_fid  = bootstrap_mean_vs_n(d1_fid,  args.n_values, args.n_trials, rng)
    boot_stoc = bootstrap_mean_vs_n(d1_stoc, args.n_values, args.n_trials, rng)

    print(f"\nMean d1 [cMpc] vs N, bootstrapped ({args.n_trials} trials/N), "
          f"z={args.redshift}, M_UV,0={args.muv0}, M_UV,lim={args.muvlim}, area={args.area_arcmin2} arcmin^2")
    print(f"{'N':>4}  {'fiducial: median [16,84]':>28}  {'stochastic: median [16,84]':>28}  "
          f"{'overlap?':>10}  {'separation [sigma]':>18}")
    print("-" * 96)
    summary = {}
    for N in args.n_values:
        f, s = boot_fid[N], boot_stoc[N]
        lo_f, med_f, hi_f = np.percentile(f, [16, 50, 84])
        lo_s, med_s, hi_s = np.percentile(s, [16, 50, 84])
        overlap = not (hi_f < lo_s or hi_s < lo_f)  # 68% bands overlap?
        sep_sigma = abs(med_f - med_s) / np.sqrt(f.std() ** 2 + s.std() ** 2)
        print(f"{N:>4}  {med_f:>8.3f} [{lo_f:.3f},{hi_f:.3f}]      "
              f"{med_s:>8.3f} [{lo_s:.3f},{hi_s:.3f}]      "
              f"{'yes' if overlap else 'no':>10}  {sep_sigma:>18.2f}")
        summary[N] = dict(fid=(lo_f, med_f, hi_f), stoc=(lo_s, med_s, hi_s),
                           overlap=overlap, sep_sigma=sep_sigma)

    np.savez(
        args.output_dir / f"d1_meanboot_z{args.redshift}_M{bright_key}_lim{faint_key}.npz",
        n_values=np.array(args.n_values),
        **{f"fid_N{N}": boot_fid[N] for N in args.n_values},
        **{f"stoc_N{N}": boot_stoc[N] for N in args.n_values},
        d1_fid=d1_fid, d1_stoc=d1_stoc,
    )

    # ------------------------------------------------------------------
    # Plot -- exactly the "does N=2 overlap, N=5 separate" check
    # ------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-ticks")
    plt.rcParams.update({"font.size": 14, "xtick.top": True, "ytick.right": True,
                         "xtick.direction": "in", "ytick.direction": "in"})
    fig, ax = plt.subplots(figsize=(6, 5))
    for boot, color, label in [(boot_fid, "#d94701", "high luminosity"),
                                (boot_stoc, "#2171b5", "high stochasticity")]:
        meds = [np.percentile(boot[N], 50) for N in args.n_values]
        los  = [np.percentile(boot[N], 16) for N in args.n_values]
        his  = [np.percentile(boot[N], 84) for N in args.n_values]
        ax.errorbar(args.n_values, meds,
                     yerr=[np.array(meds) - np.array(los), np.array(his) - np.array(meds)],
                     fmt="o-", color=color, label=label, capsize=4)
    ax.set_xlabel("N (independent pointings)")
    ax.set_ylabel(r"mean $d_1$ [cMpc]")
    ax.set_xticks(args.n_values)
    ax.set_title(f"z={args.redshift}, area={args.area_arcmin2} arcmin$^2$\n"
                 rf"$M_{{\rm UV,0}}={args.muv0}$, $M_{{\rm UV,lim}}={args.muvlim}$", fontsize=11)
    ax.legend(fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"d1_meanboot_z{args.redshift}_M{bright_key}_lim{faint_key}.pdf")
    log.info(f"Saved plot + npz to {args.output_dir}")


if __name__ == "__main__":
    main()