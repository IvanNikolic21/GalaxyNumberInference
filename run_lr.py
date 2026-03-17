#!/usr/bin/env python
"""
run_lr.py
---------
Run the likelihood ratio test analysis on existing d1s cache files.
Results cached per (bright_key, redshift, significance) for easy reuse.

Usage
-----
    python run_lr.py --redshift 10.5
    python run_lr.py --redshift 8.0 --significance 0.01
    python run_lr.py --redshift 12.0 --force-recompute
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from galaxy_neighbors import AnalysisConfig, RedshiftConfig, compute_bright_counts
from galaxy_d1s import load_d1s
from galaxy_lr import LRConfig, run_lr_analysis, plot_lr_results, plot_lr_summary_bars, summarise_lr

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redshift registry
# ---------------------------------------------------------------------------
_CACHE_BASE = "/lustre/astro/ivannik/21cmFAST_cache/d12b21e80b7885d62d31717c2c2d8421"
_HASH       = "ffa852ccaa39d8f82951cc98ff798ab4"

REDSHIFT_CONFIGS = {
    8.0:  RedshiftConfig(redshift=8.0,  halo_catalog_path=Path(f"{_CACHE_BASE}/1955/{_HASH}/8.0000/HaloCatalog.h5"),  muv_fiducial_path=Path("/lustre/astro/ivannik/catalog_fiducial_bigger_z8.h5"),          muv_stochastic_path=Path("/lustre/astro/ivannik/catalog_stoch_bigger_z8.h5")),
    10.5: RedshiftConfig(redshift=10.5, halo_catalog_path=Path(f"{_CACHE_BASE}/1952/{_HASH}/10.5000/HaloCatalog.h5"), muv_fiducial_path=Path("/lustre/astro/ivannik/catalog_fiducial_bigger_new_save.h5"), muv_stochastic_path=Path("/lustre/astro/ivannik/catalog_stoch_bigger_new3.h5")),
    12.0: RedshiftConfig(redshift=12.0, halo_catalog_path=Path(f"{_CACHE_BASE}/1955/{_HASH}/12.0000/HaloCatalog.h5"), muv_fiducial_path=Path("/lustre/astro/ivannik/catalog_fiducial_bigger_z12.h5"),        muv_stochastic_path=Path("/lustre/astro/ivannik/catalog_stoch_bigger_z12.h5")),
    14.0: RedshiftConfig(redshift=14.0, halo_catalog_path=Path(f"{_CACHE_BASE}/1955/{_HASH}/14.0000/HaloCatalog.h5"), muv_fiducial_path=Path("/lustre/astro/ivannik/catalog_fiducial_bigger_z14.h5"),    muv_stochastic_path=Path("/lustre/astro/ivannik/catalog_stoch_bigger_z14.h5")),
}

N_REALIZATIONS = {8.0: 1, 10.5: 5, 12.0: 50, 14.0: 100}

CACHE_ROOT = Path("/groups/astro/ivannik/projects/Neighbors/cache")
CACHE_FILES = {
    8.0:  (CACHE_ROOT / "z8.0"  / "d1s_fiducial_real1.npz",   CACHE_ROOT / "z8.0"  / "d1s_stochastic_real1.npz"),
    10.5: (CACHE_ROOT / "z10.5" / "d1s_fiducial_real20.npz",   CACHE_ROOT / "z10.5" / "d1s_stochastic_real20.npz"),
    12.0: (CACHE_ROOT / "z12.0" / "d1s_fiducial_real50.npz",  CACHE_ROOT / "z12.0" / "d1s_stochastic_real50.npz"),
    14.0: (CACHE_ROOT / "z14.0" / "d1s_fiducial_real100.npz", CACHE_ROOT / "z14.0" / "d1s_stochastic_real100.npz"),
}

AVAILABLE_REDSHIFTS = sorted(CACHE_FILES.keys())
OUTPUT_ROOT = Path("/groups/astro/ivannik/projects/Neighbors/lr_results")

# ---------------------------------------------------------------------------
# Analysis config
# ---------------------------------------------------------------------------
cfg = AnalysisConfig(
    bright_limits         = [-20.5, -20.75, -21.0, -21.25, -21.5, -21.75, -22.0],
    faint_limits          = [-16.5, -16.6, -16.7, -16.8, -16.9, -17.0, -17.1, -17.2,
                             -17.3, -17.4, -17.5, -17.6, -17.7, -17.8, -17.9, -18.0,
                             -18.1, -18.2, -18.3, -18.4, -18.5, -18.6, -18.7, -18.8,
                             -18.9, -19.0, -19.1, -19.2, -19.3, -19.4, -19.5, -19.6],
    preselect_faint_limit = -16.5,
    survey_area_arcmin2   = 12.24,
)

# ---------------------------------------------------------------------------
# p_neighbor correction
# ---------------------------------------------------------------------------

def apply_p_neighbor_correction(results, d1s_fid, bright_counts, bright_key):
    n_total = bright_counts[bright_key]
    corrected = {}
    for fkey in results:
        n_passed = len(d1s_fid[bright_key][fkey])
        p = n_passed / n_total if n_total > 0 else 1.0
        if p == 0:
            log.warning(f"    {bright_key} | {fkey}: p_neighbor=0, skipping.")
            corrected[fkey] = np.full_like(results[fkey], np.nan)
            continue
        corrected[fkey] = results[fkey] / p
        log.info(f"    {bright_key} | {fkey}: p_neighbor={p:.3f}  factor={1/p:.2f}")
    return corrected

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Likelihood ratio test analysis for a given redshift",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--redshift", type=float, required=True,
                   choices=AVAILABLE_REDSHIFTS, metavar="Z",
                   help=f"Redshift. Available: {AVAILABLE_REDSHIFTS}")
    p.add_argument("--significance", type=float, default=0.05,
                   help="False positive rate for LR threshold calibration.")
    p.add_argument("--n-trials", type=int, default=2000,
                   help="Number of bootstrap trials per faint limit.")
    p.add_argument("--max-sample", type=int, default=100,
                   help="Maximum sample size per trial.")
    p.add_argument("--n-null-bootstrap", type=int, default=1000,
                   help="Samples used to calibrate null LR threshold.")
    p.add_argument("--bw-method", type=str, default="scott",
                   help="KDE bandwidth method.")
    p.add_argument("--force-recompute", action="store_true",
                   help="Ignore existing cache and recompute from scratch.")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    z    = args.redshift
    z_cfg = REDSHIFT_CONFIGS[z]

    lr_cfg = LRConfig(
        n_trials          = args.n_trials,
        max_sample        = args.max_sample,
        significance      = args.significance,
        n_null_bootstrap  = args.n_null_bootstrap,
        bw_method         = args.bw_method,
    )

    sig_str    = str(args.significance).replace(".", "p")
    cache_fid, cache_stoc = CACHE_FILES[z]
    output_dir = OUTPUT_ROOT / f"z{z}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Redshift: {z}")
    log.info(f"Significance: {args.significance}")
    log.info(f"Output dir: {output_dir}")

    log.info("Loading d1s ...")
    d1s_fid  = load_d1s(cache_fid,  cfg)
    d1s_stoc = load_d1s(cache_stoc, cfg)

    log.info("Computing bright counts for p_neighbor correction ...")
    bright_counts = compute_bright_counts(z_cfg, cfg, n_realizations=N_REALIZATIONS[z])
    for bkey, n in bright_counts.items():
        log.info(f"  {bkey}: {n} bright galaxies")

    plt.style.use("seaborn-v0_8-ticks")
    plt.rcParams.update({"font.size": 14, "xtick.top": True, "ytick.right": True,
                         "xtick.direction": "in", "ytick.direction": "in"})

    for bright_key in cfg.bright_names:
        cache_path = output_dir / f"lr_results_{bright_key}_z{z}_sig{sig_str}.npz"

        if cache_path.exists() and not args.force_recompute:
            log.info(f"  {bright_key}: loading from cache ...")
            archive = np.load(cache_path)
            results = {fkey: archive[fkey] for fkey in cfg.faint_names if fkey in archive}
        else:
            log.info(f"  {bright_key}: running LR analysis ...")
            t0 = time.perf_counter()
            results = run_lr_analysis(
                d1s_fid, d1s_stoc, cfg, lr_cfg,
                bright_key=bright_key,
                seed=42,
            )
            log.info(f"  Done in {time.perf_counter() - t0:.1f}s")
            np.savez(cache_path, **{fkey: results[fkey] for fkey in results})

        # p_neighbor correction
        results_corrected = apply_p_neighbor_correction(
            results, d1s_fid, bright_counts, bright_key
        )

        # Summary
        print(f"\n{'='*68}")
        print(f"bright_key={bright_key}  z={z}  sig={args.significance}  (p_neighbor corrected)")
        print(summarise_lr(results_corrected, lr_cfg))

        # Plots
        fig = plot_lr_results(results_corrected, lr_cfg,
                              bright_key=bright_key, redshift_label=z)
        fig.savefig(output_dir / f"lr_hist_{bright_key}_z{z}_sig{sig_str}.pdf",
                    bbox_inches="tight")
        plt.close(fig)

        fig = plot_lr_summary_bars(results_corrected, lr_cfg,
                                   bright_key=bright_key, redshift_label=z)
        fig.savefig(output_dir / f"lr_bars_{bright_key}_z{z}_sig{sig_str}.pdf",
                    bbox_inches="tight")
        plt.close(fig)

        log.info(f"  Plots saved for {bright_key}")

    log.info("All done.")


if __name__ == "__main__":
    main()