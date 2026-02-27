#!/usr/bin/env python
"""
run_ks.py
---------
Run the KS/AD test analysis on existing d1s cache files for a given redshift.

Usage
-----
    python run_ks.py --redshift 10.5
    python run_ks.py --redshift 8.0
    python run_ks.py --redshift 12.0
    python run_ks.py --redshift 14.0
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from galaxy_neighbors import AnalysisConfig
from galaxy_d1s import load_d1s
from galaxy_ks import KSConfig, run_ks_analysis, plot_ks_results, plot_ks_summary_bars, summarise_ks

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
# Cache file registry — one entry per redshift
# ---------------------------------------------------------------------------
CACHE_ROOT = Path("/groups/astro/ivannik/projects/Neighbors/cache")

CACHE_FILES = {
    8.0:  (
        CACHE_ROOT / "z8.0"  / "d1s_fiducial_real1.npz",
        CACHE_ROOT / "z8.0"  / "d1s_stochastic_real1.npz",
    ),
    10.5: (
        CACHE_ROOT / "z10.5" / "d1s_fiducial_real5.npz",
        CACHE_ROOT / "z10.5" / "d1s_stochastic_real5.npz",
    ),
    12.0: (
        CACHE_ROOT / "z12.0" / "d1s_fiducial_real50.npz",
        CACHE_ROOT / "z12.0" / "d1s_stochastic_real50.npz",
    ),
    14.0: (
        CACHE_ROOT / "z14.0" / "d1s_fiducial_real100.npz",
        CACHE_ROOT / "z14.0" / "d1s_stochastic_real100.npz",
    ),
}

AVAILABLE_REDSHIFTS = sorted(CACHE_FILES.keys())
OUTPUT_ROOT = Path("/groups/astro/ivannik/projects/Neighbors/ks_results")

# ---------------------------------------------------------------------------
# Must match the config used when the cache files were computed
# ---------------------------------------------------------------------------
cfg = AnalysisConfig(
    bright_limits         = [-20.5, -20.75, -21.0, -21.25, -21.5, -21.75, -22.0],
    faint_limits          = [-17.0, -17.1, -17.2, -17.3, -17.4, -17.5, -17.6, -17.7,
                             -17.8, -17.9, -18.0, -18.1, -18.2, -18.3, -18.4, -18.5,
                             -18.6, -18.7, -18.8, -18.9, -19.0, -19.1, -19.2],
    preselect_faint_limit = -17.0,
    survey_area_arcmin2   = 12.24,
)

ks_cfg = KSConfig(
    n_trials           = 2000,
    max_sample         = 100,
    significance       = 0.05,
    summary_percentile = 90.0,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="KS/AD test analysis for a given redshift")
    p.add_argument(
        "--redshift", type=float, required=True,
        choices=AVAILABLE_REDSHIFTS, metavar="Z",
        help=f"Redshift to analyse. Available: {AVAILABLE_REDSHIFTS}",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    z = args.redshift

    cache_fid, cache_stoc = CACHE_FILES[z]
    output_dir = OUTPUT_ROOT / f"z{z}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Redshift: {z}")
    log.info(f"Fiducial cache:   {cache_fid}")
    log.info(f"Stochastic cache: {cache_stoc}")
    log.info(f"Output dir:       {output_dir}")

    log.info("Loading d1s from cache ...")
    d1s_fid  = load_d1s(cache_fid,  cfg)
    d1s_stoc = load_d1s(cache_stoc, cfg)

    plt.style.use("seaborn-v0_8-ticks")
    plt.rcParams.update({
        "font.size": 14, "xtick.top": True, "ytick.right": True,
        "xtick.direction": "in", "ytick.direction": "in",
    })

    for bright_key in cfg.bright_names:
        log.info(f"Running KS/AD analysis: bright_key={bright_key} ...")
        t0 = time.perf_counter()

        results = run_ks_analysis(
            d1s_fid, d1s_stoc, cfg, ks_cfg,
            bright_key=bright_key,
            seed=42,
        )
        log.info(f"  Done in {time.perf_counter() - t0:.1f}s")

        print(f"\n{'='*68}")
        print(f"bright_key={bright_key}  z={z}")
        print(summarise_ks(results, ks_cfg))

        fig = plot_ks_results(
            results, ks_cfg, bright_key=bright_key, redshift_label=z,
        )
        fig.savefig(output_dir / f"ks_hist_{bright_key}_z{z}.pdf", bbox_inches="tight")
        plt.close(fig)

        fig = plot_ks_summary_bars(
            results, ks_cfg, bright_key=bright_key, redshift_label=z,
        )
        fig.savefig(output_dir / f"ks_bars_{bright_key}_z{z}.pdf", bbox_inches="tight")
        plt.close(fig)

        log.info(f"  Saved plots for {bright_key}")

    log.info("All done.")


if __name__ == "__main__":
    main()
