#!/usr/bin/env python
"""
run_ks.py
---------
Run the KS test analysis on existing d1s cache files and save results + plots.

These particular cache files predate the redshift folder structure so their
paths are set explicitly below.
"""

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
# Paths  —  explicitly set since these predate the z-subfolder structure
# ---------------------------------------------------------------------------
CACHE_DIR  = Path("/groups/astro/ivannik/projects/Neighbors/cache")
OUTPUT_DIR = Path("/groups/astro/ivannik/projects/Neighbors/ks_results")

CACHE_FID  = CACHE_DIR / "d1s_fiducial_idx0.npz"
CACHE_STOC = CACHE_DIR / "d1s_stochastic_idx0.npz"

# ---------------------------------------------------------------------------
# Must match the config used when these files were computed
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

REDSHIFT_LABEL = 10.5   # for plot annotations

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading d1s from cache ...")
    d1s_fid  = load_d1s(CACHE_FID,  cfg)
    d1s_stoc = load_d1s(CACHE_STOC, cfg)

    plt.style.use("seaborn-v0_8-ticks")
    plt.rcParams.update({
        "font.size": 14, "xtick.top": True, "ytick.right": True,
        "xtick.direction": "in", "ytick.direction": "in",
    })

    for bright_key in cfg.bright_names:
        log.info(f"Running KS analysis for bright_key={bright_key} ...")
        t0 = time.perf_counter()

        results = run_ks_analysis(
            d1s_fid, d1s_stoc, cfg, ks_cfg,
            bright_key=bright_key,
            seed=42,
        )
        log.info(f"  Done in {time.perf_counter() - t0:.1f}s")

        # Print summary table
        print(f"\n{'='*48}")
        print(f"bright_key={bright_key}  z={REDSHIFT_LABEL}")
        print(summarise_ks(results, ks_cfg))

        # Histogram plot
        fig = plot_ks_results(
            results, ks_cfg,
            bright_key=bright_key,
            redshift_label=REDSHIFT_LABEL,
        )
        fig.savefig(OUTPUT_DIR / f"ks_hist_{bright_key}_z{REDSHIFT_LABEL}.pdf", bbox_inches="tight")
        plt.close(fig)

        # Bar chart
        fig = plot_ks_summary_bars(
            results, ks_cfg,
            bright_key=bright_key,
            redshift_label=REDSHIFT_LABEL,
        )
        fig.savefig(OUTPUT_DIR / f"ks_bars_{bright_key}_z{REDSHIFT_LABEL}.pdf", bbox_inches="tight")
        plt.close(fig)

        log.info(f"  Plots saved to {OUTPUT_DIR}")

    log.info("All done.")


if __name__ == "__main__":
    main()
