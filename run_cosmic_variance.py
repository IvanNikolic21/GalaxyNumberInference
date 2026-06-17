

#!/usr/bin/env python
"""
run_cosmic_variance.py
-----------------------
Estimate the cosmic (field-to-field) variance of galaxy number counts in
NIRCam-sized pointings, for the fiducial and stochastic models, built from
the single z=10.5 coeval box.

Two assumed lightcone depths are evaluated, both centered such that the box
is treated as representative of the whole range:
    z9p2_10p9 :  9.2 < z < 10.9   (depth ~368 cMpc, fits inside the box)
    z8p6_11p3 :  8.6 < z < 11.3   (depth ~596 cMpc, truncated to one box
                                    length — see cosmic_variance.effective_los_depth)

Usage
-----
    python run_cosmic_variance.py
    python run_cosmic_variance.py --n-realizations 50 --n-trials 5000
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from galaxy_neighbors import load_halo_catalog
from cosmic_variance import (
    PointingConfig,
    comoving_depth_mpc,
    footprint_side_mpc,
    effective_los_depth,
    build_pointing_pool,
    bootstrap_group_stats,
    bootstrap_fractional_cosmic_variance,
    summarize_group_stats,
    pool_moments,
    summarize_pool_moments,
    plot_fractional_cosmic_variance,
    save_cosmic_variance,
    load_cosmic_variance,
)
from run_analysis import REDSHIFT_CONFIGS

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
# Redshift ranges to evaluate — both lightcones are built from the z=10.5 box
# ---------------------------------------------------------------------------
BOX_REDSHIFT = 10.5

Z_RANGES = {
    "z9p2_10p9": (9.2, 10.9),
    "z8p6_11p3": (8.6, 11.3),
}

CACHE_DIR = Path("/groups/astro/ivannik/projects/Neighbors/cache/cosmic_variance")
OUTPUT_DIR = Path("/groups/astro/ivannik/projects/Neighbors/cosmic_variance_plots")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Cosmic variance of NIRCam pointings (fiducial vs stochastic)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n-realizations", type=int, default=100,
        help="Number of MUV realizations to load per model.",
    )
    p.add_argument(
        "--n-trials", type=int, default=2000,
        help="Number of bootstrap trials (surveys of --group-size pointings).",
    )
    p.add_argument(
        "--group-size", type=int, default=28,
        help="Number of pointings per bootstrap survey draw.",
    )
    p.add_argument(
        "--force-recompute", action="store_true",
        help="Ignore existing cache and recompute from scratch.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_summaries(results: dict, cfg: PointingConfig) -> None:
    for model, by_zrange in results.items():
        for zrange_label, (means, varis, mean, mean_sq, sigma_cv2_trials) in by_zrange.items():
            zlo, zhi = Z_RANGES[zrange_label]
            print(f"\n{'=' * 78}\nmodel={model}  z-range=({zlo}, {zhi})  [{zrange_label}]")
            print(summarize_group_stats(means, varis, cfg.thresholds))
            print(f"\n  Per-pointing moments & Poisson/cosmic variance decomposition:")
            print(summarize_pool_moments(mean, mean_sq, cfg.thresholds, group_size=cfg.group_size))


def _save_plot(results: dict, cfg: PointingConfig, plot_path: Path) -> None:
    fig = plot_fractional_cosmic_variance(results, cfg.thresholds, Z_RANGES)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    log.info(f"Saved plot: {plot_path}")


def main():
    args = parse_args()
    z_cfg = REDSHIFT_CONFIGS[BOX_REDSHIFT]

    cfg = PointingConfig(
        group_size=args.group_size,
        n_trials=args.n_trials,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"cosmic_variance_real{args.n_realizations}_trials{args.n_trials}_g{args.group_size}.npz"
    plot_path = OUTPUT_DIR / f"sigma_cv2_vs_Muv_real{args.n_realizations}_trials{args.n_trials}_g{args.group_size}.pdf"

    if cache_path.exists() and not args.force_recompute:
        log.info(f"Cache found, loading: {cache_path}")
        results = load_cosmic_variance(cache_path)
        _print_summaries(results, cfg)
        _save_plot(results, cfg, plot_path)
        return

    log.info(f"Loading halo catalog: {z_cfg.halo_catalog_path}")
    halo_coords, _ = load_halo_catalog(z_cfg.halo_catalog_path)
    log.info(f"  {len(halo_coords)} halos with M > 0")

    footprint_side = footprint_side_mpc(BOX_REDSHIFT, cfg.fov_arcmin)
    log.info(f"NIRCam footprint side at z={BOX_REDSHIFT}: {footprint_side:.3f} Mpc")

    muv_paths = {
        "fiducial": z_cfg.muv_fiducial_path,
        "stochastic": z_cfg.muv_stochastic_path,
    }

    results: dict = {model: {} for model in muv_paths}

    for zrange_label, (zlo, zhi) in Z_RANGES.items():
        depth_requested = comoving_depth_mpc(zlo, zhi)
        depth, truncated = effective_los_depth(depth_requested, cfg.box_len_mpc)
        log.info(
            f"[{zrange_label}] z=({zlo}, {zhi})  requested depth={depth_requested:.1f} Mpc  "
            f"effective depth={depth:.1f} Mpc"
            + ("  [TRUNCATED]" if truncated else "")
        )

        for model, muv_path in muv_paths.items():
            log.info(f"  Building pointing pool: model={model}, n_realizations={args.n_realizations} ...")
            pool = build_pointing_pool(
                halo_coords, muv_path, args.n_realizations, cfg, depth, footprint_side,
            )
            log.info(f"    pool shape: {pool.shape}")

            means, varis = bootstrap_group_stats(pool, cfg)
            mean, mean_sq = pool_moments(pool)
            sigma_cv2_trials = bootstrap_fractional_cosmic_variance(pool, cfg)
            results[model][zrange_label] = (means, varis, mean, mean_sq, sigma_cv2_trials)

    _print_summaries(results, cfg)
    save_cosmic_variance(cache_path, results)
    _save_plot(results, cfg, plot_path)
    log.info("All done.")


if __name__ == "__main__":
    main()