#!/usr/bin/env python
"""
run_required_area.py
---------------------
How large does a survey need to be, at z=14, to have >= --confidence
probability of detecting >= --target-count galaxies brighter than
--muv-threshold -- accounting for both Poisson shot noise and cosmic
(field-to-field) variance, not just the mean expected count.

Motivation: `run_ks.py`'s `compute_required_survey_area` only targets the
*mean* count from the UVLF (e.g. "0.6 deg^2 -> <N>=5"), which says nothing
about how likely a single real survey is to actually see 5. If the true
field-to-field scatter is large, a survey sized to <N>=5 can easily come back
with 2 -- not necessarily evidence against the model. This script instead
bootstraps mock surveys directly from the catalog (Poisson + cosmic variance
both included, since the catalog's own spatial clustering is what's sampled)
and reports P(N >= target_count) as a function of area, growing the area
until that probability crosses --confidence.

Usage
-----
    python run_required_area.py
    python run_required_area.py --target-count 5 --confidence 0.9 --muv-threshold -20.5
    python run_required_area.py --area-min-deg2 0.6 --area-max-deg2 5.0 --n-areas 30
"""

import argparse
import logging

import numpy as np

from galaxy_neighbors import load_halo_catalog
from cosmic_variance import (
    PointingConfig,
    comoving_depth_mpc,
    footprint_side_mpc,
    effective_los_depth,
    build_pointing_pool,
    required_area_for_target_count,
    summarize_required_area,
)
from run_analysis import REDSHIFT_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BOX_REDSHIFT = 14.0
# Matches N_REALIZATIONS[14.0] in run_ks.py.
DEFAULT_N_REALIZATIONS = 60


def parse_args():
    p = argparse.ArgumentParser(
        description="Required survey area for a target detection count at z=14, "
                     "including cosmic variance (not just the mean UVLF count).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--muv-threshold", type=float, default=-20.5,
                   help="M_UV threshold; galaxies with M_UV < this count.")
    p.add_argument("--target-count", type=int, default=5,
                   help="Number of galaxies you want a good chance of detecting.")
    p.add_argument("--confidence", type=float, default=0.9,
                   help="Target probability P(N >= target-count).")
    p.add_argument("--z-lo", type=float, default=13.5, help="Lightcone lower z bound.")
    p.add_argument("--z-hi", type=float, default=14.5, help="Lightcone upper z bound.")
    p.add_argument("--n-realizations", type=int, default=DEFAULT_N_REALIZATIONS,
                   help="Number of MUV realizations to load.")
    p.add_argument("--n-trials", type=int, default=2000,
                   help="Bootstrap trials per area point.")
    p.add_argument("--fov-area-arcmin2", type=float, default=PointingConfig().fov_area_arcmin2,
                   help="Footprint (tile) size used to sample cosmic variance. Keep this "
                        "small (NIRCam-scale) even though the swept areas are much larger "
                        "-- see module docstring in cosmic_variance.py for why a single "
                        "giant footprint can't sample real spatial variance.")
    p.add_argument("--area-min-deg2", type=float, default=0.6,
                   help="Smallest survey area to test [deg^2].")
    p.add_argument("--area-max-deg2", type=float, default=5.0,
                   help="Largest survey area to test [deg^2].")
    p.add_argument("--n-areas", type=int, default=25,
                   help="Number of area points between area-min-deg2 and area-max-deg2 "
                        "(log-spaced).")
    p.add_argument("--model", choices=["fiducial", "stochastic"], default="fiducial",
                   help="Which model's MUV catalog to use.")
    return p.parse_args()


def main():
    args = parse_args()
    z_cfg = REDSHIFT_CONFIGS[BOX_REDSHIFT]
    muv_path = z_cfg.muv_fiducial_path if args.model == "fiducial" else z_cfg.muv_stochastic_path

    cfg = PointingConfig(
        fov_area_arcmin2=args.fov_area_arcmin2,
        thresholds=[args.muv_threshold],
        n_trials=args.n_trials,
    )

    log.info(f"Loading halo catalog: {z_cfg.halo_catalog_path}")
    halo_coords, _ = load_halo_catalog(z_cfg.halo_catalog_path)
    log.info(f"  {len(halo_coords)} halos with M > 0")

    depth_requested = comoving_depth_mpc(args.z_lo, args.z_hi)
    depth, truncated = effective_los_depth(depth_requested, cfg.box_len_mpc)
    log.info(
        f"z=({args.z_lo}, {args.z_hi})  requested depth={depth_requested:.1f} Mpc  "
        f"effective depth={depth:.1f} Mpc" + ("  [TRUNCATED]" if truncated else "")
    )

    footprint_side = footprint_side_mpc(BOX_REDSHIFT, cfg.fov_area_arcmin2)
    log.info(f"Footprint side at z={BOX_REDSHIFT}: {footprint_side:.3f} Mpc "
             f"({cfg.fov_area_arcmin2:.2f} arcmin^2)")

    log.info(f"Building pointing pool: model={args.model}, n_realizations={args.n_realizations} ...")
    pool = build_pointing_pool(
        halo_coords, muv_path, args.n_realizations, cfg, depth, footprint_side,
    )
    log.info(f"  pool shape: {pool.shape}")

    area_grid_arcmin2 = np.geomspace(
        args.area_min_deg2 * 3600.0, args.area_max_deg2 * 3600.0, args.n_areas,
    )

    result = required_area_for_target_count(
        pool, cfg,
        threshold_idx=0,
        target_count=args.target_count,
        confidence=args.confidence,
        area_grid_arcmin2=area_grid_arcmin2,
    )

    print(
        f"\n{'=' * 78}\n"
        f"model={args.model}  z=({args.z_lo}, {args.z_hi})  M_UV<{args.muv_threshold}\n"
    )
    print(summarize_required_area(result, args.target_count, args.confidence))

    if result["required_area_arcmin2"] is None:
        log.warning(
            "No area in the tested grid reached the target confidence -- rerun "
            "with a larger --area-max-deg2."
        )


if __name__ == "__main__":
    main()