#!/usr/bin/env python
"""
build_reference_mock_obs.py
-----------------------------
Build ONE mock observation at a given theta, saved in the standard
coords/offsets/params .npz format infer_nre.py's --obs-file expects.
Reuses build_mock_obs() from run_sbc_one_truth.py directly (same forward
model as the rest of the pipeline) rather than reimplementing it.

Needed as --obs-file for a --uvlf-only posterior run: that run discards the
environment content anyway, but infer_nre.py still requires a valid file to
load (for the `params` array used to label the truth on the corner plot).

Usage
-----
    python build_reference_mock_obs.py --truth 0.3 -0.34 0.6 \\
        --output /groups/astro/ivannik/projects/Neighbors/ref_obs_highstoch.npz
"""
import argparse
import logging

import numpy as np
from scipy.spatial import cKDTree

from galaxy_neighbors import load_halo_catalog
from generate_catalog_database import load_muv_mh_dict
from run_sbc_one_truth import build_mock_obs, HALO_CATALOG_PATH, MUV_MH_FILE, cfg, REDSHIFT

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth", type=float, nargs=3, required=True,
                   metavar=("Muv_add", "sigmaUV_a", "sigmaUV_b"))
    p.add_argument("--n-obs", type=int, default=50,
                   help="Number of bright-galaxy environments to include.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    theta = np.array(args.truth)
    rng = np.random.default_rng(args.seed)

    log.info("Loading halo catalog + Muv-Mh relation ...")
    halo_coords, logmhs = load_halo_catalog(HALO_CATALOG_PATH)
    muv_mh_dict = load_muv_mh_dict(MUV_MH_FILE)
    log.info("Building 2D cKDTree ...")
    halo_tree_2d = cKDTree(halo_coords[:, :2])
    half_side = cfg.search_box_mpc(REDSHIFT)

    built = build_mock_obs(theta, halo_coords, logmhs, halo_tree_2d, muv_mh_dict, half_side,
                            n_obs_needed=args.n_obs, rng=rng, seed=args.seed)
    if built is None:
        log.error(f"theta={theta} produced zero usable environments -- cannot build mock obs.")
        return
    coords_flat, offsets_arr, n_bright_true = built

    np.savez(args.output, coords=coords_flat, offsets=offsets_arr,
             params=theta.astype(np.float64), n_bright_true=n_bright_true)
    log.info(f"Saved: {args.output}  ({len(offsets_arr) - 1} environments, "
             f"{n_bright_true} bright galaxies available)")


if __name__ == "__main__":
    main()
