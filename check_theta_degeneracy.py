#!/usr/bin/env python
"""
check_theta_degeneracy.py
--------------------------
Network-independent check: are the two theta points that the balanced-NRE
posterior's "bimodal" structure sits on (main mode vs. the pile-up against
the Muv_add prior boundary, see repro_balanced_sb2p4_N50) actually physically
degenerate -- i.e. do they produce statistically indistinguishable d1
(nearest-neighbor distance) distributions on the same halo catalog -- or are
they genuinely different, in which case the network *should* have been able
to tell them apart and the bimodality points to a network/architecture
shortfall instead?

Uses the same fixed halo catalog, UV-magnitude assignment, and box-search
machinery as build_nre_database.py / galaxy_neighbors.py -- no network,
no training, just direct forward simulation + a KS test.

Usage
-----
    python check_theta_degeneracy.py
    python check_theta_degeneracy.py --n-realizations 10 --max-bright-per-realization 3000
"""

import argparse
import logging

import numpy as np
from scipy.stats import ks_2samp

from galaxy_neighbors import load_halo_catalog, AnalysisConfig, RedshiftConfig, find_neighbors_in_box
from generate_catalog_database import load_muv_mh_dict, sample_muv

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Must match build_nre_database.py exactly, so this check uses the identical
# forward model / survey setup as the actual NRE training and inference.
HALO_CATALOG_PATH = "/lustre/astro/ivannik/21cmFAST_cache/d12b21e80b7885d62d31717c2c2d8421/1952/ffa852ccaa39d8f82951cc98ff798ab4/10.5000/HaloCatalog.h5"
MUV_MH_FILE       = "/groups/astro/ivannik/notebooks/clustering_project/Muv_Mh_z=10.txt"
BRIGHT_LIMIT      = -21.5
FAINT_LIMIT       = -18.5
REDSHIFT          = 10.5

# Data-driven from the actual repro_balanced_sb2p4_N50 posterior samples
# (2026-08-25): mode A = the main peak, mode B = the pile-up against the
# Muv_add prior's upper boundary (+2.0). Override with --theta-a/--theta-b
# if re-deriving these from a different/future run.
DEFAULT_THETA_A = (-0.70, -0.34, 1.67)   # main mode
DEFAULT_THETA_B = (+1.86, -0.86, 2.10)   # boundary pile-up


def d1_distribution_for_theta(theta, logmhs, coords, muv_mh_dict, cfg, half_side,
                               n_realizations, max_bright_per_realization, rng):
    """Forward-simulate n_realizations independent Muv draws at this theta on
    the fixed halo catalog, and return the pooled array of d1 (nearest
    faint-neighbor distance) values across all bright galaxies in all
    realizations (subsampled to max_bright_per_realization each, for speed)."""
    Muv_add, sigmaUV_a, sigmaUV_b = theta
    all_d1 = []
    n_bright_per_real = []

    for r in range(n_realizations):
        muvs = sample_muv(logmhs, muv_mh_dict, Muv_add, sigmaUV_a, sigmaUV_b)
        bright_mask = muvs < BRIGHT_LIMIT
        faint_mask  = (muvs < FAINT_LIMIT) & (muvs >= BRIGHT_LIMIT)
        bright_idx  = np.where(bright_mask)[0]
        n_bright_per_real.append(len(bright_idx))

        if len(bright_idx) > max_bright_per_realization:
            bright_idx = rng.choice(bright_idx, size=max_bright_per_realization, replace=False)

        faint_coords = coords[faint_mask]
        faint_mags   = muvs[faint_mask]

        for bi in bright_idx:
            _, _, distances = find_neighbors_in_box(
                coords[bi], faint_coords, faint_mags, half_side, FAINT_LIMIT
            )
            if len(distances) > 0:
                all_d1.append(distances.min())

    return np.array(all_d1), n_bright_per_real


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--theta-a", type=float, nargs=3, default=DEFAULT_THETA_A,
                   metavar=("MUV_ADD", "SIGMA_A", "SIGMA_B"))
    p.add_argument("--theta-b", type=float, nargs=3, default=DEFAULT_THETA_B,
                   metavar=("MUV_ADD", "SIGMA_A", "SIGMA_B"))
    p.add_argument("--n-realizations", type=int, default=10,
                   help="Independent Muv noise draws per theta, pooled together for the KS test.")
    p.add_argument("--max-bright-per-realization", type=int, default=3000,
                   help="Subsample cap per realization -- both test points are in the "
                        "high-sigma_b regime where bright-galaxy counts can reach ~1e5, "
                        "so this keeps runtime reasonable without biasing the d1 distribution "
                        "shape (subsample is uniform random).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    log.info(f"Theta A (main mode):        Muv_add={args.theta_a[0]:+.3f}  "
             f"sigmaUV_a={args.theta_a[1]:+.3f}  sigmaUV_b={args.theta_a[2]:+.3f}")
    log.info(f"Theta B (boundary pile-up): Muv_add={args.theta_b[0]:+.3f}  "
             f"sigmaUV_a={args.theta_b[1]:+.3f}  sigmaUV_b={args.theta_b[2]:+.3f}")

    log.info("Loading halo catalog and Muv-Mh relation ...")
    coords, logmhs = load_halo_catalog(HALO_CATALOG_PATH)
    muv_mh_dict = load_muv_mh_dict(MUV_MH_FILE)

    cfg = AnalysisConfig(
        bright_limits=[BRIGHT_LIMIT], faint_limits=[FAINT_LIMIT],
        preselect_faint_limit=FAINT_LIMIT, survey_area_arcmin2=12.24,
    )
    half_side = cfg.search_box_mpc(REDSHIFT)

    log.info(f"Simulating theta A ({args.n_realizations} realizations) ...")
    d1_a, nbright_a = d1_distribution_for_theta(
        args.theta_a, logmhs, coords, muv_mh_dict, cfg, half_side,
        args.n_realizations, args.max_bright_per_realization, rng,
    )
    log.info(f"  Bright galaxies/realization: mean={np.mean(nbright_a):.0f} "
             f"(range {min(nbright_a)}-{max(nbright_a)})")
    log.info(f"  Pooled d1 samples: {len(d1_a)}  median d1={np.median(d1_a):.3f} cMpc")

    log.info(f"Simulating theta B ({args.n_realizations} realizations) ...")
    d1_b, nbright_b = d1_distribution_for_theta(
        args.theta_b, logmhs, coords, muv_mh_dict, cfg, half_side,
        args.n_realizations, args.max_bright_per_realization, rng,
    )
    log.info(f"  Bright galaxies/realization: mean={np.mean(nbright_b):.0f} "
             f"(range {min(nbright_b)}-{max(nbright_b)})")
    log.info(f"  Pooled d1 samples: {len(d1_b)}  median d1={np.median(d1_b):.3f} cMpc")

    stat, pvalue = ks_2samp(d1_a, d1_b)
    log.info("")
    log.info("=== KS test: are theta A's and theta B's d1 distributions distinguishable? ===")
    log.info(f"  KS statistic D = {stat:.4f}   p-value = {pvalue:.3e}")
    log.info("")
    if pvalue > 0.05:
        log.info("  ==> NOT significantly different (p > 0.05). The two theta points genuinely "
                 "predict near-indistinguishable environments given this many bright galaxies -- "
                 "the network's bimodal/boundary-pinned posterior may be correctly reporting a "
                 "real degeneracy, not a shortfall. Consider this a caveat to report, not a bug "
                 "to fix in the network.")
    else:
        log.info("  ==> SIGNIFICANTLY different (p <= 0.05). The two theta points predict "
                 "physically distinguishable environments, so the network should in principle "
                 "be able to tell them apart. The posterior's bimodal/boundary structure is "
                 "then more likely a network/training shortfall -- worth pursuing the "
                 "permutation-invariant architecture direction, or more targeted training "
                 "coverage in this region of the prior.")

    np.savez("theta_degeneracy_check_results.npz",
             theta_a=args.theta_a, theta_b=args.theta_b,
             d1_a=d1_a, d1_b=d1_b, ks_stat=stat, ks_pvalue=pvalue,
             nbright_a=nbright_a, nbright_b=nbright_b)
    log.info("Saved: theta_degeneracy_check_results.npz")


if __name__ == "__main__":
    main()
