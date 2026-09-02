#!/usr/bin/env python
"""
run_sbc_one_truth.py
---------------------
One SBC test: build a fresh mock observation at a single test truth, run
NRE inference on it, thin the posterior, and record the per-parameter rank
statistic. Designed to be called once per SLURM array task (see
run_sbc_slurm.sh), one truth-index at a time.

The mock observation is built with the exact same forward model as
build_nre_database.py (same halo catalog, same BRIGHT_LIMIT/FAINT_LIMIT/
REDSHIFT/MAX_ENV_NEIGHBORS/box-search algorithm, imported directly from
that module so there's no risk of a silent mismatch) -- but using
sample_muv() to draw UV magnitudes on the fly for an arbitrary continuous
theta, the same cheap forward-simulation approach already used in
check_theta_degeneracy.py, rather than requiring a pre-built catalog file
to exist in catalogs_grid_prior for this exact theta (SBC test truths are
continuous draws from the UVLF-only posterior, not grid points).

Rank statistic: thin the returned posterior to --n-thin samples (random
subsample, not stride-thinning -- the MCMC output is a flattened,
autocorrelated emcee chain, and a modest random subsample avoids assuming
anything about the walker/step ordering), then per parameter:
    rank = count(thinned_samples[:, p] < theta[p])
An integer in [0, n_thin]. Under correct calibration, ranks pooled across
many test truths should be uniform on {0, ..., n_thin} (Talts et al. 2018).

Usage
-----
    python run_sbc_one_truth.py \\
        --truths-file /groups/astro/ivannik/projects/Neighbors/sbc/sbc_truths.npy \\
        --truth-index 0 \\
        --model-dir /groups/astro/ivannik/projects/Neighbors/nre_model_balanced_only_ang_only_ang \\
        --n-obs 50 --n-thin 20 \\
        --output-dir /groups/astro/ivannik/projects/Neighbors/sbc
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from galaxy_neighbors import load_halo_catalog
from generate_catalog_database import load_muv_mh_dict, sample_muv
import build_nre_database as bdb  # re-use its exact forward-model constants, single source of truth

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

HALO_CATALOG_PATH = bdb.HALO_CATALOG_PATH
BRIGHT_LIMIT       = bdb.BRIGHT_LIMIT
FAINT_LIMIT        = bdb.FAINT_LIMIT
REDSHIFT           = bdb.REDSHIFT
MAX_ENV_NEIGHBORS  = bdb.MAX_ENV_NEIGHBORS
cfg                = bdb.cfg

# build_nre_database.py doesn't define MUV_MH_FILE (it reads from pre-built
# catalogs, never calls sample_muv itself) -- same path already used by
# check_theta_degeneracy.py for the identical purpose.
MUV_MH_FILE = "/groups/astro/ivannik/notebooks/clustering_project/Muv_Mh_z=10.txt"


def build_mock_obs(theta, halo_coords, logmhs, halo_tree_2d, muv_mh_dict, half_side,
                    n_obs_needed, rng, seed):
    """Cheaply forward-simulate one mock observation at `theta`, formatted
    exactly like a build_nre_database.py output file (coords/offsets/params).

    Returns (coords_flat, offsets_arr, n_bright_true) or None if theta
    produces zero bright galaxies (can happen at the low-sigma_b edge of
    the prior).
    """
    Muv_add, sigmaUV_a, sigmaUV_b = theta
    muvs = sample_muv(logmhs, muv_mh_dict, Muv_add, sigmaUV_a, sigmaUV_b).astype(np.float32)

    bright_mask = muvs < BRIGHT_LIMIT
    faint_mask  = (muvs < FAINT_LIMIT) & (muvs >= BRIGHT_LIMIT)

    bright_coords_sel = halo_coords[bright_mask]
    n_bright_true = len(bright_coords_sel)
    if n_bright_true == 0:
        return None

    bright_coords_sel = bright_coords_sel[rng.permutation(n_bright_true)]
    if n_bright_true > n_obs_needed:
        bright_coords_sel = bright_coords_sel[:n_obs_needed]
    elif n_bright_true < n_obs_needed:
        log.warning(f"  Only {n_bright_true} bright galaxies available at this theta "
                    f"(wanted {n_obs_needed} environments) -- using all of them.")

    all_coords = []
    offsets = [0]
    for bright_coord in bright_coords_sel:
        candidates = halo_tree_2d.query_ball_point(bright_coord[:2], r=half_side, p=np.inf)
        if len(candidates) == 0:
            offsets.append(offsets[-1])
            continue
        cands = np.asarray(candidates, dtype=np.intp)
        cands = cands[faint_mask[cands]]
        if len(cands) == 0:
            offsets.append(offsets[-1])
            continue
        bz = bright_coord[2]
        z_ok = (halo_coords[cands, 2] >= bz - half_side) & (halo_coords[cands, 2] <= bz + half_side)
        cands = cands[z_ok]
        if len(cands) == 0:
            offsets.append(offsets[-1])
            continue
        if len(cands) > MAX_ENV_NEIGHBORS:
            dx2d = halo_coords[cands, 0] - bright_coord[0]
            dy2d = halo_coords[cands, 1] - bright_coord[1]
            d2d = dx2d * dx2d + dy2d * dy2d
            order = np.argpartition(d2d, MAX_ENV_NEIGHBORS)[:MAX_ENV_NEIGHBORS]
            cands = cands[order]
        faint_coords_box = halo_coords[cands]
        faint_mags_box = muvs[cands]
        env = np.column_stack([
            (faint_coords_box - bright_coord).astype(np.float32),
            faint_mags_box.astype(np.float32),
        ])
        all_coords.append(env)
        offsets.append(offsets[-1] + len(env))

    if len(all_coords) == 0:
        return None
    coords_flat = np.concatenate(all_coords, axis=0).astype(np.float32)
    offsets_arr = np.array(offsets, dtype=np.int32)
    return coords_flat, offsets_arr, n_bright_true


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truths-file", type=Path, required=True)
    p.add_argument("--truth-index", type=int, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--n-obs", type=int, default=50)
    p.add_argument("--n-thin", type=int, default=20,
                   help="Random-subsample size of the returned posterior used for the rank stat.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--infer-script", type=str, default="infer_nre.py")
    return p.parse_args()


def main():
    args = parse_args()
    theta = np.load(args.truths_file)[args.truth_index]
    log.info(f"Truth #{args.truth_index}: Muv_add={theta[0]:+.3f}  sigmaUV_a={theta[1]:+.3f}  "
             f"sigmaUV_b={theta[2]:+.3f}")

    obs_dir   = args.output_dir / "obs"
    infer_dir = args.output_dir / "infer" / f"truth{args.truth_index:03d}"
    rank_dir  = args.output_dir / "ranks"
    for d in (obs_dir, infer_dir, rank_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed + args.truth_index)

    log.info("Loading halo catalog + Muv-Mh relation ...")
    halo_coords, logmhs = load_halo_catalog(HALO_CATALOG_PATH)
    muv_mh_dict = load_muv_mh_dict(MUV_MH_FILE)
    log.info("Building 2D cKDTree ...")
    halo_tree_2d = cKDTree(halo_coords[:, :2])
    half_side = cfg.search_box_mpc(REDSHIFT)

    built = build_mock_obs(theta, halo_coords, logmhs, halo_tree_2d, muv_mh_dict, half_side,
                            n_obs_needed=args.n_obs, rng=rng, seed=args.seed)
    if built is None:
        log.error(f"Truth #{args.truth_index} produced zero usable environments -- skipping.")
        np.savez(rank_dir / f"rank_truth{args.truth_index:03d}.npz",
                 theta=theta, rank=np.full(3, np.nan), n_thin=args.n_thin, skipped=True)
        return
    coords_flat, offsets_arr, n_bright_true = built

    obs_path = obs_dir / f"sbc_truth{args.truth_index:03d}.npz"
    np.savez(obs_path, coords=coords_flat, offsets=offsets_arr,
             params=np.asarray(theta, dtype=np.float64), n_bright_true=n_bright_true)
    log.info(f"Built mock obs: {obs_path}  ({len(offsets_arr) - 1} environments, "
             f"{n_bright_true} bright galaxies available)")

    cmd = [
        sys.executable, args.infer_script,
        "--obs-file", str(obs_path),
        "--model-dir", str(args.model_dir),
        "--truths", f"{theta[0]}", f"{theta[1]}", f"{theta[2]}",
        "--n-obs", str(args.n_obs),
        "--output-dir", str(infer_dir),
    ]
    log.info(f"Running inference: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"Inference failed for truth #{args.truth_index}:\n{result.stderr[-3000:]}")
        np.savez(rank_dir / f"rank_truth{args.truth_index:03d}.npz",
                 theta=theta, rank=np.full(3, np.nan), n_thin=args.n_thin, skipped=True)
        return

    candidates = sorted(infer_dir.glob("posterior_samples_*.npy"))
    if not candidates:
        log.error(f"No posterior_samples_*.npy found in {infer_dir} after inference -- skipping.")
        np.savez(rank_dir / f"rank_truth{args.truth_index:03d}.npz",
                 theta=theta, rank=np.full(3, np.nan), n_thin=args.n_thin, skipped=True)
        return
    posterior = np.load(candidates[-1])

    thin_idx = rng.choice(len(posterior), size=min(args.n_thin, len(posterior)), replace=False)
    thinned = posterior[thin_idx]
    rank = (thinned < theta[np.newaxis, :]).sum(axis=0)  # shape (3,), each in [0, n_thin]

    np.savez(rank_dir / f"rank_truth{args.truth_index:03d}.npz",
             theta=theta, rank=rank, n_thin=len(thinned), skipped=False)
    log.info(f"Truth #{args.truth_index} rank: {rank} / {len(thinned)}")


if __name__ == "__main__":
    main()