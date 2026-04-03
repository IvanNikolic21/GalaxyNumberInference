#!/usr/bin/env python
"""
build_nre_database.py
---------------------
For each catalog in the parameter grid, run a neighbor search for all bright
galaxies and save the faint neighbor environments as NRE inputs.

Each environment is the set of faint neighbors around one bright galaxy,
stored as (x, y, z, MUV) 4-vectors.

Output format per catalog: compressed .npz with:
    coords  : float32, shape (total_neighbors, 4) — flat (x,y,z,MUV) array
    offsets : int32,   shape (n_bright + 1,)      — environment boundaries
    params  : float64, shape (3,)                 — (Muv_add, sigmaUV_a, sigmaUV_b)

To retrieve environment i:
    coords[offsets[i]:offsets[i+1]]  ->  shape (n_neighbors_i, 4)

Usage
-----
    python build_nre_database.py --param-file params.dat
    python build_nre_database.py --param-file params.dat --n-workers 8
"""

import argparse
import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

from galaxy_neighbors import (
    AnalysisConfig,
    RedshiftConfig,
    find_neighbors_in_box,
    load_halo_catalog,
    load_muv_catalog,
)

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
# Fixed settings
# ---------------------------------------------------------------------------
_CACHE_BASE = "/lustre/astro/ivannik/21cmFAST_cache/d12b21e80b7885d62d31717c2c2d8421"
_HASH       = "ffa852ccaa39d8f82951cc98ff798ab4"

HALO_CATALOG_PATH = Path(f"{_CACHE_BASE}/1952/{_HASH}/10.5000/HaloCatalog.h5")
CATALOG_DIR       = Path("/lustre/astro/ivannik/catalogs_grid")
OUTPUT_DIR        = Path("/groups/astro/ivannik/projects/Neighbors/nre_database")

BRIGHT_LIMIT = -21.5
FAINT_LIMIT  = -18.5
REDSHIFT     = 10.5

cfg = AnalysisConfig(
    bright_limits         = [BRIGHT_LIMIT],
    faint_limits          = [FAINT_LIMIT],
    preselect_faint_limit = FAINT_LIMIT,
    survey_area_arcmin2   = 12.24,
)

z_cfg = RedshiftConfig(
    redshift            = REDSHIFT,
    halo_catalog_path   = HALO_CATALOG_PATH,
    muv_fiducial_path   = HALO_CATALOG_PATH,   # placeholder — not used directly
    muv_stochastic_path = HALO_CATALOG_PATH,
)

# ---------------------------------------------------------------------------
# Naming — must match generate_catalogs_grid.py
# ---------------------------------------------------------------------------

def encode_param(val: float, prefix: str) -> str:
    s = f"{val:.2f}".replace(".", "p")
    return f"{prefix}{s}"


def make_catalog_name(Muv_add: float, sigmaUV_a: float, sigmaUV_b: float) -> str:
    parts = [
        encode_param(Muv_add,   "Madd"),
        encode_param(sigmaUV_a, "sa"),
        encode_param(sigmaUV_b, "sb"),
    ]
    return "catalog_" + "_".join(parts) + ".h5"


def make_output_name(Muv_add: float, sigmaUV_a: float, sigmaUV_b: float) -> str:
    parts = [
        encode_param(Muv_add,   "Madd"),
        encode_param(sigmaUV_a, "sa"),
        encode_param(sigmaUV_b, "sb"),
    ]
    return "nre_" + "_".join(parts) + ".npz"

# ---------------------------------------------------------------------------
# Core: build environments for one catalog
# ---------------------------------------------------------------------------

def process_one(
    args_tuple,
    halo_coords: np.ndarray,
    n_total: int,
    output_dir: Path,
):
    i, (Muv_add, sigmaUV_a, sigmaUV_b) = args_tuple

    out_path = output_dir / make_output_name(Muv_add, sigmaUV_a, sigmaUV_b)
    if out_path.exists():
        log.info(f"  [{i+1}/{n_total}] Already exists, skipping.")
        return

    cat_path = CATALOG_DIR / make_catalog_name(Muv_add, sigmaUV_a, sigmaUV_b)
    if not cat_path.exists():
        log.warning(f"  [{i+1}/{n_total}] Catalog not found: {cat_path.name}, skipping.")
        return

    log.info(f"  [{i+1}/{n_total}] Processing: {cat_path.name}")

    # Load MUV for this parameter set (single realization, index 0)
    muvs = load_muv_catalog(cat_path, index=0)

    # Build bright and faint pools
    bright_cut = BRIGHT_LIMIT
    faint_cut  = FAINT_LIMIT

    bright_mask = muvs < bright_cut
    faint_mask  = (muvs < faint_cut) & (muvs >= bright_cut)

    bright_coords_sel = halo_coords[bright_mask]
    bright_mags_sel   = muvs[bright_mask]
    faint_coords_sel  = halo_coords[faint_mask]
    faint_mags_sel    = muvs[faint_mask]

    half_side = cfg.search_box_mpc(REDSHIFT)

    # Loop over bright galaxies — collect environments
    all_coords = []   # list of (N_i, 4) arrays
    offsets    = [0]

    for bright_coord, bright_mag in zip(bright_coords_sel, bright_mags_sel):
        faint_mags_box, faint_coords_box, _ = find_neighbors_in_box(
            bright_coord  = bright_coord,
            faint_coords  = faint_coords_sel,
            faint_mags    = faint_mags_sel,
            half_side     = half_side,
            faint_limit   = FAINT_LIMIT,
        )

        if len(faint_mags_box) == 0:
            offsets.append(offsets[-1])
            continue

        # Stack (x, y, z, MUV) — float32 for storage efficiency
        env = np.column_stack([
            faint_coords_box.astype(np.float32),
            faint_mags_box.astype(np.float32),
        ])
        all_coords.append(env)
        offsets.append(offsets[-1] + len(env))

    if len(all_coords) == 0:
        log.warning(f"  [{i+1}/{n_total}] No environments found, skipping.")
        return

    coords_flat = np.concatenate(all_coords, axis=0).astype(np.float32)
    offsets_arr = np.array(offsets, dtype=np.int32)
    params_arr  = np.array([Muv_add, sigmaUV_a, sigmaUV_b], dtype=np.float64)

    np.savez_compressed(
        out_path,
        coords  = coords_flat,
        offsets = offsets_arr,
        params  = params_arr,
    )
    log.info(f"  [{i+1}/{n_total}] Saved: {out_path.name}  "
             f"({len(bright_coords_sel)} bright, {len(coords_flat)} total neighbors)")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build NRE input database from catalog grid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--param-file", type=Path, required=True,
                   help="Path to .dat file with columns: Muv_add sigmaUV_a sigmaUV_b log_like")
    p.add_argument("--n-workers", type=int, default=4,
                   help="Number of parallel workers.")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                   help="Directory to save NRE input files.")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load parameter grid
    params     = np.loadtxt(args.param_file)
    if params.ndim == 1:
        params = params[np.newaxis, :]
    Muv_adds   = params[:, 0]
    sigmaUV_as = params[:, 1]
    sigmaUV_bs = params[:, 2]

    log.info(f"Parameter sets: {len(params)}")
    log.info(f"Output dir:     {args.output_dir}")

    # Load halo catalog once — shared across all workers (read-only)
    log.info(f"Loading halo catalog: {HALO_CATALOG_PATH}")
    halo_coords, _ = load_halo_catalog(HALO_CATALOG_PATH)
    log.info(f"  {len(halo_coords)} halos")

    worker = partial(
        process_one,
        halo_coords = halo_coords,
        n_total     = len(params),
        output_dir  = args.output_dir,
    )

    items = list(enumerate(zip(Muv_adds, sigmaUV_as, sigmaUV_bs)))

    if args.n_workers == 1:
        for item in items:
            worker(item)
    else:
        with Pool(args.n_workers) as pool:
            pool.map(worker, items)

    log.info("All done.")


if __name__ == "__main__":
    main()