#!/usr/bin/env python
"""
build_nre_database.py
---------------------
For each catalog in the parameter grid, run a neighbor search for all bright
galaxies and save the faint neighbor environments as NRE inputs.

Each environment is the set of faint neighbors around one bright galaxy,
stored as (dx, dy, dz, MUV) 4-vectors relative to the bright galaxy.

Output format per catalog: compressed .npz with:
    coords  : float32, shape (total_neighbors, 4) — flat (dx,dy,dz,MUV) array, relative to bright galaxy
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
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from scipy.spatial import cKDTree

from galaxy_neighbors import (
    AnalysisConfig,
    RedshiftConfig,
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
CATALOG_DIR       = Path("/lustre/astro/ivannik/catalogs_grid_prior")
OUTPUT_DIR        = Path("/groups/astro/ivannik/projects/Neighbors/nre_database")

BRIGHT_LIMIT      = -21.5
FAINT_LIMIT       = -18.5
REDSHIFT          = 10.5
# Maximum faint neighbors stored per bright galaxy.
# The NRE uses only the 10 closest, but we keep more so that the total-count
# feature (n_total / 200) stays meaningful.  For photo-z databases the search
# volume is huge (±170 Mpc in z for dz=0.5) and without this cap individual
# environments can reach 10k–100k neighbors, producing multi-GB .npz files.
MAX_ENV_NEIGHBORS = 200

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

# Globals — set in the main process BEFORE Pool creation so that all worker
# processes inherit them via fork + copy-on-write.  Workers never write to
# these arrays, so the pages are never dirtied → truly shared memory with
# zero per-worker duplication regardless of how many workers are used.
_halo_coords: np.ndarray = None
_halo_tree_2d: cKDTree   = None


def process_one(
    args_tuple,
    n_total: int,
    output_dir: Path,
    dz_lower: float | None = None,
    dz_upper: float | None = None,
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

    # Load MUV for this parameter set (single realization, index 0).
    # Force float32 to halve memory vs float64.
    muvs = load_muv_catalog(cat_path, index=0).astype(np.float32, copy=False)

    # MUV masks — indices into _halo_coords / muvs
    bright_mask = muvs < BRIGHT_LIMIT
    faint_mask  = (muvs < FAINT_LIMIT) & (muvs >= BRIGHT_LIMIT)

    bright_coords_sel = _halo_coords[bright_mask]

    half_side = cfg.search_box_mpc(REDSHIFT)

    # Z half-extents: asymmetric for photo-z, symmetric otherwise
    hz_lo = dz_lower if dz_lower is not None else half_side
    hz_hi = dz_upper if dz_upper is not None else half_side

    # Collect environments — query the shared cKDTree one bright galaxy at a time.
    # Querying all at once would build a large candidate list (hundreds of MB) that
    # Python's arena allocator never returns to the OS, causing memory to grow over
    # many catalogs.  Per-galaxy queries are tiny and freed at each iteration.
    all_coords = []   # list of (N_i, 4) arrays
    offsets    = [0]

    for bright_coord in bright_coords_sel:
        candidates = _halo_tree_2d.query_ball_point(bright_coord[:2], r=half_side, p=np.inf)
        if len(candidates) == 0:
            offsets.append(offsets[-1])
            continue

        cands = np.asarray(candidates, dtype=np.intp)

        # Keep only faint galaxies
        cands = cands[faint_mask[cands]]
        if len(cands) == 0:
            offsets.append(offsets[-1])
            continue

        # Z filter (symmetric or photo-z asymmetric)
        bz   = bright_coord[2]
        z_ok = (_halo_coords[cands, 2] >= bz - hz_lo) & \
               (_halo_coords[cands, 2] <= bz + hz_hi)
        cands = cands[z_ok]
        if len(cands) == 0:
            offsets.append(offsets[-1])
            continue

        # Keep only the closest MAX_ENV_NEIGHBORS by 2D projected distance.
        # This caps file sizes and per-task memory regardless of how large the
        # search volume is (critical for photo-z databases with wide z windows).
        if len(cands) > MAX_ENV_NEIGHBORS:
            dx2d = _halo_coords[cands, 0] - bright_coord[0]
            dy2d = _halo_coords[cands, 1] - bright_coord[1]
            d2d  = dx2d * dx2d + dy2d * dy2d   # squared — no sqrt needed for sort
            order = np.argpartition(d2d, MAX_ENV_NEIGHBORS)[:MAX_ENV_NEIGHBORS]
            cands = cands[order]

        faint_coords_box = _halo_coords[cands]
        faint_mags_box   = muvs[cands]

        # Stack (dx, dy, dz, MUV) relative to bright galaxy — float32 for storage efficiency
        env = np.column_stack([
            (faint_coords_box - bright_coord).astype(np.float32),
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
    p.add_argument("--photo-z-uncertainty", type=float, default=None,
                   help="Photometric redshift uncertainty (delta-z). "
                        "Converted to comoving Mpc asymmetrically via astropy: "
                        "dz_upper = D_C(z+dz) - D_C(z), dz_lower = D_C(z) - D_C(z-dz). "
                        "When set, the z search box uses these asymmetric bounds and the "
                        "output directory gets a '_photzX' suffix.")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _halo_coords, _halo_tree_2d

    args = parse_args()

    # Compute photo-z comoving bounds and auto-suffix output dir
    dz_lower = dz_upper = None
    if args.photo_z_uncertainty is not None:
        dz = args.photo_z_uncertainty
        z  = REDSHIFT
        dz_upper = (cosmo.comoving_distance(z + dz) - cosmo.comoving_distance(z)).to(u.Mpc).value
        dz_lower = (cosmo.comoving_distance(z)       - cosmo.comoving_distance(z - dz)).to(u.Mpc).value
        dz_str   = f"{dz:.2f}".replace(".", "p")
        args.output_dir = args.output_dir.parent / (args.output_dir.name + f"_photz{dz_str}")
        log.info(f"Photo-z uncertainty: dz={dz}  "
                 f"dz_lower={dz_lower:.2f} Mpc  dz_upper={dz_upper:.2f} Mpc")

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

    # Load halo catalog and build the shared spatial index — both set as globals
    # BEFORE Pool creation so all worker processes inherit them via fork + COW.
    # Workers only read these; pages are never dirtied → one copy in RAM total.
    log.info(f"Loading halo catalog: {HALO_CATALOG_PATH}")
    _halo_coords, _ = load_halo_catalog(HALO_CATALOG_PATH)
    log.info(f"  {len(_halo_coords)} halos")

    log.info("Building shared 2D cKDTree on halo (x,y) positions ...")
    _halo_tree_2d = cKDTree(_halo_coords[:, :2])
    log.info("  cKDTree ready.")

    worker = partial(
        process_one,
        n_total    = len(params),
        output_dir = args.output_dir,
        dz_lower   = dz_lower,
        dz_upper   = dz_upper,
    )

    items = list(enumerate(zip(Muv_adds, sigmaUV_as, sigmaUV_bs)))

    if args.n_workers == 1:
        for item in items:
            worker(item)
    else:
        with Pool(args.n_workers) as pool:   # no initializer — globals already set
            pool.map(worker, items)

    log.info("All done.")


if __name__ == "__main__":
    main()