#!/usr/bin/env python
"""
patch_bright_logmh.py
-----------------------
Add the missing `bright_logmh` field to ALREADY-BUILT build_nre_database.py
output files, WITHOUT redoing the expensive neighbor search (the part that
took close to a month for the 4-box multibox database). This only
reproduces the cheap bright-galaxy selection step (load Stage-1 MUV
catalog, deterministic shuffle + cap using the same seed+i as the original
run) to recover which halo mass hosted each bright galaxy, then appends it
to the existing .npz file in place -- no cKDTree, no neighbor search.

Row order in --param-file determines the per-theta RNG seed (seed+i,
matching build_nre_database.py's process_one exactly), so this MUST be run
with the identical prior.dat used for the original build, and the same
--seed / --max-environments-per-catalog values -- otherwise the reproduced
selection won't match what's already baked into the existing coords/offsets.
As a safety check, each file's reproduced bright-galaxy count is compared
against its existing offsets array before writing anything; a mismatch
aborts that file instead of silently writing misaligned data.

Usage
-----
    python patch_bright_logmh.py --param-file prior.dat \\
        --halo-catalog-path /lustre/astro/ivannik/21cmFAST_cache/a4c5e3a912f09f0efa4f82b5a91a56e0/1955/ffa852ccaa39d8f82951cc98ff798ab4/10.5000/HaloCatalog.h5 \\
        --catalog-dir /lustre/astro/ivannik/catalogs_grid_prior_seed1955 \\
        --output-dir /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed1955
"""
import argparse
import logging
from pathlib import Path

import numpy as np

from galaxy_neighbors import load_halo_catalog, load_muv_catalog
from build_nre_database import make_output_name, make_catalog_name, BRIGHT_LIMIT

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--param-file", type=Path, required=True)
    p.add_argument("--halo-catalog-path", type=Path, required=True)
    p.add_argument("--catalog-dir", type=Path, required=True,
                   help="Stage-1 output dir (catalogs_grid_prior_seed<N>) -- cheap MUV "
                        "catalogs, not the expensive Stage-2 neighbor-search output.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Existing Stage-2 nre_database_prior_capped_seed<N> dir to patch in place.")
    p.add_argument("--max-environments-per-catalog", type=int, default=100,
                   help="MUST match the value used in the original build_nre_database.py run.")
    p.add_argument("--seed", type=int, default=42,
                   help="MUST match the --seed used in the original run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Check everything (including the length cross-check) but don't "
                        "actually write any files.")
    return p.parse_args()


def main():
    args = parse_args()
    params = np.loadtxt(args.param_file)
    if params.ndim == 1:
        params = params[np.newaxis, :]

    log.info(f"Loading halo catalog: {args.halo_catalog_path}")
    halo_coords, halo_logmhs = load_halo_catalog(args.halo_catalog_path)
    log.info(f"  {len(halo_coords)} halos")

    max_env_per_catalog = args.max_environments_per_catalog or None

    n_patched = n_missing_out = n_missing_cat = n_already_done = n_mismatch = n_length_fail = 0

    for i, (Muv_add, sigmaUV_a, sigmaUV_b) in enumerate(params):
        out_path = args.output_dir / make_output_name(Muv_add, sigmaUV_a, sigmaUV_b)
        if not out_path.exists():
            n_missing_out += 1
            continue

        with np.load(out_path) as d:
            if "bright_logmh" in d.files:
                n_already_done += 1
                continue
            offsets = d["offsets"]
            stored_params = d["params"]
            coords = d["coords"]
            n_bright_true_stored = int(d["n_bright_true"])

        if not np.allclose(stored_params, [Muv_add, sigmaUV_a, sigmaUV_b], atol=1e-6):
            log.error(f"  [{i}] {out_path.name}: stored params {stored_params} != "
                       f"param-file row {[Muv_add, sigmaUV_a, sigmaUV_b]} -- row-order "
                       f"mismatch, skipping this file.")
            n_mismatch += 1
            continue

        cat_path = args.catalog_dir / make_catalog_name(Muv_add, sigmaUV_a, sigmaUV_b)
        if not cat_path.exists():
            log.warning(f"  [{i}] Stage-1 catalog not found: {cat_path.name}, skipping.")
            n_missing_cat += 1
            continue

        # --- reproduce process_one's bright-galaxy selection exactly, no neighbor search ---
        muvs = load_muv_catalog(cat_path, index=0).astype(np.float32, copy=False)
        bright_mask = muvs < BRIGHT_LIMIT
        bright_logmhs_sel = halo_logmhs[bright_mask]
        n_bright_true = len(bright_logmhs_sel)

        rng = np.random.default_rng(args.seed + i)
        if n_bright_true > 0:
            bright_logmhs_sel = bright_logmhs_sel[rng.permutation(n_bright_true)]
        if max_env_per_catalog is not None and n_bright_true > max_env_per_catalog:
            bright_logmhs_sel = bright_logmhs_sel[:max_env_per_catalog]

        n_envs_expected = len(offsets) - 1
        if n_bright_true_stored != n_bright_true or len(bright_logmhs_sel) != n_envs_expected:
            log.error(f"  [{i}] {out_path.name}: reproduced selection length "
                       f"({len(bright_logmhs_sel)}, n_bright_true={n_bright_true}) != existing "
                       f"file ({n_envs_expected} envs, n_bright_true={n_bright_true_stored}) -- "
                       f"ABORTING this file. Check --seed/--max-environments-per-catalog/"
                       f"--param-file match the original run.")
            n_length_fail += 1
            continue

        if args.dry_run:
            n_patched += 1
            continue

        np.savez_compressed(
            out_path,
            coords=coords,
            offsets=offsets,
            params=stored_params,
            n_bright_true=n_bright_true_stored,
            bright_logmh=bright_logmhs_sel.astype(np.float32),
        )
        n_patched += 1
        if (i + 1) % 200 == 0:
            log.info(f"  [{i+1}/{len(params)}] patched so far: {n_patched}")

    log.info(f"Done. patched={n_patched}  already_done={n_already_done}  "
             f"missing_output_file={n_missing_out}  missing_stage1_catalog={n_missing_cat}  "
             f"param_mismatch={n_mismatch}  length_mismatch={n_length_fail}")
    if n_length_fail > 0:
        log.warning(f"{n_length_fail} files FAILED the length cross-check and were NOT patched -- "
                    f"investigate before assuming this database is complete.")


if __name__ == "__main__":
    main()
