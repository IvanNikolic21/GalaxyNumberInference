#!/usr/bin/env python
"""
diagnose_missing_catalogs.py
------------------------------
For a given box, check EVERY row in prior.dat against the existing Stage-2
output directory and classify each missing file by whether its theta
would actually trigger sample_muv's known negative-sigma failure mode
(sigma_uv(Mh) = sigma_a*(logMh-12)+sigma_b going negative somewhere in
this box's real halo mass range) -- rather than guessing from a handful of
alphabetically-adjacent example filenames, which can be misleading since
Python's sorted() glob order clusters by Muv_add prefix, not representative
of the missing set as a whole.

Usage
-----
    python diagnose_missing_catalogs.py --param-file prior.dat \\
        --halo-catalog-path /lustre/astro/ivannik/21cmFAST_cache/a4c5e3a912f09f0efa4f82b5a91a56e0/1955/ffa852ccaa39d8f82951cc98ff798ab4/10.5000/HaloCatalog.h5 \\
        --output-dir /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed1955
"""
import argparse
from pathlib import Path

import numpy as np

from galaxy_neighbors import load_halo_catalog
from build_nre_database import make_output_name


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--param-file", type=Path, required=True)
    p.add_argument("--halo-catalog-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    params = np.loadtxt(args.param_file)
    if params.ndim == 1:
        params = params[np.newaxis, :]

    print(f"Loading halo catalog: {args.halo_catalog_path}")
    _, logmhs = load_halo_catalog(args.halo_catalog_path)
    print(f"  {len(logmhs)} halos, log(Mh/Msun) range: [{logmhs.min():.2f}, {logmhs.max():.2f}]")

    missing_rows = []
    for i, (Muv_add, sigmaUV_a, sigmaUV_b) in enumerate(params):
        out_path = args.output_dir / make_output_name(Muv_add, sigmaUV_a, sigmaUV_b)
        if not out_path.exists():
            missing_rows.append((i, Muv_add, sigmaUV_a, sigmaUV_b))

    print(f"\nTotal rows: {len(params)}  Missing: {len(missing_rows)}")

    # For each missing row, check if sigma_uv(Mh) actually goes negative
    # anywhere in this box's real halo mass range -- the exact condition
    # that crashes sample_muv with "invalid (negative) scale".
    n_would_crash = 0
    n_other = []
    for i, Muv_add, sigmaUV_a, sigmaUV_b in missing_rows:
        sig = sigmaUV_a * (logmhs - 12) + sigmaUV_b
        if sig.min() < 0:
            n_would_crash += 1
        else:
            n_other.append((i, Muv_add, sigmaUV_a, sigmaUV_b))

    print(f"\nOf the {len(missing_rows)} missing rows:")
    print(f"  {n_would_crash} WOULD trigger sample_muv's negative-sigma crash "
          f"(sigma_uv(Mh) < 0 somewhere in this box's real mass range)")
    print(f"  {len(n_other)} would NOT crash -- missing for some other reason "
          f"(e.g. genuinely zero bright galaxies, or a different failure)")

    if n_other:
        other_arr = np.array([(a, b, c) for _, a, b, c in n_other])
        print(f"\n  Non-crash missing rows' theta ranges:")
        print(f"    Muv_add:   [{other_arr[:,0].min():.2f}, {other_arr[:,0].max():.2f}]")
        print(f"    sigmaUV_a: [{other_arr[:,1].min():.2f}, {other_arr[:,1].max():.2f}]")
        print(f"    sigmaUV_b: [{other_arr[:,2].min():.2f}, {other_arr[:,2].max():.2f}]")

    # Full missing-set theta ranges, for comparison against the prior's own
    # overall range (shows whether missingness really does concentrate at
    # one edge/sign, across the WHOLE set, not just a handful of examples).
    all_missing = np.array([(a, b, c) for _, a, b, c in missing_rows])
    print(f"\nFull missing-set theta ranges ({len(missing_rows)} rows):")
    print(f"  Muv_add:   [{all_missing[:,0].min():.2f}, {all_missing[:,0].max():.2f}]  "
          f"(prior range: [{params[:,0].min():.2f}, {params[:,0].max():.2f}])")
    print(f"  sigmaUV_a: [{all_missing[:,1].min():.2f}, {all_missing[:,1].max():.2f}]  "
          f"(prior range: [{params[:,1].min():.2f}, {params[:,1].max():.2f}])")
    print(f"  sigmaUV_b: [{all_missing[:,2].min():.2f}, {all_missing[:,2].max():.2f}]  "
          f"(prior range: [{params[:,2].min():.2f}, {params[:,2].max():.2f}])")
    print(f"  fraction of missing set with sigmaUV_a < 0: "
          f"{np.mean(all_missing[:,1] < 0):.1%}  (vs {np.mean(params[:,1] < 0):.1%} in the full prior)")


if __name__ == "__main__":
    main()
