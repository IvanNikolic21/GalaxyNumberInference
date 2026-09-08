#!/usr/bin/env python
"""
find_corrupt_muv_catalogs.py
------------------------------
Dry-run scan of a generate_catalog_database.py output directory for
truncated/corrupted per-theta MUV catalog files -- the "exists but was never
finished" failure mode that generate_catalog_database.py's _worker doesn't
guard against (only checks file existence, not validity, so a catalog
truncated by an interrupted run silently survives every subsequent rerun).

Checks each catalog_*.h5 file opens cleanly via PyTables and its 'data'
earray has the expected number of rows (--n-iter, matching whatever was
passed to generate_catalog_database.py originally).

Does NOT delete anything -- lists broken files so you can review before
removing them.

Usage
-----
    python find_corrupt_muv_catalogs.py /lustre/astro/ivannik/catalogs_grid_prior_seed1955 --n-iter 1
"""
import argparse
from pathlib import Path

import tables


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("catalog_dir", type=Path)
    p.add_argument("--n-iter", type=int, default=1,
                   help="Expected number of rows in each catalog's 'data' earray -- "
                        "must match whatever --n-iter generate_catalog_database.py used.")
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(args.catalog_dir.glob("catalog_*.h5"))
    print(f"Scanning {len(files)} catalog files in {args.catalog_dir} ...\n")

    broken = []
    for f in files:
        try:
            with tables.open_file(str(f), mode='r') as h5:
                n_rows = h5.root.data.shape[0]
                if n_rows != args.n_iter:
                    broken.append((f, f"has {n_rows} rows, expected {args.n_iter}"))
        except Exception as e:
            broken.append((f, f"UNREADABLE: {type(e).__name__}: {e}"))

    if not broken:
        print("No broken files found.")
        return

    print(f"Found {len(broken)} broken file(s):\n")
    for f, reason in broken:
        size = f.stat().st_size
        print(f"  {size:>10,} bytes  {f}")
        print(f"      -> {reason}")

    print("\nNothing was deleted. Review the list above, then:")
    print("  rm <bad files>")
    print("  # rerun generate_catalog_database.py with the same --param-file/--output-dir --")
    print("  # it will only regenerate the ones you just deleted, everything else is skipped.")


if __name__ == "__main__":
    main()
