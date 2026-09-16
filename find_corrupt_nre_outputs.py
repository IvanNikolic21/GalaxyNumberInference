#!/usr/bin/env python
"""
find_corrupt_nre_outputs.py
------------------------------
Companion to find_corrupt_muv_catalogs.py, for Stage 2's output instead of
Stage 1's: scans a build_nre_database.py output directory (nre_*.npz files)
for truncated/corrupted files, using the exact same validity check
build_nre_database.py itself uses to decide skip-vs-regenerate (imported
directly, not reimplemented, so this can never drift out of sync with what
the pipeline actually considers "valid").

Does NOT delete anything -- lists broken files so you can review first.

Usage
-----
    python find_corrupt_nre_outputs.py /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed1955
"""
import argparse
from pathlib import Path

from build_nre_database import _is_valid_output_npz


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(args.output_dir.glob("nre_*.npz"))
    print(f"Scanning {len(files)} files in {args.output_dir} ...\n")

    broken = [f for f in files if not _is_valid_output_npz(f)]

    if not broken:
        print("No broken files found.")
        return

    print(f"Found {len(broken)} broken file(s):\n")
    for f in broken:
        print(f"  {f.stat().st_size:>10,} bytes  {f}")

    print("\nNothing was deleted. Review the list above, then rerun build_nre_database.py "
          "with the same --param-file/--catalog-dir/--output-dir -- it will only "
          "regenerate these, everything else is skipped.")


if __name__ == "__main__":
    main()
