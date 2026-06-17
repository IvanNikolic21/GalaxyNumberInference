#!/usr/bin/env python
"""
plot_num_dens.py
----------------
Plot cumulative number of neighbors vs 3D radius for fiducial and stochastic
models, with 16/84th percentile bands.

Each input npz must contain:
  data : float array, shape (n_galaxies, n_bins-1)   — per-galaxy radial histograms
  bins : float array, shape (n_bins,)                — bin edges in cMpc

These are produced by catalog.radial_distribution().

Usage
-----
    python plot_num_dens.py \\
        --fid-file  /path/to/radial_fid.npz \\
        --stoc-file /path/to/radial_stoc.npz \\
        --output    /path/to/output.pdf
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from plotter import cumulative_neighbors_with_radius

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Cumulative N(< R) plot for fiducial and stochastic models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fid-file",  type=Path, required=True,
                   help="npz file with fiducial radial histograms (keys: data, bins).")
    p.add_argument("--stoc-file", type=Path, required=True,
                   help="npz file with stochastic radial histograms (keys: data, bins).")
    p.add_argument("--output",    type=Path,
                   default=Path("/groups/astro/ivannik/projects/Neighbors/num_dens_with_distr.pdf"),
                   help="Output PDF path.")
    return p.parse_args()


def main():
    args = parse_args()

    fid  = np.load(args.fid_file)
    stoc = np.load(args.stoc_file)

    data_fid  = fid['data']
    data_stoc = stoc['data']
    bins      = fid['bins']

    log.info(f"Fiducial:    {data_fid.shape[0]} galaxies, {data_fid.shape[1]} radial bins")
    log.info(f"Stochastic:  {data_stoc.shape[0]} galaxies, {data_stoc.shape[1]} radial bins")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cumulative_neighbors_with_radius(data_fid, data_stoc, bins, output_path=args.output)
    log.info(f"Saved: {args.output}")


if __name__ == "__main__":
    main()