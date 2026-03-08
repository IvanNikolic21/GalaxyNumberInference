#!/usr/bin/env python
"""
generate_catalogs.py
--------------------
Generate fiducial and stochastic MUV catalogs for a given redshift.

Usage
-----
    python generate_catalogs.py --redshift 14.0 --n-iter 300
    python generate_catalogs.py --redshift 10.5 --n-iter 100
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import h5py
import tables

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
# Paths
# ---------------------------------------------------------------------------
_CACHE_BASE  = "/lustre/astro/ivannik/21cmFAST_cache/d12b21e80b7885d62d31717c2c2d8421"
_HASH        = "ffa852ccaa39d8f82951cc98ff798ab4"
_MUV_MH_DIR  = Path("/groups/astro/ivannik/notebooks/clustering_project")
_OUTPUT_DIR  = Path("/lustre/astro/ivannik")

HALO_CATALOG_PATHS = {
    8.0:  Path(f"{_CACHE_BASE}/1955/{_HASH}/8.0000/HaloCatalog.h5"),
    10.5: Path(f"{_CACHE_BASE}/1952/{_HASH}/10.5000/HaloCatalog.h5"),
    12.0: Path(f"{_CACHE_BASE}/1955/{_HASH}/12.0000/HaloCatalog.h5"),
    14.0: Path(f"{_CACHE_BASE}/1955/{_HASH}/14.0000/HaloCatalog.h5"),
}

MUV_MH_FILES = {
    8.0:  _MUV_MH_DIR / "Muv_Mh_z=8.txt",
    10.5: _MUV_MH_DIR / "Muv_Mh_z=10.txt",
    12.0: _MUV_MH_DIR / "Muv_Mh_z=12.txt",
    14.0: _MUV_MH_DIR / "Muv_Mh_z=14.txt",
}

AVAILABLE_REDSHIFTS = sorted(HALO_CATALOG_PATHS.keys())

# ---------------------------------------------------------------------------
# Model parameters — fixed per model type
# ---------------------------------------------------------------------------
FIDUCIAL_PARAMS = dict(
    sigmaUV_a = 0.0,
    sigmaUV_b = 0.3,
    Muv_add   = -0.8,
)

STOCHASTIC_PARAMS = dict(
    sigmaUV_a = -0.34,
    sigmaUV_b = 0.6,
    Muv_add   = +0.3,
)

# ---------------------------------------------------------------------------
# MUV-Mh relation
# ---------------------------------------------------------------------------

def load_muv_mh_dict(z: float) -> dict:
    path = MUV_MH_FILES[z]
    log.info(f"Loading Muv-Mh relation from {path}")
    data = np.genfromtxt(path, dtype=None, names=True)
    return {data['logMh'][i]: [data['Muv'][i], data['Muv_dust'][i]]
            for i in range(len(data))}


def muv_from_logmh_nodust(logMh: float, muv_mh_dict: dict) -> float:
    return muv_mh_dict[logMh][0]


def muv_from_logmh_dust(logMh: float, muv_mh_dict: dict) -> float:
    return muv_mh_dict[logMh][1]


def median_muv(logmhs: np.ndarray, muv_mh_dict: dict) -> np.ndarray:
    """Median MUV (dust) for an array of log halo masses."""
    return np.array([
        muv_from_logmh_dust(np.round(np.float64(lm), 1), muv_mh_dict)
        for lm in logmhs
    ])


def sigma_uv(logmhs: np.ndarray, sigmaUV_a: float, sigmaUV_b: float) -> np.ndarray:
    """Mass-dependent UV scatter: sigma(Mh) = a*(logMh - 12) + b."""
    return sigmaUV_a * (logmhs - 12) + sigmaUV_b


def sample_muv(
    logmhs: np.ndarray,
    muv_mh_dict: dict,
    sigmaUV_a: float,
    sigmaUV_b: float,
    Muv_add: float,
) -> np.ndarray:
    """Draw one MUV realization for all halos."""
    med = median_muv(logmhs, muv_mh_dict) + Muv_add
    sig = sigma_uv(logmhs, sigmaUV_a, sigmaUV_b)
    scatter = np.random.normal(0, sig, len(sig))
    return med + scatter

# ---------------------------------------------------------------------------
# HDF5 catalog generation
# ---------------------------------------------------------------------------

def generate_catalog(
    output_path: Path,
    n_iter: int,
    logmhs: np.ndarray,
    muv_mh_dict: dict,
    model_params: dict,
) -> None:
    """Generate a catalog of n_iter MUV realizations, saved iteratively to HDF5."""
    log.info(f"Generating {n_iter} realizations -> {output_path}")

    f = tables.open_file(str(output_path), mode='w')
    test_sample = sample_muv(logmhs, muv_mh_dict, **model_params)
    atom    = tables.Float64Atom()
    array_c = f.create_earray(
        f.root, 'data', atom,
        shape=(0,) + test_sample.shape,
        expectedrows=n_iter,
    )

    for i in range(n_iter):
        if (i + 1) % 50 == 0:
            log.info(f"  Realization {i+1}/{n_iter}")
        muv_samples = sample_muv(logmhs, muv_mh_dict, **model_params)
        array_c.append(muv_samples[np.newaxis, :])

    f.close()
    log.info(f"  Done: {output_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate fiducial and stochastic MUV catalogs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--redshift", type=float, required=True,
        choices=AVAILABLE_REDSHIFTS, metavar="Z",
        help=f"Redshift snapshot. Available: {AVAILABLE_REDSHIFTS}",
    )
    p.add_argument(
        "--n-iter", type=int, required=True,
        help="Number of MUV realizations to generate.",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    z     = args.redshift
    n     = args.n_iter
    z_str = str(z).replace(".", "p")

    log.info(f"Redshift: {z},  n_iter: {n}")

    # Load halo catalog
    halo_path = HALO_CATALOG_PATHS[z]
    log.info(f"Loading halo catalog: {halo_path}")
    with h5py.File(halo_path, 'r') as f:
        halo_masses = np.array(f['HaloCatalog']['OutputFields']['halo_masses'])

    logmhs = np.log10(halo_masses[halo_masses > 0.0])
    log.info(f"  {len(logmhs)} halos with M > 0")

    # Load Muv-Mh relation
    muv_mh_dict = load_muv_mh_dict(z)

    # Output paths
    out_fid  = _OUTPUT_DIR / f"catalog_fiducial_bigger_z{z_str}_{n}.h5"
    out_stoc = _OUTPUT_DIR / f"catalog_stoch_bigger_z{z_str}_{n}.h5"

    # Generate
    generate_catalog(out_fid,  n, logmhs, muv_mh_dict, FIDUCIAL_PARAMS)
    generate_catalog(out_stoc, n, logmhs, muv_mh_dict, STOCHASTIC_PARAMS)

    log.info("All done.")


if __name__ == "__main__":
    main()
