#!/usr/bin/env python
"""
generate_catalogs_grid.py
-------------------------
Generate one MUV catalog per parameter set from a .dat file.

Each row in the .dat file contains:
    Muv_add   sigmaUV_a   sigmaUV_b   (log_likelihood — ignored)

Output files are named by encoded parameter values, e.g.:
    catalog_Madd-0p40_sa-0p02_sb0p72.h5

Usage
-----
    python generate_catalogs_grid.py --param-file params.dat
    python generate_catalogs_grid.py --param-file params.dat --n-iter 5
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
# Fixed: z=10.5
# ---------------------------------------------------------------------------
_CACHE_BASE = "/lustre/astro/ivannik/21cmFAST_cache/d12b21e80b7885d62d31717c2c2d8421"
_HASH       = "ffa852ccaa39d8f82951cc98ff798ab4"
_MUV_MH_DIR = Path("/groups/astro/ivannik/notebooks/clustering_project")
_OUTPUT_DIR = Path("/lustre/astro/ivannik/catalogs_grid")

HALO_CATALOG_PATH = Path(f"{_CACHE_BASE}/1952/{_HASH}/10.5000/HaloCatalog.h5")
MUV_MH_FILE       = _MUV_MH_DIR / "Muv_Mh_z=10.txt"
REDSHIFT          = 10.5

# ---------------------------------------------------------------------------
# MUV-Mh relation (same as generate_catalogs.py)
# ---------------------------------------------------------------------------

def load_muv_mh_dict(path: Path) -> dict:
    data = np.genfromtxt(path, dtype=None, names=True)
    return {data['logMh'][i]: [data['Muv'][i], data['Muv_dust'][i]]
            for i in range(len(data))}


def median_muv(logmhs: np.ndarray, muv_mh_dict: dict) -> np.ndarray:
    return np.array([
        muv_mh_dict[np.round(np.float64(lm), 1)][1]
        for lm in logmhs
    ])


def sigma_uv(logmhs: np.ndarray, sigmaUV_a: float, sigmaUV_b: float) -> np.ndarray:
    return sigmaUV_a * (logmhs - 12) + sigmaUV_b


def sample_muv(
    logmhs: np.ndarray,
    muv_mh_dict: dict,
    Muv_add: float,
    sigmaUV_a: float,
    sigmaUV_b: float,
) -> np.ndarray:
    med = median_muv(logmhs, muv_mh_dict) + Muv_add
    sig = np.clip(sigma_uv(logmhs, sigmaUV_a, sigmaUV_b), a_min=0, a_max=np.inf)
    scatter = np.random.normal(0, sig, len(sig))
    return med + scatter

# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def encode_param(val: float, prefix: str) -> str:
    """Encode a float as a compact string, e.g. -0.401 -> '-0p40'."""
    s = f"{val:.2f}".replace("-", "-").replace(".", "p")
    return f"{prefix}{s}"


def make_catalog_name(Muv_add: float, sigmaUV_a: float, sigmaUV_b: float) -> str:
    parts = [
        encode_param(Muv_add,   "Madd"),
        encode_param(sigmaUV_a, "sa"),
        encode_param(sigmaUV_b, "sb"),
    ]
    return "catalog_" + "_".join(parts) + ".h5"

# ---------------------------------------------------------------------------
# Catalog generation
# ---------------------------------------------------------------------------

def generate_catalog(
    output_path: Path,
    n_iter: int,
    logmhs: np.ndarray,
    muv_mh_dict: dict,
    Muv_add: float,
    sigmaUV_a: float,
    sigmaUV_b: float,
) -> None:
    f = tables.open_file(str(output_path), mode='w')
    test_sample = sample_muv(logmhs, muv_mh_dict, Muv_add, sigmaUV_a, sigmaUV_b)
    atom    = tables.Float64Atom()
    array_c = f.create_earray(
        f.root, 'data', atom,
        shape=(0,) + test_sample.shape,
        expectedrows=n_iter,
    )
    for _ in range(n_iter):
        muv_samples = sample_muv(logmhs, muv_mh_dict, Muv_add, sigmaUV_a, sigmaUV_b)
        array_c.append(muv_samples[np.newaxis, :])
    f.close()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate MUV catalogs for all parameter sets in a .dat file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--param-file", type=Path, required=True,
                   help="Path to .dat file with columns: Muv_add sigmaUV_a sigmaUV_b log_like")
    p.add_argument("--n-iter", type=int, default=1,
                   help="Number of MUV realizations per parameter set.")
    p.add_argument("--output-dir", type=Path, default=_OUTPUT_DIR,
                   help="Directory to save catalogs.")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load parameter grid
    params = np.loadtxt(args.param_file)
    if params.ndim == 1:
        params = params[np.newaxis, :]
    Muv_adds   = params[:, 0]
    sigmaUV_as = params[:, 1]
    sigmaUV_bs = params[:, 2]
    # column 3 is log_likelihood — ignored

    log.info(f"Parameter sets: {len(params)}")
    log.info(f"n_iter per set: {args.n_iter}")
    log.info(f"Output dir:     {args.output_dir}")

    # Load halo catalog once
    log.info(f"Loading halo catalog: {HALO_CATALOG_PATH}")
    with h5py.File(HALO_CATALOG_PATH, 'r') as f:
        halo_masses = np.array(f['HaloCatalog']['OutputFields']['halo_masses'])
    logmhs = np.log10(halo_masses[halo_masses > 0.0])
    log.info(f"  {len(logmhs)} halos with M > 0")

    # Load Muv-Mh relation once
    muv_mh_dict = load_muv_mh_dict(MUV_MH_FILE)

    # Generate catalogs
    for i, (Muv_add, sigmaUV_a, sigmaUV_b) in enumerate(
        zip(Muv_adds, sigmaUV_as, sigmaUV_bs)
    ):
        name = make_catalog_name(Muv_add, sigmaUV_a, sigmaUV_b)
        out  = args.output_dir / name

        if out.exists():
            log.info(f"  [{i+1}/{len(params)}] Already exists, skipping: {name}")
            continue

        log.info(f"  [{i+1}/{len(params)}] Generating: {name}")
        generate_catalog(out, args.n_iter, logmhs, muv_mh_dict,
                         Muv_add, sigmaUV_a, sigmaUV_b)

    log.info("All done.")


if __name__ == "__main__":
    main()