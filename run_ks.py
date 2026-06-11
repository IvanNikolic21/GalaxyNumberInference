#!/usr/bin/env python
"""
run_ks.py
---------
Run the KS/AD test analysis on existing d1s cache files for a given redshift.

Usage
-----
    python run_ks.py --redshift 10.5
    python run_ks.py --redshift 8.0
    python run_ks.py --redshift 12.0
    python run_ks.py --redshift 14.0
"""

import argparse
import logging
import time
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.cosmology import Planck18 as cosmo
from astropy import units as u

from galaxy_neighbors import AnalysisConfig, RedshiftConfig, compute_bright_counts, load_muv_catalog
from galaxy_d1s import load_d1s
from galaxy_ks import KSConfig, run_ks_analysis, plot_ks_results, plot_ks_summary_bars, summarise_ks

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
# Cache file registry — one entry per redshift
# ---------------------------------------------------------------------------
CACHE_ROOT = Path("/groups/astro/ivannik/projects/Neighbors/cache")

CACHE_FILES = {
    8.0:  (
        CACHE_ROOT / "z8.0"  / "d1s_fiducial_real1.npz",
        CACHE_ROOT / "z8.0"  / "d1s_stochastic_real1.npz",
    ),
    10.5: (
        CACHE_ROOT / "z10.5" / "d1s_fiducial_real20.npz",
        CACHE_ROOT / "z10.5" / "d1s_stochastic_real20.npz",
    ),
    12.0: (
        CACHE_ROOT / "z12.0" / "d1s_fiducial_real100.npz",
        CACHE_ROOT / "z12.0" / "d1s_stochastic_real100.npz",
    ),
    14.0: (
        CACHE_ROOT / "z14.0" / "d1s_fiducial_real60.npz",
        CACHE_ROOT / "z14.0" / "d1s_stochastic_real60.npz",
    ),
}

AVAILABLE_REDSHIFTS = sorted(CACHE_FILES.keys())
OUTPUT_ROOT = Path("/groups/astro/ivannik/projects/Neighbors/ks_results")
_CACHE_BASE = "/lustre/astro/ivannik/21cmFAST_cache/d12b21e80b7885d62d31717c2c2d8421"
_HASH       = "ffa852ccaa39d8f82951cc98ff798ab4"

REDSHIFT_CONFIGS = {
    8.0:  RedshiftConfig(redshift=8.0,  halo_catalog_path=Path(f"{_CACHE_BASE}/1955/{_HASH}/8.0000/HaloCatalog.h5"),  muv_fiducial_path=Path("/lustre/astro/ivannik/catalog_fiducial_bigger_z8n.h5"),          muv_stochastic_path=Path("/lustre/astro/ivannik/catalog_stoch_bigger_z8n.h5")),
    10.5: RedshiftConfig(redshift=10.5, halo_catalog_path=Path(f"{_CACHE_BASE}/1952/{_HASH}/10.5000/HaloCatalog.h5"), muv_fiducial_path=Path("/lustre/astro/ivannik/catalog_fiducial_bigger_new_s100.h5"), muv_stochastic_path=Path("/lustre/astro/ivannik/catalog_stoch_bigger_new3.h5")),
    12.0: RedshiftConfig(redshift=12.0, halo_catalog_path=Path(f"{_CACHE_BASE}/1955/{_HASH}/12.0000/HaloCatalog.h5"), muv_fiducial_path=Path("/lustre/astro/ivannik/catalog_fiducial_bigger_z12_ns.h5"),        muv_stochastic_path=Path("/lustre/astro/ivannik/catalog_stoch_bigger_z12.h5")),
    14.0: RedshiftConfig(redshift=14.0, halo_catalog_path=Path(f"{_CACHE_BASE}/1955/{_HASH}/14.0000/HaloCatalog.h5"), muv_fiducial_path=Path("/lustre/astro/ivannik/catalog_fiducial_bigger_z14_400_n.h5"),        muv_stochastic_path=Path("/lustre/astro/ivannik/catalog_stoch_bigger_z14_800_ns.h5")),
}

N_REALIZATIONS = {8.0: 2, 10.5: 20, 12.0: 100, 14.0: 60}
# ---------------------------------------------------------------------------
# Must match the config used when the cache files were computed
# ---------------------------------------------------------------------------
cfg = AnalysisConfig(
    bright_limits         = [-20.5, -20.75, -21.0, -21.25, -21.5, -21.75, -22.0],
    faint_limits          = [-16.5, -16.6, -16.7, -16.8, -16.9, -17.0, -17.1, -17.2,
                             -17.3, -17.4, -17.5, -17.6, -17.7, -17.8, -17.9, -18.0,
                             -18.1, -18.2, -18.3, -18.4, -18.5, -18.6, -18.7, -18.8,
                             -18.9, -19.0, -19.1, -19.2, -19.3, -19.4, -19.5, -19.6],
    preselect_faint_limit = -16.5,
    survey_area_arcmin2   = 12.24,
)


# ---------------------------------------------------------------------------
# UVLF from catalog + required survey area
# ---------------------------------------------------------------------------
_UVLF_BIN_EDGES = np.arange(-25.5, -9.5, 0.5)          # mag, half-integer edges
_UVLF_DMUV      = float(_UVLF_BIN_EDGES[1] - _UVLF_BIN_EDGES[0])
_BOX_LEN_MPC    = 512.0


def get_LF(Muv, n_realizations=1, BOX_LEN=_BOX_LEN_MPC):
    """UV luminosity function from a stacked catalog.

    Muv : concatenated MUV array across n_realizations catalogs
    n_realizations : scales the effective volume (n_realizations × BOX_LEN³)
    BOX_LEN : simulation box side in Mpc

    Returns bin_centers [mag] and phi [Mpc⁻³ mag⁻¹].
    """
    hist, _ = np.histogram(Muv, bins=_UVLF_BIN_EDGES)
    ndens = hist / (n_realizations * BOX_LEN**3) / _UVLF_DMUV
    bin_centers = 0.5 * (_UVLF_BIN_EDGES[:-1] + _UVLF_BIN_EDGES[1:])
    return bin_centers, ndens


def compute_required_survey_area(
    n_pointings: np.ndarray,
    bin_centers: np.ndarray,
    phi: np.ndarray,
    bright_limit: float,
    redshift: float,
    delta_z: float = 1.0,
) -> np.ndarray:
    """Survey area [arcmin²] needed to observe n_pointings bright galaxies.

    Integrates the UVLF brighter than bright_limit to get the comoving number
    density, then converts to a projected surface density via the volume element
    dV/dΩ/dz × Δz.  Area = N_pointings / surface_density.

    n_pointings : bootstrap distribution of critical sample sizes (may have NaN)
    bin_centers : UVLF magnitude bin centers [mag]
    phi : UVLF [Mpc⁻³ mag⁻¹]
    bright_limit : galaxies with M_UV < bright_limit are counted
    redshift : central redshift of the survey slice
    delta_z : redshift slice width (default 1.0)
    """
    bright_mask = bin_centers < bright_limit
    if not bright_mask.any():
        return np.full(len(n_pointings), np.nan)

    n_bright = np.trapz(phi[bright_mask], bin_centers[bright_mask])  # Mpc⁻³

    D_C = cosmo.comoving_distance(redshift).to(u.Mpc).value          # Mpc
    H_z = cosmo.H(redshift).to(u.km / u.s / u.Mpc).value            # km/s/Mpc
    dV_dOmega_dz = D_C**2 * (2.998e5 / H_z)                         # Mpc³ sr⁻¹

    arcmin2_per_sr = (180.0 * 60.0 / np.pi) ** 2
    n_surf = n_bright * dV_dOmega_dz * delta_z / arcmin2_per_sr      # arcmin⁻²

    return n_pointings / n_surf                                       # arcmin²


def apply_p_neighbor_correction(results, d1s_fid, bright_counts, bright_key,
                                d1s_stoc=None, bright_counts_stoc=None):
    n_total      = bright_counts[bright_key]
    n_total_stoc = (bright_counts_stoc or bright_counts)[bright_key]
    corrected = {}
    threshold_fid  = None
    threshold_stoc = None
    for fkey in results:
        n_passed = len(d1s_fid[bright_key][fkey])
        p = n_passed / n_total if n_total > 0 else 1.0
        if n_total == 0 or p == 0:
            log.warning(f"    {bright_key} | {fkey}: no bright galaxies, skipping.")
            corrected[fkey] = {'ks': np.full_like(results[fkey]['ks'], np.nan),
                               'ad': np.full_like(results[fkey]['ad'], np.nan)}
            continue
        corrected[fkey] = {
            'ks': results[fkey]['ks'] / p,
            'ad': results[fkey]['ad'] / p,
        }
        if threshold_fid is None and p < 0.5:
            threshold_fid = fkey
        if d1s_stoc is not None:
            p_stoc = len(d1s_stoc[bright_key][fkey]) / n_total_stoc if n_total_stoc > 0 else 1.0
            if threshold_stoc is None and p_stoc < 0.5:
                threshold_stoc = fkey
            log.info(f"    {bright_key} | {fkey}: p_fid={p:.3f}  p_stoc={p_stoc:.3f}  factor={1/p:.2f}")
        else:
            log.info(f"    {bright_key} | {fkey}: p_neighbor={p:.3f}  factor={1/p:.2f}")
    fid_str  = threshold_fid  if threshold_fid  is not None else "never (<16.5)"
    stoc_str = threshold_stoc if threshold_stoc is not None else "never (<16.5)"
    if d1s_stoc is not None:
        log.info(f"  50%-empty threshold: {bright_key} | fid: {fid_str}  stoc: {stoc_str}")
    else:
        log.info(f"  50%-empty threshold: {bright_key} | fid: {fid_str}")
    return corrected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="KS/AD test analysis for a given redshift")
    p.add_argument(
        "--redshift", type=float, required=True,
        choices=AVAILABLE_REDSHIFTS, metavar="Z",
        help=f"Redshift to analyse. Available: {AVAILABLE_REDSHIFTS}",
    )
    p.add_argument(
        "--force-recompute", action="store_true",
        help="Ignore existing KS cache and recompute from scratch.",
    )
    p.add_argument(
        "--significance", type=float, default=0.05,
        help="P-value threshold for KS test. Default: 0.05",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    z = args.redshift
    z_cfg = REDSHIFT_CONFIGS[z]

    ks_cfg = KSConfig(
        n_trials=2000,
        max_sample=100,
        significance=args.significance,
        summary_percentile=90.0,
    )

    cache_fid, cache_stoc = CACHE_FILES[z]
    output_dir = OUTPUT_ROOT / f"z{z}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Redshift: {z}")
    log.info(f"Fiducial cache:   {cache_fid}")
    log.info(f"Stochastic cache: {cache_stoc}")
    log.info(f"Output dir:       {output_dir}")

    log.info("Loading d1s from cache ...")
    d1s_fid  = load_d1s(cache_fid,  cfg)
    d1s_stoc = load_d1s(cache_stoc, cfg)
    log.info("Computing bright counts for p_neighbor correction ...")
    bright_counts      = compute_bright_counts(z_cfg, cfg, n_realizations=N_REALIZATIONS[z])
    bright_counts_stoc = compute_bright_counts(z_cfg, cfg, n_realizations=N_REALIZATIONS[z], use_stochastic=True)

    log.info("Computing UVLF from fiducial catalog ...")
    muvs_all = load_muv_catalog(z_cfg.muv_fiducial_path, index=None, n_realizations=N_REALIZATIONS[z])
    uvlf_bins, uvlf_phi = get_LF(muvs_all, n_realizations=N_REALIZATIONS[z])
    log.info(f"  UVLF computed: {N_REALIZATIONS[z]} realizations, BOX_LEN={_BOX_LEN_MPC} Mpc")
    for bkey, n in bright_counts.items():
        log.info(f"  {bkey}: {n} bright galaxies")
    plt.style.use("seaborn-v0_8-ticks")
    plt.rcParams.update({
        "font.size": 14, "xtick.top": True, "ytick.right": True,
        "xtick.direction": "in", "ytick.direction": "in",
    })

    for bright_limit, bright_key in zip(cfg.bright_limits, cfg.bright_names):
        log.info(f"Running KS/AD analysis: bright_key={bright_key} ...")
        t0 = time.perf_counter()
        sig_str = str(args.significance).replace(".", "p")
        cache_path = output_dir / f"ks_results_{bright_key}_z{z}_sig{sig_str}.npz"
        if cache_path.exists() and not args.force_recompute:
            log.info(f"  {bright_key} already cached, skipping.")
            # still need to load for plotting
            archive = np.load(cache_path)
            results = {fkey: {'ks': archive[f"{fkey}__ks"], 'ad': archive[f"{fkey}__ad"]}
                       for fkey in cfg.faint_names}
        else:
            results = run_ks_analysis(
                d1s_fid, d1s_stoc, cfg, ks_cfg,
                bright_key=bright_key,
                seed=42,
            )

            log.info(f"  Done in {time.perf_counter() - t0:.1f}s")
            cache_path = output_dir / f"ks_results_{bright_key}_z{z}.npz"
            np.savez(cache_path, **{f"{fkey}__{test}": results[fkey][test]
                                    for fkey in results
                                    for test in ['ks', 'ad']})
        results_corrected = apply_p_neighbor_correction(results, d1s_fid, bright_counts, bright_key,
                                                         d1s_stoc=d1s_stoc, bright_counts_stoc=bright_counts_stoc)

        print(f"\n{'='*68}")
        print(f"bright_key={bright_key}  z={z}")
        print(summarise_ks(results, ks_cfg))

        pct = ks_cfg.summary_percentile
        print(f"\nRequired survey area  (bright_key={bright_key}, z={z}, Δz=1.0, BOX_LEN={_BOX_LEN_MPC} Mpc)")
        print(f"  {'faint_key':<12}  {'N_point med':>11}  {'Area med [arcmin²]':>20}  {'Area p'+str(int(pct))+' [arcmin²]':>20}")
        print(f"  {'-'*67}")
        for fkey in cfg.faint_names:
            n_pt   = results_corrected[fkey]['ks']
            areas  = compute_required_survey_area(n_pt, uvlf_bins, uvlf_phi, bright_limit, z)
            print(f"  {fkey:<12}  {np.nanmedian(n_pt):>11.1f}  "
                  f"{np.nanmedian(areas):>20.1f}  "
                  f"{np.nanpercentile(areas, pct):>20.1f}")

        fig = plot_ks_results(
            results_corrected, ks_cfg, bright_key=bright_key, redshift_label=z,
        )
        fig.savefig(output_dir / f"ks_hist_{bright_key}_z{z}.pdf", bbox_inches="tight")
        plt.close(fig)

        fig = plot_ks_summary_bars(
            results_corrected, ks_cfg, bright_key=bright_key, redshift_label=z,
        )
        fig.savefig(output_dir / f"ks_bars_{bright_key}_z{z}.pdf", bbox_inches="tight")
        plt.close(fig)

        log.info(f"  Saved plots for {bright_key}")

    log.info("All done.")


if __name__ == "__main__":
    main()
