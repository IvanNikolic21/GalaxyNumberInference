#!/usr/bin/env python
"""
check_bias_dilution.py
------------------------
Test whether M_halo -> M_UV scatter dilutes the *effective* bias of
M_UV-selected galaxies relative to halos of the same abundance.

check_halo_bias.py showed halos in this catalog cluster correctly for their
mass (if anything slightly *more* strongly than Tinker et al. 2010 predicts)
-- ruling out an under-clustered halo catalog as the explanation for our low
sigma_CV vs PANORAMIC/UniverseMachine. The next candidate: even with
correctly-biased halos, if M_UV selection at a given threshold pulls in a
wider range of halo masses than a clean mass cut (because of M_halo->M_UV
scatter -- the whole premise of the "stochastic" model, and plausibly present
in pre-JWST's UM-matching too), the *effective* bias of that M_UV-selected
sample can come out lower than a mass-threshold cut at the same abundance,
simply by including lower-mass, less-biased halos that happen to scatter
bright.

Method
------
For each (model, M_UV threshold, z-range):
    1. Build the galaxy pointing pool exactly as run_cosmic_variance.py does
       (`build_pointing_pool`) and take its full-pool method-of-moments
       sigma_CV (`fractional_cosmic_variance` on `pool_moments`) -- the same
       precise full-pool point estimate already used elsewhere in this
       project as the "ground truth" benchmark (e.g. the precision-check
       cross-reference for the Gamma/NB fits).
    2. Build halo-mass-threshold pointing pools (same LOS depth + footprint
       tiling) over a fine grid of mass thresholds, and find the one whose
       mean count *per pointing* matches the galaxy pool's mean count --
       abundance-matched directly in "counts per pointing" units, since both
       use the identical pencil-beam geometry (no comoving-density
       conversion needed).
    3. b_galaxy = sigma_CV_galaxy / sigma_DM_lin(window)
       b_halo    = sigma_CV_halo(abundance-matched) / sigma_DM_lin(window)
       using the *same* sigma_DM_lin (computed once per z-range via
       check_box_missing_power.sigma2_true_aniso, since it only depends on
       survey geometry, not on model/threshold).
    4. ratio = b_galaxy / b_halo. Well below 1 -- and more so for stochastic
       than fiducial -- is the scatter-dilution signature.

Uses the *effective* (possibly box-truncated) LOS depth throughout, for both
the pointing pools and sigma_DM_lin's window: unlike check_box_missing_power.py
(which deliberately compared against the untruncated, intended survey depth
to assess the box's own shortfall), this script compares two quantities
measured from the *same* box, so both sides need the *same* window that
actually produced the data.

Usage
-----
    python check_bias_dilution.py --n-realizations 30
"""

import argparse

import numpy as np

from galaxy_neighbors import load_halo_catalog
from cosmic_variance import (
    PointingConfig, comoving_depth_mpc, footprint_side_mpc, effective_los_depth,
    build_pointing_pool, pool_moments, fractional_cosmic_variance,
)
from check_box_missing_power import sigma2_true_aniso, BOX_REDSHIFT
from run_analysis import REDSHIFT_CONFIGS

Z_RANGES = {
    "z9p2_10p9": (9.2, 10.9),
    "z8p6_11p3": (8.6, 11.3),
}


def count_pointings_by_mass(
    halo_coords: np.ndarray, log10_masses: np.ndarray, mass_thresholds: list,
    cfg: PointingConfig, depth_mpc: float, footprint_side: float,
) -> np.ndarray:
    """Counts-in-cells for cumulative halo-mass thresholds (M_halo > threshold),
    same LOS-slice + transverse-tiling convention as
    cosmic_variance.count_pointings_for_realization, but for one fixed halo
    catalog (no MUV-realization loop -- halo positions/masses are
    deterministic)."""
    los = halo_coords[:, cfg.los_axis]
    in_slice = los < depth_mpc
    tx, ty = cfg.transverse_axes
    coords_t = halo_coords[in_slice][:, [tx, ty]]
    masses_s = log10_masses[in_slice]

    n_per_side = int(cfg.box_len_mpc // footprint_side)
    edges = np.arange(n_per_side + 1) * footprint_side

    counts = np.empty((n_per_side, n_per_side, len(mass_thresholds)))
    for k, threshold in enumerate(mass_thresholds):
        sel = masses_s > threshold
        hist, _, _ = np.histogram2d(coords_t[sel, 0], coords_t[sel, 1], bins=[edges, edges])
        counts[:, :, k] = hist
    return counts.reshape(-1, len(mass_thresholds))


def matched_mass_threshold(
    halo_coords: np.ndarray, log10_masses: np.ndarray, cfg: PointingConfig,
    depth_mpc: float, footprint_side: float, target_mean: float,
) -> float:
    """log10(M_halo) threshold whose mean count-per-pointing matches `target_mean`.

    Grid is rank-based (log-spaced from the single rarest halo in the slice
    up to all of them), not percentile-based: with O(1e7) halos in the slice,
    even the 99.9th percentile still corresponds to tens of thousands of
    halos -- far more abundant than some M_UV-selected thresholds (which can
    have <N> << 1 per pointing). A percentile grid that doesn't reach far
    enough into the tail makes every such target_mean fall below the grid's
    achievable minimum, and np.interp silently *clips* out-of-range inputs to
    the boundary value rather than erroring -- so every threshold below the
    grid's reach collapses to the same (wrong) matched mass silently.
    """
    los = halo_coords[:, cfg.los_axis]
    masses_s = log10_masses[los < depth_mpc]
    n_s = len(masses_s)
    sorted_desc = np.sort(masses_s)[::-1]
    ranks = np.unique(np.geomspace(3, n_s - 1, 100).astype(int))
    grid = sorted_desc[ranks]

    pool = count_pointings_by_mass(halo_coords, log10_masses, list(grid), cfg, depth_mpc, footprint_side)
    mean_halo = pool.mean(axis=0)
    if not (mean_halo.min() <= target_mean <= mean_halo.max()):
        print(
            f"  WARNING: target_mean={target_mean:.5f} outside the abundance-matchable "
            f"range [{mean_halo.min():.5f}, {mean_halo.max():.5f}] for this slice -- "
            "result below is clipped to the nearest achievable threshold, not a true match."
        )
    order = np.argsort(mean_halo)
    return float(np.interp(target_mean, mean_halo[order], grid[order]))


def parse_args():
    p = argparse.ArgumentParser(description="Galaxy vs. abundance-matched halo bias (scatter-dilution check)")
    p.add_argument("--n-realizations", type=int, default=30,
                   help="MUV realizations for fiducial/stochastic (full-pool MoM sigma_CV).")
    p.add_argument("--include-prejwst", action="store_true",
                   help="Also check the pre-JWST model (its faintest threshold, M_UV<-20.5, "
                        "is already known to be data-starved -- treat that row cautiously).")
    return p.parse_args()


def main():
    args = parse_args()
    z_cfg = REDSHIFT_CONFIGS[10.5]
    cfg = PointingConfig()

    print(f"Loading halo catalog: {z_cfg.halo_catalog_path}")
    halo_coords, log10_masses = load_halo_catalog(z_cfg.halo_catalog_path)
    print(f"  {len(log10_masses)} halos with M > 0\n")

    footprint_side = footprint_side_mpc(BOX_REDSHIFT, cfg.fov_area_arcmin2)

    models = {"fiducial": (z_cfg.muv_fiducial_path, args.n_realizations),
              "stochastic": (z_cfg.muv_stochastic_path, args.n_realizations)}
    if args.include_prejwst:
        from run_cosmic_variance import PREJWST_MUV_PATH, PREJWST_N_REALIZATIONS
        models["prejwst"] = (PREJWST_MUV_PATH, PREJWST_N_REALIZATIONS)

    for zrange_label, (zlo, zhi) in Z_RANGES.items():
        depth_requested = comoving_depth_mpc(zlo, zhi)
        depth, truncated = effective_los_depth(depth_requested, cfg.box_len_mpc)
        print(f"=== {zrange_label}  z=({zlo}, {zhi})  effective depth={depth:.1f} Mpc"
              + ("  [TRUNCATED]" if truncated else "") + " ===")

        sigma_dm = np.sqrt(sigma2_true_aniso(footprint_side, footprint_side, depth, BOX_REDSHIFT, k_max=6.0, n_perp=200))
        print(f"sigma_DM_linear(window) = {sigma_dm:.5f}\n")

        header = (
            f"  {'model':<12} {'M_UV':>6} {'<N>_gal':>9} {'b_galaxy':>9} "
            f"{'logM_match':>10} {'<N>_halo':>9} {'b_halo':>8} {'ratio':>7}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        for model, (muv_path, n_real) in models.items():
            pool = build_pointing_pool(halo_coords, muv_path, n_real, cfg, depth, footprint_side)
            mean, mean_sq = pool_moments(pool)
            sigma_cv_gal = np.sqrt(fractional_cosmic_variance(mean, mean_sq))

            for k, threshold in enumerate(cfg.thresholds):
                target_mean = mean[k]
                logM_match = matched_mass_threshold(halo_coords, log10_masses, cfg, depth, footprint_side, target_mean)
                halo_pool = count_pointings_by_mass(halo_coords, log10_masses, [logM_match], cfg, depth, footprint_side)
                hmean, hmean_sq = pool_moments(halo_pool)
                sigma_cv_halo = float(np.sqrt(fractional_cosmic_variance(hmean, hmean_sq))[0])

                b_gal = sigma_cv_gal[k] / sigma_dm
                b_halo = sigma_cv_halo / sigma_dm
                ratio = b_gal / b_halo if b_halo > 0 else float("nan")
                print(
                    f"  {model:<12} {threshold:>6.1f} {mean[k]:>9.3f} {b_gal:>9.3f} "
                    f"{logM_match:>10.2f} {hmean[0]:>9.3f} {b_halo:>8.3f} {ratio:>7.2f}"
                )
        print()


if __name__ == "__main__":
    main()