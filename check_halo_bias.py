#!/usr/bin/env python
"""
check_halo_bias.py
-------------------
Check whether halos in the 21cmFAST excursion-set catalog cluster as
strongly as their mass implies they should, by comparing an EMPIRICAL bias
(measured directly from counts-in-cells in the box) against the standard
analytic Tinker et al. (2010) halo bias relation at the same mass and
redshift.

Motivation
----------
check_box_missing_power.py ruled out "missing large-scale power from a
finite periodic box" as the dominant cause of our low sigma_CV relative to
PANORAMIC/UniverseMachine (only ~1.6-1.8% of the true variance is
structurally inaccessible). The next candidate: if this excursion-set halo
finder (run_ks.py / 21cmFAST) under-clusters halos relative to a full N-body
calculation at fixed mass -- i.e. produces halos that are genuinely less
biased than their mass should make them -- that directly suppresses
sigma_CV for every model built on this catalog (fiducial, stochastic,
pre-JWST alike), independent of the M_halo-M_UV painting.

Method
------
For each of several cumulative mass thresholds (chosen from this catalog's
own mass percentiles):
    1. Tile the box into n_per_side^3 cubic cells and count halos with
       M_halo > threshold per cell (pure 3D counts-in-cells, the whole box,
       no line-of-sight selection -- this is a question about the box's own
       3D clustering, not about any particular survey geometry).
    2. sigma_CV^2(threshold) = excess (over-Poisson) variance / <N>^2,
       reusing cosmic_variance.py's pool_moments/fractional_cosmic_variance.
    3. b_empirical(threshold) = sigma_CV(threshold) / sigma_DM_lin(R_cell, z)
       -- same b_g,CV = sigma_CV/sigma_DM definition Weibel et al. 2025 use
       (see cosmic_variance.PANORAMIC_SIGMA_CV docstring), with
       sigma_DM_lin from linear theory (BBKS P(k), reused from
       check_box_missing_power.py).
    4. b_Tinker(threshold) = Tinker et al. (2010) bias(nu) at the threshold
       mass's own Lagrangian scale -- the analytic prediction for what a
       halo of (at least) this mass *should* be biased by.

If b_empirical is systematically well below b_Tinker across thresholds,
that's direct evidence the halo finder under-clusters relative to mass,
which would explain low sigma_CV independent of galaxy-formation choices.

Additionally (unless --skip-galaxies), repeats the *same* whole-box,
same-cell-size, same sigma_DM_linear measurement for M_UV-selected galaxies
in each model -- there's no analytic b_Tinker equivalent for an M_UV cut, so
this table has no `ratio` column, but its b_empirical is directly comparable
to the halo table's b_empirical (same scale, same normalization), letting
you see e.g. whether stochastic's extra M_halo->M_UV scatter pulls galaxy
bias below halo bias at the same scale -- the same question
check_bias_dilution.py asks via abundance-matching, but at this fixed,
isotropic cell scale instead of the survey pencil-beam window.

Usage
-----
    python check_halo_bias.py --halo-catalog-path /path/to/HaloCatalog.h5
"""

import argparse

import numpy as np
from scipy.integrate import quad
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

from galaxy_neighbors import load_halo_catalog, load_muv_catalog
from cosmic_variance import PointingConfig, pool_moments, fractional_cosmic_variance, poisson_cosmic_variance
from check_box_missing_power import power_spectrum, _tophat_window, BOX_REDSHIFT

BOX_LEN_MPC = 512.0
N_PER_SIDE = 8  # 64 Mpc cells, 512 cells total -- a few % missing-power floor
# at this scale too (k_cell ~ 1/64 Mpc^-1 is not *that* far above the box's
# own k_min = 2*pi/512 ~ 0.0123 Mpc^-1), but the same order as already shown
# to be small for the survey geometry -- treat as a secondary caveat, not a
# reason to distrust an order-of-magnitude mismatch if one shows up.

DELTA_C = 1.686  # linear collapse threshold (spherical collapse)
OVERDENSITY = 200  # Tinker et al. 2010 Delta_mean definition

RHO_M0_MSUN_MPC3 = (cosmo.Om0 * cosmo.critical_density0).to(u.Msun / u.Mpc ** 3).value


def sigma2_tophat(R_mpc: float, z: float, k_max: float = 50.0) -> float:
    """Linear top-hat mass variance sigma^2(R, z) -- a plain isotropic 1D
    integral (no pencil-beam anisotropy issues here, see
    check_box_missing_power.py's sigma2_true_aniso for why that needed
    special handling)."""
    integrand = lambda k: k ** 2 * power_spectrum(k, z) * _tophat_window(k * R_mpc) ** 2
    val, _ = quad(integrand, 1e-6, k_max, limit=400)
    return val / (2 * np.pi ** 2)


def lagrangian_radius_mpc(mass_msun: float) -> float:
    """Comoving Lagrangian radius enclosing mass `mass_msun` at the mean
    comoving matter density."""
    return (3 * mass_msun / (4 * np.pi * RHO_M0_MSUN_MPC3)) ** (1 / 3)


def tinker_bias_2010(nu: np.ndarray, overdensity: float = OVERDENSITY) -> np.ndarray:
    """Tinker et al. (2010) ApJ 724, 878, Eq. 6 halo bias as a function of
    peak height nu = delta_c / sigma(M, z)."""
    y = np.log10(overdensity)
    A = 1.0 + 0.24 * y * np.exp(-(4 / y) ** 4)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4)
    c = 2.4
    return 1 - A * nu ** a / (nu ** a + DELTA_C ** a) + B * nu ** b + C * nu ** c


def count_cells_by_mass(
    halo_coords: np.ndarray, log10_masses: np.ndarray, thresholds: list,
    n_per_side: int, box_len_mpc: float,
) -> np.ndarray:
    """3D counts-in-cells for cumulative mass thresholds (M_halo > threshold),
    tiling the *whole* box (all 3 axes) into n_per_side^3 cubic cells."""
    edges = np.linspace(0.0, box_len_mpc, n_per_side + 1)
    counts = np.empty((n_per_side, n_per_side, n_per_side, len(thresholds)))
    for k, threshold in enumerate(thresholds):
        sel = log10_masses > threshold
        hist, _ = np.histogramdd(halo_coords[sel], bins=[edges, edges, edges])
        counts[..., k] = hist
    return counts.reshape(-1, len(thresholds))


def count_cells_by_muv(
    halo_coords: np.ndarray, muv_catalog_path, n_realizations: int, thresholds: list,
    n_per_side: int, box_len_mpc: float,
) -> np.ndarray:
    """3D counts-in-cells for cumulative M_UV thresholds (M_UV < threshold),
    same whole-box n_per_side^3 cubic-cell tiling as `count_cells_by_mass`,
    stacked across MUV realizations (same halo positions every time, just a
    fresh M_halo->M_UV draw per realization -- exactly mirroring how
    `build_pointing_pool` stacks realizations for the survey-window pools,
    just with this isotropic whole-box tiling instead of the pencil-beam one)."""
    edges = np.linspace(0.0, box_len_mpc, n_per_side + 1)
    rows = []
    for i in range(n_realizations):
        muvs = load_muv_catalog(muv_catalog_path, index=i)
        counts = np.empty((n_per_side, n_per_side, n_per_side, len(thresholds)))
        for k, threshold in enumerate(thresholds):
            sel = muvs < threshold
            hist, _ = np.histogramdd(halo_coords[sel], bins=[edges, edges, edges])
            counts[..., k] = hist
        rows.append(counts.reshape(-1, len(thresholds)))
    return np.concatenate(rows, axis=0)


def parse_args():
    p = argparse.ArgumentParser(description="Empirical vs. Tinker (2010) halo bias check, plus galaxy bias at the same scale")
    p.add_argument("--halo-catalog-path", type=str, default=None,
                   help="Defaults to REDSHIFT_CONFIGS[10.5].halo_catalog_path from run_analysis.py.")
    p.add_argument("--n-per-side", type=int, default=N_PER_SIDE)
    p.add_argument("--n-realizations", type=int, default=30,
                   help="MUV realizations per galaxy model, for the galaxy-bias table.")
    p.add_argument("--skip-galaxies", action="store_true",
                   help="Only run the halo-bias-vs-Tinker table (original behavior).")
    p.add_argument("--include-prejwst", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.halo_catalog_path is None:
        from run_analysis import REDSHIFT_CONFIGS
        halo_catalog_path = REDSHIFT_CONFIGS[10.5].halo_catalog_path
    else:
        halo_catalog_path = args.halo_catalog_path

    print(f"Loading halo catalog: {halo_catalog_path}")
    halo_coords, log10_masses = load_halo_catalog(halo_catalog_path)
    print(f"  {len(log10_masses)} halos with M > 0, "
          f"log10(M) range [{log10_masses.min():.2f}, {log10_masses.max():.2f}]")

    pctiles = [50, 70, 85, 93, 97, 99]
    thresholds = sorted(set(np.round(np.percentile(log10_masses, pctiles), 2)))

    cell_side = BOX_LEN_MPC / args.n_per_side
    R_cell = (3 * cell_side ** 3 / (4 * np.pi)) ** (1 / 3)  # equal-volume sphere
    sigma_dm = np.sqrt(sigma2_tophat(R_cell, BOX_REDSHIFT))

    pool = count_cells_by_mass(halo_coords, log10_masses, thresholds, args.n_per_side, BOX_LEN_MPC)
    mean, mean_sq = pool_moments(pool)
    _, _, var_cosmic = poisson_cosmic_variance(mean, mean_sq)
    sigma_cv = np.sqrt(fractional_cosmic_variance(mean, mean_sq))

    print(f"\nCell side: {cell_side:.1f} Mpc  (R_eff={R_cell:.2f} Mpc)  "
          f"sigma_DM_linear(R_eff, z={BOX_REDSHIFT}) = {sigma_dm:.5f}\n")

    header = f"  {'log10(M_thr)':>12} {'N(>thr)':>9} {'<N>/cell':>9} {'b_empirical':>12} {'b_Tinker':>9} {'ratio':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k, threshold in enumerate(thresholds):
        n_total = int((log10_masses > threshold).sum())
        mass_thr = 10 ** threshold
        nu = DELTA_C / np.sqrt(sigma2_tophat(lagrangian_radius_mpc(mass_thr), BOX_REDSHIFT))
        b_tinker = float(tinker_bias_2010(np.array([nu]))[0])
        b_emp = sigma_cv[k] / sigma_dm
        ratio = b_emp / b_tinker
        print(
            f"  {threshold:>12.2f} {n_total:>9d} {mean[k]:>9.3f} "
            f"{b_emp:>12.3f} {b_tinker:>9.3f} {ratio:>7.2f}"
        )

    if np.any(var_cosmic <= 0):
        print(
            "\nNote: var_cosmic clipped to 0 for at least one threshold (sub-Poisson "
            "counts) -- see poisson_cosmic_variance's warning above; b_empirical=0 "
            "there is a floor artifact, not a real measurement."
        )

    if args.skip_galaxies:
        return

    from run_analysis import REDSHIFT_CONFIGS
    z_cfg = REDSHIFT_CONFIGS[10.5]
    muv_thresholds = PointingConfig().thresholds  # [-19.5, -20.0, -20.5]

    models = {"fiducial": (z_cfg.muv_fiducial_path, args.n_realizations),
              "stochastic": (z_cfg.muv_stochastic_path, args.n_realizations)}
    if args.include_prejwst:
        from run_cosmic_variance import PREJWST_MUV_PATH, PREJWST_N_REALIZATIONS
        models["prejwst"] = (PREJWST_MUV_PATH, PREJWST_N_REALIZATIONS)

    print(
        f"\nGalaxy bias, *same* whole-box {cell_side:.1f} Mpc cells and same "
        f"sigma_DM_linear = {sigma_dm:.5f} as the halo table above "
        "(no analytic b_Tinker equivalent for an M_UV cut -- this is just "
        "b_empirical, directly comparable to the halo b_empirical column, "
        "NOT to b_Tinker):\n"
    )
    header2 = f"  {'model':<12} {'M_UV':>6} {'<N>/cell':>9} {'b_empirical':>12}"
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for model, (muv_path, n_real) in models.items():
        pool_gal = count_cells_by_muv(halo_coords, muv_path, n_real, muv_thresholds, args.n_per_side, BOX_LEN_MPC)
        mean_gal, mean_sq_gal = pool_moments(pool_gal)
        sigma_cv_gal = np.sqrt(fractional_cosmic_variance(mean_gal, mean_sq_gal))
        for k, threshold in enumerate(muv_thresholds):
            b_emp_gal = sigma_cv_gal[k] / sigma_dm
            print(f"  {model:<12} {threshold:>6.1f} {mean_gal[k]:>9.3f} {b_emp_gal:>12.3f}")


if __name__ == "__main__":
    main()