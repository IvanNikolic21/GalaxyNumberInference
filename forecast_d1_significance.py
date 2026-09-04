#!/usr/bin/env python
"""
forecast_d1_significance.py
-----------------------------
Companion to forecast_count_significance.py, for the JWST proposal
discussion with Charlotte: forecast the MEAN distance to the nearest
neighbor (d1) as a function of the number of pointings N, with bootstrap
error bars, reported in arcmin.

Per Ivan's 2026-09-02 correction: the proposal targets NIRCam imaging
without spectroscopy, so 3D distances (this script's earlier version) are
not observable -- only angular position plus a photometric-redshift
window. This version instead uses the pencil-beam approach already
established in Paper I Sect. 3.4 ("Angular separation of galaxies"):
search a narrow angular aperture (half_side_xy, from
AnalysisConfig.survey_area_arcmin2, same convention/caveat as
forecast_count_significance.py) but a wide line-of-sight window
(half_side_z, from --photo-z-uncertainty, default Delta_z=0.5 -- Paper I's
own fiducial choice, motivated there by typical medium-band photo-z
uncertainty), then report the PROJECTED (2D, sqrt(dx^2+dy^2)) distance to
the nearest candidate within that pencil beam, not the 3D distance.

This bypasses run_neighbor_analysis/GalaxyModel.run() (which only ever
searches an isotropic box) and instead calls find_neighbors_in_box()
directly per bright galaxy -- the same direct-loop pattern already used in
build_nre_database.py's process_one() and check_theta_degeneracy.py's
d1_distribution_for_theta(), just computing a 2D distance from the
returned candidate coordinates instead of the 3D one.

For each of --n-trials bootstrap trials and each N in --n-values:
    draw N (2D, arcmin) d1 values (with replacement) from the model's
    pooled, filtered d1 array, take their mean.
This gives, per model per N, a distribution of "mean d1 at that N" -- its
spread IS the bootstrap error. Also reports a derived separation-in-sigma
= |mean_fid - mean_stoc| / sqrt(std_fid^2 + std_stoc^2), for direct
comparison with forecast_count_significance.py's log-likelihood-ratio
sigma.

Usage
-----
    # single depth
    python forecast_d1_significance.py \\
        --redshift 14.0 --area-arcmin2 4.84 --muv0 -20.5 --muvlim -18.0 \\
        --photo-z-uncertainty 0.5 \\
        --n-values 1 2 5 --n-realizations 100 --n-trials 2000 \\
        --output-dir /groups/astro/ivannik/projects/Neighbors/d1_forecast

    # depth sweep -- one panel per value in a single comparison figure,
    # plus one npz per depth
    python forecast_d1_significance.py \\
        --redshift 14.0 --area-arcmin2 4.84 --muv0 -20.5 \\
        --muvlim -18.0 -18.1 -18.2 -18.3 \\
        --photo-z-uncertainty 0.5 \\
        --n-values 1 2 5 --n-realizations 100 --n-trials 2000 \\
        --output-dir /groups/astro/ivannik/projects/Neighbors/d1_forecast
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

from galaxy_neighbors import AnalysisConfig, load_halo_catalog, load_muv_catalog, find_neighbors_in_box
from run_ks import REDSHIFT_CONFIGS, N_REALIZATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def available_realizations(path: Path) -> int:
    """Actual number of realizations stored in a muv catalog's 'data' dataset --
    same guard as forecast_count_significance.py, see that script for why."""
    import h5py
    with h5py.File(path, "r") as f:
        return f["data"].shape[0]


def pencil_beam_d1_arcmin(
    muv_path: Path, halo_coords: np.ndarray, bright_limit: float, faint_limit: float,
    half_side_xy: float, half_side_z_lower: float, half_side_z_upper: float,
    n_realizations: int, redshift: float, min_neighbors: int,
) -> np.ndarray:
    """Pooled, projected (2D) d1 values [arcmin] across n_realizations mock
    catalogs, using a pencil-beam search (narrow xy, wide z) rather than an
    isotropic box -- see module docstring."""
    kpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(redshift).to(u.Mpc / u.arcmin).value

    d1_values_mpc = []
    for idx in range(n_realizations):
        muvs = load_muv_catalog(muv_path, index=idx)
        bright_mask = muvs < bright_limit
        faint_mask  = (muvs < faint_limit) & (muvs >= bright_limit)

        bright_coords = halo_coords[bright_mask]
        faint_coords  = halo_coords[faint_mask]
        faint_mags    = muvs[faint_mask]

        for bright_coord in bright_coords:
            _, matched_coords, _ = find_neighbors_in_box(
                bright_coord, faint_coords, faint_mags,
                half_side=half_side_xy, faint_limit=faint_limit,
                half_side_z_lower=half_side_z_lower, half_side_z_upper=half_side_z_upper,
            )
            if len(matched_coords) < min_neighbors:
                continue
            d2d = np.sqrt((matched_coords[:, 0] - bright_coord[0]) ** 2 +
                          (matched_coords[:, 1] - bright_coord[1]) ** 2)
            d1_values_mpc.append(d2d.min())

    d1_values_mpc = np.array(d1_values_mpc)
    return d1_values_mpc / kpc_per_arcmin  # cMpc -> arcmin


def bright_halo_masses(muv_path: Path, halo_logmhs: np.ndarray, bright_limit: float,
                        n_realizations: int) -> np.ndarray:
    """Pooled log10(M_h/Msun) of halos hosting M_UV < bright_limit galaxies,
    across n_realizations mock luminosity draws on the same fixed halo
    catalog -- same selection Paper I Sect. 4.1 uses for the real GN-z11
    halo-mass estimate, just applied here to the model population instead
    of one specific object. Depends only on bright_limit, not on the faint
    depth, so this is computed once per model, not per --muvlim panel."""
    logmhs_pooled = []
    for idx in range(n_realizations):
        muvs = load_muv_catalog(muv_path, index=idx)
        logmhs_pooled.append(halo_logmhs[muvs < bright_limit])
    return np.concatenate(logmhs_pooled)


def bootstrap_mean_vs_n(d1_arr: np.ndarray, n_values: list[int], n_trials: int,
                         rng: np.random.Generator) -> dict[int, np.ndarray]:
    """For each N, return an array of shape (n_trials,) of bootstrap sample means."""
    out = {}
    for N in n_values:
        means = np.empty(n_trials)
        for t in range(n_trials):
            draw = rng.choice(d1_arr, size=N, replace=True)
            means[t] = draw.mean()
        out[N] = means
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--redshift", type=float, required=True, choices=sorted(REDSHIFT_CONFIGS.keys()))
    p.add_argument("--area-arcmin2", type=float, required=True,
                   help="Sets the angular (xy) aperture half-width -- see the aperture-convention "
                        "note in forecast_count_significance.py (same 4x-area caveat applies).")
    p.add_argument("--muv0", type=float, required=True)
    p.add_argument("--muvlim", type=float, nargs="+", required=True,
                   help="One or more faint-neighbor depths to sweep -- each gets its own bootstrap "
                        "and npz, plus one shared multi-panel comparison figure.")
    p.add_argument("--photo-z-uncertainty", type=float, default=0.5,
                   help="Line-of-sight pencil-beam half-width, as a redshift interval Delta_z. "
                        "Default 0.5 is Paper I Sect. 3.4's own fiducial choice, motivated there by "
                        "typical medium-band photo-z uncertainty.")
    p.add_argument("--min-neighbors", type=int, default=1,
                   help="Minimum number of candidates found within the pencil beam for a bright "
                        "galaxy to contribute a d1 value at all.")
    p.add_argument("--n-realizations", type=int, default=None)
    p.add_argument("--n-values", type=int, nargs="+", default=[1, 2, 5])
    p.add_argument("--n-trials", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    z_cfg = REDSHIFT_CONFIGS[args.redshift]
    n_realizations = args.n_realizations or N_REALIZATIONS[args.redshift]

    n_avail_fid  = available_realizations(z_cfg.muv_fiducial_path)
    n_avail_stoc = available_realizations(z_cfg.muv_stochastic_path)
    n_avail = min(n_avail_fid, n_avail_stoc)
    if n_realizations > n_avail:
        log.warning(f"Requested n_realizations={n_realizations}, but fiducial catalog has "
                    f"{n_avail_fid} and stochastic has {n_avail_stoc} available -- "
                    f"clipping to {n_avail}.")
        n_realizations = n_avail

    half_side_xy = AnalysisConfig(survey_area_arcmin2=args.area_arcmin2).search_box_mpc(args.redshift)

    dz = args.photo_z_uncertainty
    half_side_z_upper = (cosmo.comoving_distance(args.redshift + dz)
                          - cosmo.comoving_distance(args.redshift)).to(u.Mpc).value
    half_side_z_lower = (cosmo.comoving_distance(args.redshift)
                          - cosmo.comoving_distance(args.redshift - dz)).to(u.Mpc).value

    log.info(f"z={args.redshift}  area={args.area_arcmin2} arcmin^2  M_UV,0={args.muv0}  "
             f"M_UV,lim={args.muvlim}  n_realizations={n_realizations}")
    log.info(f"Pencil beam: half_side_xy={half_side_xy:.2f} cMpc, "
             f"half_side_z=[{half_side_z_lower:.1f},{half_side_z_upper:.1f}] cMpc "
             f"(Delta_z={dz})")

    log.info("Loading halo catalog (shared across all realizations and depths) ...")
    halo_coords, halo_logmhs = load_halo_catalog(z_cfg.halo_catalog_path)

    log.info("Computing halo masses of M_UV,0-bright galaxies for both models ...")
    logmh_fid  = bright_halo_masses(z_cfg.muv_fiducial_path,   halo_logmhs, args.muv0, n_realizations)
    logmh_stoc = bright_halo_masses(z_cfg.muv_stochastic_path, halo_logmhs, args.muv0, n_realizations)
    p16_mh_fid,  p50_mh_fid,  p84_mh_fid  = np.percentile(logmh_fid,  [16, 50, 84])
    p16_mh_stoc, p50_mh_stoc, p84_mh_stoc = np.percentile(logmh_stoc, [16, 50, 84])
    print(f"\nHalo masses hosting M_UV < {args.muv0} galaxies at z={args.redshift} "
          f"(same selection as Paper I Sect. 4.1's GN-z11 estimate):")
    print(f"  fiducial:   log(M_h/Msun) = {p50_mh_fid:.2f} [{p16_mh_fid:.2f},{p84_mh_fid:.2f}] (68% C.I.), "
          f"n={len(logmh_fid)}")
    print(f"  stochastic: log(M_h/Msun) = {p50_mh_stoc:.2f} [{p16_mh_stoc:.2f},{p84_mh_stoc:.2f}] (68% C.I.), "
          f"n={len(logmh_stoc)}")

    plt.style.use("seaborn-v0_8-ticks")
    plt.rcParams.update({"font.size": 16, "xtick.top": True, "ytick.right": True,
                         "xtick.direction": "in", "ytick.direction": "in"})
    n_panels = len(args.muvlim)
    if n_panels == 1:
        fig, axes = plt.subplots(1, n_panels, figsize=(7.0,7.0), sharey=True, squeeze=False)
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5), sharey=True, squeeze=False)
    axes = axes[0]
    x_offset = 0.08

    for panel_i, muvlim in enumerate(args.muvlim):
        cfg = AnalysisConfig(bright_limits=[args.muv0], faint_limits=[muvlim],
                              preselect_faint_limit=muvlim, survey_area_arcmin2=args.area_arcmin2)
        bright_key, faint_key = cfg.bright_names[0], cfg.faint_names[0]

        log.info(f"--- M_UV,lim={muvlim} ---")
        log.info("Computing projected d1 (fiducial) ...")
        d1_fid = pencil_beam_d1_arcmin(
            z_cfg.muv_fiducial_path, halo_coords, args.muv0, muvlim,
            half_side_xy, half_side_z_lower, half_side_z_upper,
            n_realizations, args.redshift, args.min_neighbors,
        )
        log.info("Computing projected d1 (stochastic) ...")
        d1_stoc = pencil_beam_d1_arcmin(
            z_cfg.muv_stochastic_path, halo_coords, args.muv0, muvlim,
            half_side_xy, half_side_z_lower, half_side_z_upper,
            n_realizations, args.redshift, args.min_neighbors,
        )
        log.info(f"  fiducial:   {len(d1_fid)} usable environments, mean d1={d1_fid.mean():.3f} arcmin, "
                 f"std={d1_fid.std():.3f}")
        log.info(f"  stochastic: {len(d1_stoc)} usable environments, mean d1={d1_stoc.mean():.3f} arcmin, "
                 f"std={d1_stoc.std():.3f}")

        log.info(f"Bootstrapping mean d1 vs N for N={args.n_values}, {args.n_trials} trials each ...")
        boot_fid  = bootstrap_mean_vs_n(d1_fid,  args.n_values, args.n_trials, rng)
        boot_stoc = bootstrap_mean_vs_n(d1_stoc, args.n_values, args.n_trials, rng)

        print(f"\nMean d1 [arcmin] vs N, bootstrapped ({args.n_trials} trials/N), "
              f"z={args.redshift}, M_UV,0={args.muv0}, M_UV,lim={muvlim}, area={args.area_arcmin2} arcmin^2, "
              f"Delta_z={dz}")
        print(f"{'N':>4}  {'fiducial: median [16,84]':>28}  {'stochastic: median [16,84]':>28}  "
              f"{'overlap?':>10}  {'separation [sigma]':>18}")
        print("-" * 96)
        for N in args.n_values:
            f, s = boot_fid[N], boot_stoc[N]
            lo_f, med_f, hi_f = np.percentile(f, [16, 50, 84])
            lo_s, med_s, hi_s = np.percentile(s, [16, 50, 84])
            overlap = not (hi_f < lo_s or hi_s < lo_f)  # 68% bands overlap?
            sep_sigma = abs(med_f - med_s) / np.sqrt(f.std() ** 2 + s.std() ** 2)
            print(f"{N:>4}  {med_f:>8.3f} [{lo_f:.3f},{hi_f:.3f}]      "
                  f"{med_s:>8.3f} [{lo_s:.3f},{hi_s:.3f}]      "
                  f"{'yes' if overlap else 'no':>10}  {sep_sigma:>18.2f}")

        np.savez(
            args.output_dir / f"d1_meanboot_arcmin_z{args.redshift}_M{bright_key}_lim{faint_key}.npz",
            n_values=np.array(args.n_values),
            **{f"fid_N{N}": boot_fid[N] for N in args.n_values},
            **{f"stoc_N{N}": boot_stoc[N] for N in args.n_values},
            d1_fid=d1_fid, d1_stoc=d1_stoc, photo_z_uncertainty=dz,
        )

        # --------------------------------------------------------------
        # This depth's panel -- stochastic model's points shifted slightly
        # in x so the two error bars don't sit on top of each other.
        # --------------------------------------------------------------
        ax = axes[panel_i]
        for boot, color, label, dx in [(boot_fid, "#d94701", "high luminosity", -x_offset / 2),
                                        (boot_stoc, "#2171b5", "high stochasticity", +x_offset / 2)]:
            meds = np.array([np.percentile(boot[N], 50) for N in args.n_values])
            los  = np.array([np.percentile(boot[N], 16) for N in args.n_values])
            his  = np.array([np.percentile(boot[N], 84) for N in args.n_values])
            x = np.array(args.n_values, dtype=float) + dx
            ax.errorbar(x, meds, yerr=[meds - los, his - meds],
                         fmt="o-", color=color, label=label, capsize=4)
        ax.set_xlabel("Number of bright galaxies")
        ax.set_xticks(args.n_values)
        #ax.set_title(rf"$M_{{\rm UV,lim}}={muvlim}$", fontsize=12)
        if panel_i == 0:
            ax.set_ylabel(r"mean separation to \nnearest neighbor [arcmin]")
            ax.legend(fontsize=14, frameon=False)

    # fig.suptitle(f"z={args.redshift}, area={args.area_arcmin2} arcmin$^2$, "
    #              rf"$M_{{\rm UV,0}}={args.muv0}$, $\Delta z={dz}$" "\n"
    #              rf"$\log(M_h/M_\odot)$: fid$={p50_mh_fid:.2f}$, stoc$={p50_mh_stoc:.2f}$ (68% C.I.)",
    #              fontsize=13 )
    fig.tight_layout()
    muvlim_tag = "-".join(f"{m}" for m in args.muvlim)
    fig.savefig(args.output_dir / f"d1_meanboot_arcmin_z{args.redshift}_muvlimsweep_{muvlim_tag}.pdf")
    log.info(f"Saved comparison plot + per-depth npz files to {args.output_dir}")


if __name__ == "__main__":
    main()