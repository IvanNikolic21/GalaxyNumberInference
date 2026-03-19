#!/usr/bin/env python
"""
plot_ks_imshow.py
-----------------
One imshow per redshift (z=10.5, 12, 14) of the p_neighbor-corrected
median number of pointings needed to distinguish the two models.
Colorbar fixed at vmin=10, vmax=100. Cyan line = best Muv,lim per Muv,0.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from galaxy_neighbors import AnalysisConfig, RedshiftConfig, compute_bright_counts
from galaxy_d1s import load_d1s

cfg = AnalysisConfig(
    bright_limits         = [-20.5, -20.75, -21.0, -21.25, -21.5, -21.75, -22.0],
    faint_limits          = [-16.5, -16.6, -16.7, -16.8, -16.9, -17.0, -17.1, -17.2,
                             -17.3, -17.4, -17.5, -17.6, -17.7, -17.8, -17.9, -18.0,
                             -18.1, -18.2, -18.3, -18.4, -18.5, -18.6, -18.7, -18.8,
                             -18.9, -19.0, -19.1, -19.2, -19.3, -19.4, -19.5, -19.6],
    preselect_faint_limit = -16.5,
    survey_area_arcmin2   = 12.24,
)

_CACHE_BASE = "/lustre/astro/ivannik/21cmFAST_cache/d12b21e80b7885d62d31717c2c2d8421"
_HASH       = "ffa852ccaa39d8f82951cc98ff798ab4"

REDSHIFT_CONFIGS = {
    # 8.0: RedshiftConfig(redshift=8.0,
    #     halo_catalog_path=Path(f"{_CACHE_BASE}/1955/{_HASH}/8.0000/HaloCatalog.h5"),
    #     muv_fiducial_path=Path("/lustre/astro/ivannik/catalog_fiducial_bigger_z8.h5"),
    #     muv_stochastic_path=Path("/lustre/astro/ivannik/catalog_stoch_bigger_z8.h5")),
    10.5: RedshiftConfig(redshift=10.5,
        halo_catalog_path=Path(f"{_CACHE_BASE}/1952/{_HASH}/10.5000/HaloCatalog.h5"),
        muv_fiducial_path=Path("/lustre/astro/ivannik/catalog_fiducial_bigger_new_save.h5"),
        muv_stochastic_path=Path("/lustre/astro/ivannik/catalog_stoch_bigger_new3.h5")),
    # 12.0: RedshiftConfig(redshift=12.0,
    #     halo_catalog_path=Path(f"{_CACHE_BASE}/1955/{_HASH}/12.0000/HaloCatalog.h5"),
    #     muv_fiducial_path=Path("/lustre/astro/ivannik/catalog_fiducial_bigger_z12.h5"),
    #     muv_stochastic_path=Path("/lustre/astro/ivannik/catalog_stoch_bigger_z12.h5")),
    # 14.0: RedshiftConfig(redshift=14.0,
    #     halo_catalog_path=Path(f"{_CACHE_BASE}/1955/{_HASH}/14.0000/HaloCatalog.h5"),
    #     muv_fiducial_path=Path("/lustre/astro/ivannik/catalog_fiducial_bigger_z14_300.h5"),
    #     muv_stochastic_path=Path("/lustre/astro/ivannik/catalog_stoch_bigger_z14_300.h5")),
}

N_REALIZATIONS = {10.5:20}

CACHE_ROOT  = Path("/groups/astro/ivannik/projects/Neighbors/cache")
KS_ROOT     = Path("/groups/astro/ivannik/projects/Neighbors/lr_results")
OUTPUT_ROOT = Path("/groups/astro/ivannik/projects/Neighbors/lr_results")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

D1S_FILES = {
    # 8.0: CACHE_ROOT / "z8.0" / "d1s_fiducial_real1.npz",
    10.5: CACHE_ROOT / "z10.5" / "d1s_fiducial_real20.npz",
    # 12.0: CACHE_ROOT / "z12.0" / "d1s_fiducial_real50.npz",
    # 14.0: CACHE_ROOT / "z14.0" / "d1s_fiducial_real100.npz",
}

muv_lim = np.array(cfg.faint_limits)
muv_0   = np.array(cfg.bright_limits)
dx      = muv_lim[1] - muv_lim[0]
dy      = muv_0[1]   - muv_0[0]
extent  = (muv_lim[0] - dx/2, muv_lim[-1] + dx/2,
           muv_0[0]   - dy/2, muv_0[-1]   + dy/2)

VMIN = 10
VMAX = 100

plt.style.use("seaborn-v0_8-ticks")
plt.rcParams.update({"font.size": 14, "xtick.top": True, "ytick.right": True,
                     "xtick.direction": "in", "ytick.direction": "in"})

for z in [ 10.5]:
    print(f"Processing z={z} ...")
    ks_dir        = KS_ROOT / f"z{z}"
    d1s_fid       = load_d1s(D1S_FILES[z], cfg)
    bright_counts = compute_bright_counts(REDSHIFT_CONFIGS[z], cfg,
                                          n_realizations=N_REALIZATIONS[z])

    Ns = np.full((len(muv_0), len(muv_lim)), np.nan)
    print(f"  Found {len(bright_counts)} bright pointings.")
    for i, bkey in enumerate(cfg.bright_names):
        cache_path = ks_dir / f"lr_results_{bkey}_z{z}_sig0p05.npz"
        if not cache_path.exists():
            print(f"  Warning: {cache_path.name} not found, skipping.")
            continue
        archive = np.load(cache_path)
        n_total = bright_counts[bkey]
        print(f"  Found {n_total} total pointings.")
        for j, fkey in enumerate(cfg.faint_names):
            try:
                n_passed = len(d1s_fid[bkey][fkey])
                p = n_passed / n_total if n_total > 0 else 1.0
                val = np.nanmedian(archive[f"{fkey}__ks"])
                if p > 0 and np.isfinite(val):
                    Ns[i, j] = val / p
            except KeyError:
                pass  # leaves Ns[i, j] as nan

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(Ns, origin="lower", extent=extent, aspect="auto",
                   cmap="magma_r", interpolation="nearest", vmin=VMIN, vmax=VMAX)
    ns_path = OUTPUT_ROOT / f"Ns_z{z}.npz"
    np.savez(ns_path, Ns=Ns, muv_lim=muv_lim, muv_0=muv_0)
    print(f"  Saved Ns: {ns_path}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.set_ylabel("Median pointings needed", fontsize=13)
    ax.set_xlabel(r"$M_{\rm UV,lim}$", fontsize=16)
    ax.set_ylabel(r"$M_{\rm UV,0}$", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.set_title(rf"$z = {z}$", fontsize=16)

    # Only plot line for rows that have at least one valid value
    valid_rows = ~np.all(np.isnan(Ns), axis=1)
    best_xs = muv_lim[np.nanargmin(Ns[valid_rows], axis=1)]
    ax.plot(best_xs, muv_0[valid_rows], color="cyan", lw=2, marker="o", markersize=6,
            label=r"Best $M_{\rm UV,lim}$ per $M_{\rm UV,0}$")
    # ax.plot(best_xs, muv_0, color="cyan", lw=2, marker="o", markersize=6,
    #         label=r"Best $M_{\rm UV,lim}$ per $M_{\rm UV,0}$")
    ax.legend(fontsize=11, framealpha=0.7)

    fig.tight_layout()
    out = OUTPUT_ROOT / f"num_samp_z{z}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

print("All done.")