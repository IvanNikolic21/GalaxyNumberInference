"""
galaxy_d1s.py
-------------
Second stage of the galaxy neighbor analysis pipeline.

Takes the raw NeighborResult outputs from galaxy_neighbors.GalaxyModel.run()
and computes d1s — the distance from the brightest faint neighbor to its
nearest companion — after filtering for environments with at least 2 neighbors.

Pipeline
--------
    NeighborResult list  →  histogram filter  →  d1 per environment  →  save / plot

Usage
-----
    from galaxy_neighbors import AnalysisConfig, GalaxyModel
    from galaxy_d1s import D1sConfig, compute_d1s, save_d1s, load_d1s, plot_d1s_grid

    cfg = AnalysisConfig(...)
    results_fid  = GalaxyModel(...).run()
    results_stoc = GalaxyModel(...).run()

    d1s_cfg = D1sConfig(min_neighbors=2, bw_fid=0.18, bw_stoc=0.11)

    d1s_fid  = compute_d1s(results_fid,  cfg, d1s_cfg)
    d1s_stoc = compute_d1s(results_stoc, cfg, d1s_cfg)

    save_d1s(d1s_fid,  "d1s_fiducial.npz")
    save_d1s(d1s_stoc, "d1s_stochastic.npz")

    # Later, load from cache and plot — no recomputation needed
    d1s_fid  = load_d1s("d1s_fiducial.npz",  cfg)
    d1s_stoc = load_d1s("d1s_stochastic.npz", cfg)

    for bright_key in cfg.bright_names:
        fig = plot_d1s_grid(d1s_fid, d1s_stoc, cfg, d1s_cfg, bright_key=bright_key)
        fig.savefig(f"d1s_{bright_key}.pdf", bbox_inches="tight")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from scipy.stats import gaussian_kde

from galaxy_neighbors import AnalysisConfig, NeighborResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class D1sConfig:
    """Tunable parameters for the d1s computation and plotting stage.

    Parameters
    ----------
    min_neighbors : int
        Minimum number of faint neighbors a bright galaxy must have
        (total across the whole search box) to be included. Default: 2.
    n_bins : int
        Number of bins for the distance histogram used in the neighbor
        count filter. Default: 5 (matching the original 6-edge linspace).
    plot_d_max : float
        Maximum d1 value shown on the x-axis of KDE plots [cMpc].
    bw_fid : float
        KDE bandwidth for the fiducial model. Default: 0.18.
    bw_stoc : float
        KDE bandwidth for the stochastic model. Default: 0.11.
    color_fid : str
        Line color for the fiducial model in plots.
    color_stoc : str
        Line color for the stochastic model in plots.
    label_fid : str
        Legend label for the fiducial model.
    label_stoc : str
        Legend label for the stochastic model.
    """

    min_neighbors: int = 2
    n_bins: int = 5
    plot_d_max: float = 8.0
    bw_fid: float = 0.18
    bw_stoc: float = 0.11
    color_fid: str = "#a63603"
    color_stoc: str = "#08519c"
    label_fid: str = "intrinsically bright"
    label_stoc: str = "increased stochasticity"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_d1_single(neighbor: NeighborResult) -> float:
    """Compute d1 for a single bright galaxy environment.

    d1 is the distance from the brightest faint neighbor (by MUV) to its
    nearest companion among the other faint neighbors.

    Steps
    -----
    1. Sort neighbors by MUV (ascending = brightest first).
    2. Find the neighbor closest in 3-D space to the brightest one.
    3. Return the distance between them.

    Parameters
    ----------
    neighbor : NeighborResult
        Must have at least 2 entries.

    Returns
    -------
    float
        d1 in Mpc (comoving).
    """
    mags = neighbor.faint_mags
    coords = neighbor.faint_coords  # shape (N, 3)

    # Step 1: sort by magnitude (brightest = most negative = index 0 after sort)
    order = np.argsort(mags)
    mags_sorted = mags[order]       # noqa: F841 (kept for clarity)
    coords_sorted = coords[order]

    # Step 2: distances from the brightest neighbor to all others
    dists_from_brightest = np.sqrt(
        ((coords_sorted[0] - coords_sorted) ** 2).sum(axis=1)
    )

    # Step 3: sort again by distance to find the nearest companion
    # (index 0 will be the point itself with dist=0, so take index 1)
    dist_order = np.argsort(dists_from_brightest)
    nearest_companion_coord = coords_sorted[dist_order[1]]
    d1 = np.sqrt(((coords_sorted[0] - nearest_companion_coord) ** 2).sum())
    return float(d1)


def _neighbor_passes_filter(neighbor: NeighborResult, bins: np.ndarray, min_neighbors: int) -> bool:
    """Return True if the environment has >= min_neighbors in the distance histogram."""
    if neighbor.n_neighbors < min_neighbors:
        return False
    hist, _ = np.histogram(neighbor.distances, bins=bins)
    return int(hist.sum()) >= min_neighbors


def compute_d1s(
    results: dict[str, dict[str, list[NeighborResult]]],
    cfg: AnalysisConfig,
    redshift_cfg,
    d1s_cfg: Optional[D1sConfig] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute d1 values for all (bright_limit, faint_limit) combinations.

    Parameters
    ----------
    results : dict
        Output of GalaxyModel.run() —
        results[bright_key][faint_key] = list[NeighborResult].
    cfg : AnalysisConfig
        Must match the config used to produce `results`.
    d1s_cfg : D1sConfig, optional
        Defaults to D1sConfig() if not provided.

    Returns
    -------
    d1s : dict
        d1s[bright_key][faint_key] = np.ndarray of d1 values [cMpc],
        one entry per bright galaxy that passed the neighbor count filter.
    """
    if d1s_cfg is None:
        d1s_cfg = D1sConfig()

    half_side = cfg.search_box_mpc(redshift_cfg.redshift)

    bins = np.linspace(0.1, half_side * np.sqrt(2), d1s_cfg.n_bins + 1)

    d1s: dict[str, dict[str, np.ndarray]] = {
        bkey: {} for bkey in cfg.bright_names
    }

    for bkey in cfg.bright_names:
        for fkey in cfg.faint_names:
            neighbors_list: list[NeighborResult] = results[bkey][fkey]

            values = []
            for neighbor in neighbors_list:
                if not _neighbor_passes_filter(neighbor, bins, d1s_cfg.min_neighbors):
                    continue
                values.append(_compute_d1_single(neighbor))

            d1s[bkey][fkey] = np.array(values)

    return d1s


# ---------------------------------------------------------------------------
# Save / load (cache)
# ---------------------------------------------------------------------------

def save_d1s(d1s: dict, path: str | Path) -> None:
    """Save d1s to a compressed numpy archive (.npz).

    The flat key format is ``bright_key__faint_key`` (double underscore).
    The recommended naming convention for the file itself is:
      - Single realization:      d1s_fiducial_idx3.npz
      - Explicit list:           d1s_fiducial_idx0-1-2.npz
      - First N realizations:    d1s_fiducial_real10.npz
    This is handled automatically by run_analysis.py via make_cache_name().

    Parameters
    ----------
    d1s : dict
        Output of compute_d1s().
    path : str or Path
        Output file path. The .npz extension is added automatically if absent.
    """
    path = Path(path)
    flat: dict[str, np.ndarray] = {}
    for bkey, fdict in d1s.items():
        for fkey, arr in fdict.items():
            flat[f"{bkey}__{fkey}"] = arr
    np.savez_compressed(path, **flat)
    print(f"Saved d1s → {path}.npz" if path.suffix != ".npz" else f"Saved d1s → {path}")


def load_d1s(path: str | Path, cfg: AnalysisConfig)  -> dict[str, dict[str, np.ndarray]]:
    """Load d1s from a .npz cache file.

    Parameters
    ----------
    path : str or Path
    cfg : AnalysisConfig
        Used to reconstruct the nested dict structure and validate keys.

    Returns
    -------
    d1s : dict
        Same structure as the output of compute_d1s().
    """
    path = Path(path)
    archive = np.load(path)

    d1s: dict[str, dict[str, np.ndarray]] = {
        bkey: {} for bkey in cfg.bright_names
    }

    for flat_key in archive.files:
        bkey, fkey = flat_key.split("__", maxsplit=1)
        if bkey not in d1s:
            raise KeyError(
                f"Key '{bkey}' from file not found in config.bright_names. "
                "Make sure you're loading with the same AnalysisConfig."
            )
        d1s[bkey][fkey] = archive[flat_key]

    return d1s


def load_or_compute_d1s(
    path: str | Path,
    results: Optional[dict],
    cfg: AnalysisConfig,
    redshift_cfg,
    d1s_cfg: Optional[D1sConfig] = None,
    force_recompute: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Load d1s from cache if available, otherwise compute and save.

    Parameters
    ----------
    path : str or Path
        Cache file path (.npz).
    results : dict or None
        Raw NeighborResult dict from GalaxyModel.run(). Only needed if
        the cache doesn't exist yet.
    cfg : AnalysisConfig
    d1s_cfg : D1sConfig, optional
    force_recompute : bool
        If True, always recompute even if the cache exists.

    Returns
    -------
    d1s : dict
    """
    path = Path(path)
    npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")

    if npz_path.exists() and not force_recompute:
        print(f"Loading d1s from cache: {npz_path}")
        return load_d1s(npz_path, cfg)

    if results is None:
        raise ValueError(
            f"Cache file '{npz_path}' not found and no results provided to compute from."
        )

    print("Computing d1s ...")
    d1s = compute_d1s(results, cfg, redshift_cfg, d1s_cfg)
    save_d1s(d1s, npz_path)
    return d1s


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_d1s_grid(
    d1s_fid: dict[str, dict[str, np.ndarray]],
    d1s_stoc: dict[str, dict[str, np.ndarray]],
    cfg: AnalysisConfig,
    d1s_cfg: Optional[D1sConfig] = None,
    bright_key: str = None,
    n_cols: int = 2,
    figsize_per_panel: tuple[float, float] = (9, 4),
    ylim: tuple[float, float] = (0, 1.62),
    redshift_label: float = 10.5,
) -> plt.Figure:
    """Plot a grid of KDE panels, one per faint magnitude limit.

    Each panel overlays the fiducial and stochastic d1 distributions for
    a single (bright_limit, faint_limit) combination.

    Parameters
    ----------
    d1s_fid, d1s_stoc : dicts
        Output of compute_d1s() or load_d1s().
    cfg : AnalysisConfig
    d1s_cfg : D1sConfig, optional
    bright_key : str, optional
        Which bright threshold to plot. Defaults to cfg.bright_names[0].
    n_cols : int
        Number of columns in the panel grid.
    figsize_per_panel : tuple
        (width, height) per panel in inches.
    ylim : tuple
        y-axis limits for all panels.
    redshift_label : float
        Redshift shown in the title annotation.

    Returns
    -------
    fig : matplotlib.Figure
    """
    if d1s_cfg is None:
        d1s_cfg = D1sConfig()
    if bright_key is None:
        bright_key = cfg.bright_names[0]

    n_faint = len(cfg.faint_names)
    n_rows = int(np.ceil(n_faint / n_cols))
    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )

    x = np.linspace(0, d1s_cfg.plot_d_max, 200)

    for idx, fkey in enumerate(cfg.faint_names):
        row, col = divmod(idx, n_cols)
        ax = axs[row, col]

        arr_fid  = d1s_fid[bright_key][fkey]
        arr_stoc = d1s_stoc[bright_key][fkey]

        if len(arr_fid) > 1:
            try:
                kde_fid = gaussian_kde(arr_fid, bw_method=d1s_cfg.bw_fid)
                ax.plot(x, kde_fid(x), color=d1s_cfg.color_fid, lw=3, label=d1s_cfg.label_fid)
            except Exception:
                pass

        if len(arr_stoc) > 1:
            try:
                kde_stoc = gaussian_kde(arr_stoc, bw_method=d1s_cfg.bw_stoc)
                ax.plot(x, kde_stoc(x), color=d1s_cfg.color_stoc, lw=3, label=d1s_cfg.label_stoc)
            except Exception:
                pass

        faint_mag = cfg.faint_limits[idx]
        ax.text(
            0.55, 0.65,
            rf"$M_{{\rm UV, lim}}={faint_mag}$",
            transform=ax.transAxes, fontsize=16,
        )
        ax.set_ylim(*ylim)
        ax.set_xlim(0, d1s_cfg.plot_d_max)
        ax.legend(fontsize=12)

        if col == 0:
            ax.set_ylabel(r"PDF (separation)", fontsize=16)
        if row == n_rows - 1:
            ax.set_xlabel(r"$d_{12}$ = distance to the closest neighbor [cMpc]", fontsize=16)

    # Title annotation on the first panel
    bright_mag = cfg.bright_limits[cfg.bright_names.index(bright_key)]
    axs[0, 0].text(
        0.05, 0.88,
        rf"$M_{{\rm UV,0}} = {bright_mag},\quad z={redshift_label}$",
        transform=axs[0, 0].transAxes, fontsize=18,
    )

    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    return fig
