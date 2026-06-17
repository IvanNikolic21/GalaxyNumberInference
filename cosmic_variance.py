"""
cosmic_variance.py
-------------------
Cosmic (field-to-field) variance of galaxy number counts in NIRCam-sized
pointings, built from a single 21cmFAST coeval halo-catalog box.

We only have one snapshot (z=10.5), so a true multi-snapshot lightcone isn't
possible. Instead we treat the box as representative of an assumed redshift
range, slicing a line-of-sight (LOS) window out of it whose comoving depth
matches the requested z-range. If the requested depth exceeds the box length,
it is truncated to one box length (see `effective_los_depth`) rather than
tiling repeated structure.

Within that LOS slice, the box is tiled transversally into non-overlapping
NIRCam-sized footprints (2.2 x 2.2 arcmin by default), once per available MUV
realization. Each (realization, footprint) combination is one independent
"forward-modeled pointing". Groups of `group_size` pointings (default 28,
matching a planned mosaic survey) are then bootstrap-sampled to estimate the
mean and variance of galaxy counts brighter than a set of MUV thresholds.

Pipeline
--------
    halo_coords, MUV realizations  →  pointing pool (counts per footprint)
        →  bootstrap groups of 28  →  mean & variance per group

Usage
-----
    from galaxy_neighbors import RedshiftConfig, load_halo_catalog
    from cosmic_variance import (
        PointingConfig, comoving_depth_mpc, footprint_side_mpc,
        effective_los_depth, build_pointing_pool, bootstrap_group_stats,
        summarize_group_stats,
    )

    cfg = PointingConfig()
    halo_coords, _ = load_halo_catalog(z_cfg.halo_catalog_path)

    depth_request = comoving_depth_mpc(9.2, 10.9)
    depth, truncated = effective_los_depth(depth_request, cfg.box_len_mpc)
    side = footprint_side_mpc(z_cfg.redshift, cfg.fov_arcmin)

    pool = build_pointing_pool(halo_coords, z_cfg.muv_fiducial_path, 100, cfg, depth, side)
    means, varis = bootstrap_group_stats(pool, cfg)
    print(summarize_group_stats(means, varis, cfg.thresholds))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

from galaxy_neighbors import load_muv_catalog

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PointingConfig:
    """Tunable parameters for the cosmic-variance pointing calculation.

    Parameters
    ----------
    box_len_mpc : float
        Comoving side length of the simulation box. Default: 512.0
        (matches `_BOX_LEN_MPC` in run_ks.py).
    fov_arcmin : float
        Side length of the (square) NIRCam footprint on the sky. Default: 2.2.
    thresholds : list of float
        MUV thresholds; a galaxy counts if M_UV < threshold.
    group_size : int
        Number of pointings per bootstrap "survey" draw. Default: 28.
    n_trials : int
        Number of bootstrap draws. Default: 2000.
    los_axis : int
        Index (0, 1, or 2) of the halo-coordinate axis treated as the
        line-of-sight direction. Default: 2, matching the anisotropic-z
        convention already used in `galaxy_neighbors.find_neighbors_in_box`.
    seed : int
        RNG seed for the bootstrap draws.
    """

    box_len_mpc: float = 512.0
    fov_arcmin: float = 2.2
    thresholds: List[float] = field(default_factory=lambda: [-19.5, -20.0, -20.5])
    group_size: int = 28
    n_trials: int = 2000
    los_axis: int = 2
    seed: int = 42

    @property
    def transverse_axes(self) -> tuple[int, int]:
        return tuple(ax for ax in (0, 1, 2) if ax != self.los_axis)


# ---------------------------------------------------------------------------
# Cosmology helpers
# ---------------------------------------------------------------------------

def comoving_depth_mpc(z_lo: float, z_hi: float) -> float:
    """Comoving line-of-sight depth [Mpc] spanned by a redshift range."""
    d_lo = cosmo.comoving_distance(z_lo).to(u.Mpc).value
    d_hi = cosmo.comoving_distance(z_hi).to(u.Mpc).value
    return d_hi - d_lo


def footprint_side_mpc(z_eval: float, fov_arcmin: float) -> float:
    """Comoving transverse side [Mpc] of a square footprint of side `fov_arcmin`.

    Evaluated at a single redshift (the box's own snapshot redshift, by
    convention) — same arcmin-to-Mpc conversion already used by
    `AnalysisConfig.search_box_mpc` elsewhere in this codebase.
    """
    return (fov_arcmin * u.arcmin * cosmo.kpc_comoving_per_arcmin(z_eval).to(u.Mpc / u.arcmin)).value


def effective_los_depth(depth_requested: float, box_len_mpc: float) -> tuple[float, bool]:
    """Clip a requested LOS depth to at most one box length.

    Returns
    -------
    depth : float
        min(depth_requested, box_len_mpc).
    truncated : bool
        True if depth_requested exceeded the box length.
    """
    truncated = depth_requested > box_len_mpc
    if truncated:
        log.warning(
            f"Requested LOS depth {depth_requested:.1f} Mpc exceeds box length "
            f"{box_len_mpc:.1f} Mpc — truncating to one box length "
            f"({depth_requested - box_len_mpc:.1f} Mpc / "
            f"{100 * (depth_requested - box_len_mpc) / depth_requested:.1f}% of the "
            "requested depth is not covered)."
        )
    return min(depth_requested, box_len_mpc), truncated


# ---------------------------------------------------------------------------
# Pointing pool construction
# ---------------------------------------------------------------------------

def count_pointings_for_realization(
    halo_coords: np.ndarray,
    muvs: np.ndarray,
    cfg: PointingConfig,
    depth_mpc: float,
    footprint_side_mpc: float,
) -> np.ndarray:
    """Tile one realization into non-overlapping NIRCam-sized pointings.

    Slices the box to the LOS window [0, depth_mpc) along `cfg.los_axis`,
    then bins the transverse coordinates into a grid of footprints and
    counts galaxies brighter than each threshold per footprint.

    Parameters
    ----------
    halo_coords : np.ndarray, shape (N, 3)
    muvs : np.ndarray, shape (N,)
        Must be positionally aligned with `halo_coords` (same halo order).
    cfg : PointingConfig
    depth_mpc : float
        LOS window length, e.g. from `effective_los_depth`.
    footprint_side_mpc : float
        Transverse footprint side, e.g. from `footprint_side_mpc`.

    Returns
    -------
    counts : np.ndarray, shape (n_grid**2, len(cfg.thresholds))
        One row per footprint in this realization.
    """
    los = halo_coords[:, cfg.los_axis]
    in_slice = los < depth_mpc
    tx, ty = cfg.transverse_axes
    coords_t = halo_coords[in_slice][:, [tx, ty]]
    muvs_s = muvs[in_slice]

    n_per_side = int(cfg.box_len_mpc // footprint_side_mpc)
    edges = np.arange(n_per_side + 1) * footprint_side_mpc

    counts = np.empty((n_per_side, n_per_side, len(cfg.thresholds)))
    for k, threshold in enumerate(cfg.thresholds):
        sel = muvs_s < threshold
        hist, _, _ = np.histogram2d(coords_t[sel, 0], coords_t[sel, 1], bins=[edges, edges])
        counts[:, :, k] = hist

    return counts.reshape(-1, len(cfg.thresholds))


def build_pointing_pool(
    halo_coords: np.ndarray,
    muv_catalog_path: str | Path,
    n_realizations: int,
    cfg: PointingConfig,
    depth_mpc: float,
    footprint_side_mpc: float,
) -> np.ndarray:
    """Build the full pool of forward-modeled pointings across realizations.

    Parameters
    ----------
    halo_coords : np.ndarray, shape (N, 3)
        Shared across all realizations (same box, same halo positions).
    muv_catalog_path : str or Path
        HDF5 file with one MUV realization per row (see `load_muv_catalog`).
    n_realizations : int
        Number of realizations to load and stack.
    cfg : PointingConfig
    depth_mpc : float
    footprint_side_mpc : float

    Returns
    -------
    pool : np.ndarray, shape (n_realizations * n_grid**2, len(cfg.thresholds))
    """
    rows = []
    for i in range(n_realizations):
        muvs = load_muv_catalog(muv_catalog_path, index=i)
        rows.append(
            count_pointings_for_realization(halo_coords, muvs, cfg, depth_mpc, footprint_side_mpc)
        )
    return np.concatenate(rows, axis=0)


# ---------------------------------------------------------------------------
# Bootstrap statistics
# ---------------------------------------------------------------------------

def bootstrap_group_stats(
    pool: np.ndarray,
    cfg: PointingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap mean & variance of galaxy counts over groups of pointings.

    Each trial draws `cfg.group_size` distinct pointings (without
    replacement within a trial) from `pool` — mimicking one hypothetical
    28-pointing survey — and computes the sample mean and variance (ddof=1)
    of counts per threshold. Different trials may reuse pointings.

    Parameters
    ----------
    pool : np.ndarray, shape (n_pool, n_thresholds)
    cfg : PointingConfig

    Returns
    -------
    means, varis : np.ndarray, each shape (cfg.n_trials, n_thresholds)
    """
    n_pool, n_thresholds = pool.shape
    if cfg.group_size > n_pool:
        raise ValueError(
            f"group_size ({cfg.group_size}) exceeds pool size ({n_pool}) — "
            "use more realizations or a smaller footprint."
        )

    rng = np.random.default_rng(cfg.seed)
    means = np.empty((cfg.n_trials, n_thresholds))
    varis = np.empty((cfg.n_trials, n_thresholds))

    for t in range(cfg.n_trials):
        idx = rng.choice(n_pool, size=cfg.group_size, replace=False)
        sample = pool[idx]
        means[t] = sample.mean(axis=0)
        varis[t] = sample.var(axis=0, ddof=1)

    return means, varis


def summarize_group_stats(
    means: np.ndarray,
    varis: np.ndarray,
    thresholds: List[float],
) -> str:
    """Format a median + 16th/84th percentile table of the bootstrap results."""
    lines = [
        f"  {'M_UV':>8}  {'<N> med':>10}  {'<N> 16-84':>18}  "
        f"{'Var(N) med':>11}  {'Var(N) 16-84':>20}"
    ]
    lines.append(f"  {'-' * 74}")
    for k, threshold in enumerate(thresholds):
        m = means[:, k]
        v = varis[:, k]
        m_lo, m_med, m_hi = np.percentile(m, [16, 50, 84])
        v_lo, v_med, v_hi = np.percentile(v, [16, 50, 84])
        lines.append(
            f"  {threshold:>8.2f}  {m_med:>10.2f}  [{m_lo:>6.2f}, {m_hi:>6.2f}]      "
            f"{v_med:>11.2f}  [{v_lo:>8.2f}, {v_hi:>8.2f}]"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save / load (cache)
# ---------------------------------------------------------------------------

def save_cosmic_variance(
    path: str | Path,
    results: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> None:
    """Save bootstrap results to a compressed numpy archive (.npz).

    The flat key format is ``model__zrange_label__means`` /
    ``model__zrange_label__varis``.

    Parameters
    ----------
    path : str or Path
    results : dict
        results[model][zrange_label] = (means, varis), each shape
        (n_trials, n_thresholds).
    """
    path = Path(path)
    flat: dict[str, np.ndarray] = {}
    for model, by_zrange in results.items():
        for zrange_label, (means, varis) in by_zrange.items():
            flat[f"{model}__{zrange_label}__means"] = means
            flat[f"{model}__{zrange_label}__varis"] = varis
    np.savez_compressed(path, **flat)
    print(f"Saved cosmic variance results → {path}.npz" if path.suffix != ".npz" else f"Saved cosmic variance results → {path}")


def load_cosmic_variance(path: str | Path) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Load bootstrap results from a .npz cache file."""
    path = Path(path)
    archive = np.load(path)

    results: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for flat_key in archive.files:
        model, zrange_label, kind = flat_key.split("__")
        results.setdefault(model, {}).setdefault(zrange_label, [None, None])
        slot = 0 if kind == "means" else 1
        results[model][zrange_label][slot] = archive[flat_key]

    return {
        model: {zr: tuple(pair) for zr, pair in by_zrange.items()}
        for model, by_zrange in results.items()
    }