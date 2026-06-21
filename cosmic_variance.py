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
    side = footprint_side_mpc(z_cfg.redshift, cfg.fov_area_arcmin2)

    pool = build_pointing_pool(halo_coords, z_cfg.muv_fiducial_path, 100, cfg, depth, side)
    means, varis = bootstrap_group_stats(pool, cfg)
    print(summarize_group_stats(means, varis, cfg.thresholds))

    sigma_cv2_trials = bootstrap_fractional_cosmic_variance(pool, cfg)
    fig = plot_fractional_cosmic_variance(
        {"fiducial": {"z9p2_10p9": (means, varis, *pool_moments(pool), sigma_cv2_trials)}},
        cfg.thresholds, {"z9p2_10p9": (9.2, 10.9)},
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from scipy.stats import nbinom

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
    fov_area_arcmin2 : float
        On-sky area of the (square-footprint approximation of the) NIRCam
        pointing, in arcmin^2. Default: 9.7, the combined area of NIRCam's
        two modules (each ~2.2x2.2 arcmin => ~4.84 arcmin^2; ~9.7 arcmin^2
        total for both modules, as used per-pointing in wide pure-parallel
        surveys like PANORAMIC). Same area-based convention as
        `AnalysisConfig.survey_area_arcmin2` in galaxy_neighbors.py.
    thresholds : list of float
        MUV thresholds; a galaxy counts if M_UV < threshold (cumulative --
        each threshold counts everything brighter than it, so the bins are
        nested/correlated). Matches Weibel et al. 2025's (arXiv:2512.14212)
        main-text cumulative N(<M_UV) convention -- the differential-bin
        figure (panel-titled "-20<M_UV<-19.5" etc.) turned out to be from
        their appendix, not the main result PANORAMIC_SIGMA_CV is drawn from.
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
    fov_area_arcmin2: float = 9.7
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


def footprint_side_mpc(z_eval: float, fov_area_arcmin2: float) -> float:
    """Comoving transverse side [Mpc] of a square footprint of area `fov_area_arcmin2`.

    Evaluated at a single redshift (the box's own snapshot redshift, by
    convention) — same arcmin-to-Mpc conversion, and the same
    area-then-sqrt convention, already used by
    `AnalysisConfig.search_box_mpc` elsewhere in this codebase.
    """
    side_arcmin = np.sqrt(fov_area_arcmin2)
    return (side_arcmin * u.arcmin * cosmo.kpc_comoving_per_arcmin(z_eval).to(u.Mpc / u.arcmin)).value


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


def bootstrap_fractional_cosmic_variance(pool: np.ndarray, cfg: PointingConfig) -> np.ndarray:
    """Bootstrap sigma_CV^2 (see `fractional_cosmic_variance`) over survey-sized groups.

    Uses the same per-trial sampling as `bootstrap_group_stats` (group_size
    distinct pointings, without replacement within a trial), but computes
    <N> and <N^2> from each trial's own group rather than the full pool —
    i.e. the estimate you'd actually get from one real `group_size`-pointing
    survey. The spread of the returned distribution is the realistic
    survey-to-survey uncertainty on sigma_CV^2 itself (suitable as an error
    bar), while `fractional_cosmic_variance(*pool_moments(pool))` gives the
    precise point estimate from the full pool.

    A trial where a threshold's group has zero counts (<N>=0, e.g. a rare,
    bright threshold across only `group_size` pointings) leaves sigma_CV^2
    undefined for that trial (0/0) — those entries are set to NaN rather
    than silently propagating a 0/0 warning. Use `np.nanmedian` /
    `np.nanpercentile` downstream so a handful of empty-group trials don't
    blank out the whole threshold.

    Returns
    -------
    sigma_cv2 : np.ndarray, shape (cfg.n_trials, n_thresholds)
    """
    n_pool, n_thresholds = pool.shape
    if cfg.group_size > n_pool:
        raise ValueError(
            f"group_size ({cfg.group_size}) exceeds pool size ({n_pool}) — "
            "use more realizations or a smaller footprint."
        )

    rng = np.random.default_rng(cfg.seed)
    sigma_cv2 = np.empty((cfg.n_trials, n_thresholds))

    for t in range(cfg.n_trials):
        idx = rng.choice(n_pool, size=cfg.group_size, replace=False)
        sample = pool[idx]
        m = sample.mean(axis=0)
        msq = (sample ** 2).mean(axis=0)
        var_cosmic = np.clip(msq - m ** 2 - m, 0, None)
        m_sq = m ** 2
        out = np.full(n_thresholds, np.nan)
        np.divide(var_cosmic, m_sq, out=out, where=m_sq > 0)
        sigma_cv2[t] = out

    n_nan = np.isnan(sigma_cv2).sum(axis=0)
    for k, frac in enumerate((n_nan / cfg.n_trials)):
        if frac > 0:
            log.warning(
                f"threshold index {k}: {frac:.1%} of bootstrap trials had zero "
                f"counts in the group (NaN sigma_CV^2) — likely too rare a "
                f"threshold for group_size={cfg.group_size}."
            )

    return sigma_cv2


def pool_moments(pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and second raw moments of counts per threshold, over the full pool.

    Uses every independent pointing in `pool` (not a 28-pointing subsample),
    so this is the most precise estimate of <N> and <N^2> for a single
    NIRCam pointing.

    Returns
    -------
    mean, mean_sq : np.ndarray, each shape (n_thresholds,)
        <N> and <N^2>.
    """
    return pool.mean(axis=0), (pool ** 2).mean(axis=0)


def poisson_cosmic_variance(
    mean: np.ndarray,
    mean_sq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose the per-pointing count variance into Poisson + cosmic terms.

        Var_total   = <N^2> - <N>^2
        Var_poisson = <N>                      (shot noise of a Poisson process)
        Var_cosmic  = Var_total - Var_poisson   (excess variance from clustering)

    Parameters
    ----------
    mean, mean_sq : np.ndarray, shape (n_thresholds,)
        Output of `pool_moments`.

    Returns
    -------
    var_total, var_poisson, var_cosmic : np.ndarray, each shape (n_thresholds,)
        var_cosmic is clipped at 0 (sub-Poisson counts shouldn't happen for a
        large enough pool, but small-sample noise can dip slightly negative).
    """
    var_total = mean_sq - mean ** 2
    var_poisson = mean
    var_cosmic = var_total - var_poisson
    if np.any(var_cosmic < 0):
        log.warning("Negative cosmic variance encountered (sub-Poisson counts) — clipping to 0.")
        var_cosmic = np.clip(var_cosmic, 0, None)
    return var_total, var_poisson, var_cosmic


def fractional_cosmic_variance(mean: np.ndarray, mean_sq: np.ndarray) -> np.ndarray:
    """Dimensionless cosmic variance, the standard literature quantity:

        sigma_CV^2 = (sigma_total^2 - sigma_Poisson^2) / mu^2
                   = (<N^2> - <N>^2 - <N>) / <N>^2

    Same numerator as `poisson_cosmic_variance`'s var_cosmic, normalized by
    mu^2 = <N>^2.
    """
    _, _, var_cosmic = poisson_cosmic_variance(mean, mean_sq)
    return var_cosmic / mean ** 2


def summarize_pool_moments(
    mean: np.ndarray,
    mean_sq: np.ndarray,
    thresholds: List[float],
    group_size: Optional[int] = None,
) -> str:
    """Format a <N>, <N^2>, and Poisson/cosmic variance table.

    Includes both the absolute cosmic variance (Var_cv, in counts^2) and the
    dimensionless sigma_CV^2 = Var_cv / <N>^2 (see `fractional_cosmic_variance`).
    If `group_size` is given, also reports the variance of the *mean* over
    a survey of that many independent pointings (Var_total / group_size),
    split into its Poisson and cosmic parts — directly comparable to
    `summarize_group_stats`'s bootstrap variance of the group mean.
    """
    var_total, var_poisson, var_cosmic = poisson_cosmic_variance(mean, mean_sq)
    sigma_cv2 = var_cosmic / mean ** 2

    header = (
        f"  {'M_UV':>8}  {'<N>':>9}  {'<N^2>':>10}  {'Var_tot':>9}  {'Var_pois':>9}  "
        f"{'Var_cv':>9}  {'sigma_CV^2':>11}"
    )
    if group_size is not None:
        header += f"  {f'Var_tot/{group_size}':>12}  {f'Var_cv/{group_size}':>12}"
    lines = [header, f"  {'-' * (len(header) - 2)}"]

    for k, threshold in enumerate(thresholds):
        line = (
            f"  {threshold:>8.2f}  {mean[k]:>9.3f}  {mean_sq[k]:>10.3f}  "
            f"{var_total[k]:>9.3f}  {var_poisson[k]:>9.3f}  {var_cosmic[k]:>9.3f}  "
            f"{sigma_cv2[k]:>11.5f}"
        )
        if group_size is not None:
            line += f"  {var_total[k] / group_size:>12.4f}  {var_cosmic[k] / group_size:>12.4f}"
        lines.append(line)
    return "\n".join(lines)


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
# Gamma / Negative-Binomial MCMC fit (Weibel et al. 2025, Section 2.4.2)
# ---------------------------------------------------------------------------
#
# Per-field counts are modeled as Poisson draws on top of a Gamma-distributed
# field-to-field rate (mean mu, variance mu^2 sigma_CV^2). Marginalizing the
# Poisson over that Gamma rate analytically gives a Negative Binomial
# likelihood for the observed counts, with
#     Var(N) = E[Var(N|rate)] + Var(E[N|rate]) = mu + mu^2 sigma_CV^2,
# i.e. exactly the sigma_CV^2 decomposition used everywhere else in this
# module — but fit via a proper hierarchical likelihood instead of
# method-of-moments on a (possibly small) sample. This avoids the
# sigma_CV=0 floor that `bootstrap_fractional_cosmic_variance` can hit when
# many small-N bootstrap draws land in the sub-Poisson (clipped) regime.

@dataclass
class GammaCVConfig:
    """MCMC settings for the Gamma/Negative-Binomial sigma_CV fit.

    Parameters
    ----------
    mean_prior_width_frac : float
        Weak Gaussian prior on mu, centered on the sample mean, with std
        equal to this fraction of the sample mean. Default: 0.3 (matches
        Weibel et al. 2025).
    log10_sigma_cv_bounds : tuple of float
        Bounds on log10(sigma_CV) for the Jeffreys-equivalent prior
        (p(sigma_CV) ~ 1/sigma_CV is flat in log-space; sampling uniformly
        within these bounds implements that prior with a proper cutoff).
    n_walkers, n_steps, n_burn : int
        emcee.EnsembleSampler settings.
    seed : int
        RNG seed for walker initialization.
    """

    mean_prior_width_frac: float = 0.3
    log10_sigma_cv_bounds: tuple[float, float] = (-3.0, 1.0)
    n_walkers: int = 32
    n_steps: int = 4000
    n_burn: int = 1000
    seed: int = 42


def negbinom_params(mu: float, sigma_cv: float) -> tuple[float, float]:
    """Convert (mu, sigma_CV) to the (n, p) parameterization of scipy.stats.nbinom.

    Matches Var(N) = mu + mu^2 * sigma_CV^2: n = 1/sigma_CV^2, p = n/(n+mu).
    As sigma_CV -> 0, n -> inf and the distribution -> Poisson(mu).
    """
    n = 1.0 / sigma_cv ** 2
    p = n / (n + mu)
    return n, p


def negbinom_loglike(counts: np.ndarray, mu: float, sigma_cv: float) -> float:
    """Negative-binomial log-likelihood of observed per-pointing `counts`."""
    if mu <= 0 or sigma_cv <= 0:
        return -np.inf
    n, p = negbinom_params(mu, sigma_cv)
    return float(np.sum(nbinom.logpmf(counts, n, p)))


def _gamma_cv_log_prior(mu: float, sigma_cv: float, mean_obs: float, cfg: GammaCVConfig) -> float:
    if mu <= 0 or sigma_cv <= 0:
        return -np.inf
    log10_sigma = np.log10(sigma_cv)
    lo, hi = cfg.log10_sigma_cv_bounds
    if not (lo <= log10_sigma <= hi):
        return -np.inf
    # Weak Gaussian prior on mu. The Jeffreys prior p(sigma_CV) ~ 1/sigma_CV
    # is flat in log10(sigma_CV), already implemented by the uniform bounds
    # check above (no extra log-density term needed for a flat prior).
    mean_prior_std = cfg.mean_prior_width_frac * mean_obs
    return -0.5 * ((mu - mean_obs) / mean_prior_std) ** 2


def _gamma_cv_log_posterior(theta: np.ndarray, counts: np.ndarray, mean_obs: float, cfg: GammaCVConfig) -> float:
    mu, sigma_cv = theta
    lp = _gamma_cv_log_prior(mu, sigma_cv, mean_obs, cfg)
    if not np.isfinite(lp):
        return -np.inf
    return lp + negbinom_loglike(counts, mu, sigma_cv)


def fit_sigma_cv_mcmc(counts: np.ndarray, cfg: Optional[GammaCVConfig] = None) -> dict:
    """Fit (mu, sigma_CV) to per-pointing `counts` via Gamma/NB MCMC (emcee).

    Mirrors Weibel et al. 2025 Section 2.4.2: a single MCMC fit to the full
    set of observed counts, rather than a method-of-moments estimate
    bootstrapped over many small subsamples.

    Parameters
    ----------
    counts : np.ndarray, shape (n_fields,)
        Per-pointing galaxy counts for one MUV threshold (e.g. one column of
        a `build_pointing_pool` pool, or a `group_size`-sized subsample of
        it to emulate a realistic survey).
    cfg : GammaCVConfig, optional

    Returns
    -------
    result : dict
        Keys: "mu_samples", "sigma_cv_samples" (flattened post-burn-in
        chains), and "mu_median"/"mu_lo"/"mu_hi",
        "sigma_cv_median"/"sigma_cv_lo"/"sigma_cv_hi" (16th/50th/84th
        percentiles).
    """
    import emcee

    if cfg is None:
        cfg = GammaCVConfig()

    counts = np.asarray(counts)
    mean_obs = float(np.mean(counts))
    if mean_obs <= 0:
        raise ValueError(
            f"fit_sigma_cv_mcmc: zero detections across all {len(counts)} pointings "
            "in this sample -- this threshold is too rare for the sample size used; "
            "increase the sample size or drop this threshold."
        )
    rng = np.random.default_rng(cfg.seed)

    # Initialize walkers near a method-of-moments estimate, jittered.
    var_obs = float(np.var(counts, ddof=1))
    mom_sigma_cv = np.sqrt(max((var_obs - mean_obs) / mean_obs ** 2, 1e-3))

    p0 = np.empty((cfg.n_walkers, 2))
    p0[:, 0] = mean_obs * (1 + 0.05 * rng.standard_normal(cfg.n_walkers))
    p0[:, 1] = mom_sigma_cv * (1 + 0.2 * rng.standard_normal(cfg.n_walkers))
    lo_bound, hi_bound = 10 ** cfg.log10_sigma_cv_bounds[0], 10 ** cfg.log10_sigma_cv_bounds[1]
    p0[:, 1] = np.clip(p0[:, 1], lo_bound, hi_bound)
    p0[:, 0] = np.clip(p0[:, 0], 1e-6, None)

    sampler = emcee.EnsembleSampler(cfg.n_walkers, 2, _gamma_cv_log_posterior, args=(counts, mean_obs, cfg))
    sampler.run_mcmc(p0, cfg.n_steps, progress=False)

    chain = sampler.get_chain(discard=cfg.n_burn, flat=True)
    mu_samples, sigma_cv_samples = chain[:, 0], chain[:, 1]

    mu_lo, mu_med, mu_hi = np.percentile(mu_samples, [16, 50, 84])
    s_lo, s_med, s_hi = np.percentile(sigma_cv_samples, [16, 50, 84])

    return {
        "mu_samples": mu_samples,
        "sigma_cv_samples": sigma_cv_samples,
        "mu_median": mu_med, "mu_lo": mu_lo, "mu_hi": mu_hi,
        "sigma_cv_median": s_med, "sigma_cv_lo": s_lo, "sigma_cv_hi": s_hi,
    }


def summarize_gamma_fits(fits: dict[float, dict], thresholds: List[float]) -> str:
    """Format a median + 16th/84th percentile table of Gamma/NB MCMC fits.

    Parameters
    ----------
    fits : dict
        threshold -> result dict from `fit_sigma_cv_mcmc`.
    thresholds : list of float
    """
    lines = [f"  {'M_UV':>8}  {'mu med':>10}  {'mu 16-84':>18}  {'sigma_CV med':>13}  {'sigma_CV 16-84':>20}"]
    lines.append(f"  {'-' * 78}")
    for threshold in thresholds:
        fit = fits[threshold]
        lines.append(
            f"  {threshold:>8.2f}  {fit['mu_median']:>10.3f}  "
            f"[{fit['mu_lo']:>6.3f}, {fit['mu_hi']:>6.3f}]      "
            f"{fit['sigma_cv_median']:>13.3f}  "
            f"[{fit['sigma_cv_lo']:>8.3f}, {fit['sigma_cv_hi']:>8.3f}]"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save / load (cache)
# ---------------------------------------------------------------------------

_CACHE_SLOTS = ("means", "varis", "mean", "mean_sq", "sigma_cv2_trials")


def save_cosmic_variance(
    path: str | Path,
    results: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
) -> None:
    """Save bootstrap + moment results to a compressed numpy archive (.npz).

    The flat key format is ``model__zrange_label__<slot>`` for slot in
    ("means", "varis", "mean", "mean_sq", "sigma_cv2_trials").

    Parameters
    ----------
    path : str or Path
    results : dict
        results[model][zrange_label] = (means, varis, mean, mean_sq, sigma_cv2_trials):
          means, varis : shape (n_trials, n_thresholds) — bootstrap group stats.
          mean, mean_sq : shape (n_thresholds,) — <N>, <N^2> over the full pool.
          sigma_cv2_trials : shape (n_trials, n_thresholds) — bootstrap sigma_CV^2.
    """
    path = Path(path)
    flat: dict[str, np.ndarray] = {}
    for model, by_zrange in results.items():
        for zrange_label, values in by_zrange.items():
            for slot, arr in zip(_CACHE_SLOTS, values):
                flat[f"{model}__{zrange_label}__{slot}"] = arr
    np.savez_compressed(path, **flat)
    print(f"Saved cosmic variance results → {path}.npz" if path.suffix != ".npz" else f"Saved cosmic variance results → {path}")


def load_cosmic_variance(
    path: str | Path,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """Load bootstrap + moment results from a .npz cache file."""
    path = Path(path)
    archive = np.load(path)

    results: dict[str, dict[str, list]] = {}
    for flat_key in archive.files:
        model, zrange_label, slot = flat_key.split("__")
        results.setdefault(model, {}).setdefault(zrange_label, [None] * len(_CACHE_SLOTS))
        results[model][zrange_label][_CACHE_SLOTS.index(slot)] = archive[flat_key]

    return {
        model: {zr: tuple(values) for zr, values in by_zrange.items()}
        for model, by_zrange in results.items()
    }


_GAMMA_SUMMARY_SLOTS = (
    "mu_median", "mu_lo", "mu_hi",
    "sigma_cv_median", "sigma_cv_lo", "sigma_cv_hi",
)


def save_gamma_summary(
    path: str | Path,
    gamma_fits_by_model: dict[str, dict[str, dict[float, dict]]],
) -> None:
    """Save Gamma/NB fit summary statistics to a compressed numpy archive.

    Only the 6 percentile scalars per (model, z-range, threshold) are saved
    (not the MCMC chains, which `fit_sigma_cv_mcmc` also returns) — enough
    to rebuild a `gamma_fit_to_reference` entry without recomputation, e.g.
    in `plot_panoramic_comparison.py`. Flat key format:
    ``model__zrange_label__threshold__slot``.

    Parameters
    ----------
    path : str or Path
    gamma_fits_by_model : dict
        gamma_fits_by_model[model][zrange_label][threshold] = result dict
        from `fit_sigma_cv_mcmc` (the *plotted* fit for that model).
    """
    path = Path(path)
    flat: dict[str, np.ndarray] = {}
    for model, by_zrange in gamma_fits_by_model.items():
        for zrange_label, by_threshold in by_zrange.items():
            for threshold, fit in by_threshold.items():
                for slot in _GAMMA_SUMMARY_SLOTS:
                    flat[f"{model}__{zrange_label}__{threshold:g}__{slot}"] = np.asarray(fit[slot])
    np.savez_compressed(path, **flat)
    print(f"Saved Gamma/NB fit summary → {path}.npz" if path.suffix != ".npz" else f"Saved Gamma/NB fit summary → {path}")


def load_gamma_summary(
    path: str | Path,
) -> dict[str, dict[str, dict[float, dict]]]:
    """Load a Gamma/NB fit summary saved by `save_gamma_summary`."""
    path = Path(path)
    archive = np.load(path)
    out: dict[str, dict[str, dict[float, dict]]] = {}
    for flat_key in archive.files:
        model, zrange_label, threshold_str, slot = flat_key.split("__")
        fit = out.setdefault(model, {}).setdefault(zrange_label, {}).setdefault(float(threshold_str), {})
        fit[slot] = float(archive[flat_key])
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Same color convention as D1sConfig / plotter.py: fiducial=orange, stochastic=blue.
# Within each model, the full-depth-coverage range gets the dark/primary shade
# (matching D1sConfig.color_fid/color_stoc) and the truncated wide range gets a
# lighter tint from the same family (matching the light-to-dark ramps used in
# plot_d1_subpanels.py / plotter.CDF_all_gal_paper_version_split_by_N).
_ZRANGE_SHADE = {
    "fiducial":   {"z9p2_10p9": "#a63603", "z8p6_11p3": "#fd8d3c"},
    "stochastic": {"z9p2_10p9": "#08519c", "z8p6_11p3": "#6baed6"},
    "prejwst":    {"z9p2_10p9": "#238b45", "z8p6_11p3": "#74c476"},
    "fiducial_2cmpc":   {"z9p2_10p9": "#7f2704", "z8p6_11p3": "#fdae6b"},
    "stochastic_2cmpc": {"z9p2_10p9": "#08306b", "z8p6_11p3": "#9ecae1"},
}
_MODEL_LABEL = {
    "fiducial": "intrinsically bright",
    "stochastic": "increased stochasticity",
    "prejwst": "pre-JWST (UM-matched)",
    "fiducial_2cmpc": "intrinsically bright (2 cMpc box)",
    "stochastic_2cmpc": "increased stochasticity (2 cMpc box)",
}
_MODEL_MARKER = {
    "fiducial": "o", "stochastic": "o", "prejwst": "^",
    "fiducial_2cmpc": "s", "stochastic_2cmpc": "s",
}
_ZRANGE_X_OFFSET = {"z9p2_10p9": -0.04, "z8p6_11p3": 0.04}


def gamma_fit_to_reference(
    model: str,
    zlo: float,
    zhi: float,
    zrange_label: str,
    fits: dict[float, dict],
    thresholds: List[float],
) -> tuple[str, dict]:
    """Package a per-threshold Gamma/NB MCMC fit (`fit_sigma_cv_mcmc` results,
    keyed by threshold) into the `reference` overlay format expected by
    `plot_fractional_cosmic_variance`, reusing this module's color/marker/
    offset conventions for `model` so it blends in with the bootstrap-based
    series rather than looking like an unrelated literature point.
    """
    values = [fits[t]["sigma_cv_median"] for t in thresholds]
    lo = [fits[t]["sigma_cv_lo"] for t in thresholds]
    hi = [fits[t]["sigma_cv_hi"] for t in thresholds]
    label = f"{_MODEL_LABEL.get(model, model)} (Gamma/NB), ${zlo}<z<{zhi}$"
    entry = {
        "values": values, "lo": lo, "hi": hi,
        "color": _ZRANGE_SHADE[model][zrange_label],
        "marker": _MODEL_MARKER.get(model, "o"),
        "offset": _ZRANGE_X_OFFSET.get(zrange_label, 0.0),
    }
    return label, entry

# Weibel et al. 2025 (PANORAMIC z~10 cosmic variance paper, arXiv:2512.14212),
# sigma_CV = sqrt(sigma_CV^2 from their Eq. 4) for a 9.7 arcmin^2 NIRCam
# pointing, 9.2<z<10.9, at M_UV<-19.5,-20,-20.5. Confirmed linear (not
# squared) by their own bias relation b_g,CV = sigma_CV / sigma_DM (Eq. 2,
# a ratio of rms quantities) and by their text: "sigma_CV = 1.71 ... fractional
# uncertainty as high as ~110-240%" only works out (1.71-0.59=1.12 to
# 1.71+0.72=2.43 -> 112%-243%) if sigma_CV itself is the fractional rms,
# i.e. not squared.
PANORAMIC_SIGMA_CV = {
    "PANORAMIC data (MCMC)": {
        "values": [0.96, 1.46, 1.71],
        "lo": [0.96 - 0.18, 1.46 - 0.44, 1.71 - 0.59],
        "hi": [0.96 + 0.20, 1.46 + 0.54, 1.71 + 0.72],
        "color": "black", "marker": "*", "offset": -0.10,
    },
    "PANORAMIC data (bootstrap)": {
        "values": [0.96, 1.76, 2.37],
        "lo": [0.96 - 0.26, 1.76 - 0.43, 2.37 - 0.44],
        "hi": [0.96 + 0.28, 1.76 + 0.50, 2.37 + 0.64],
        "color": "dimgray", "marker": "x", "offset": 0.10,
    },
    # Read off Figure 1's green "UniverseMachine (fiducial)" points by pixel
    # position (axis-tick-calibrated), since the paper doesn't give exact
    # digits for these in the text — only "agrees well with our measurements".
    # Approximate; re-extract from the source figure data if exact values matter.
    "UniverseMachine (fiducial)": {
        "values": [0.92, 1.19, 2.10],
        "lo": [0.86, 1.06, 1.68],
        "hi": [0.97, 1.31, 2.51],
        "color": "yellowgreen", "marker": "o", "offset": 0.0,
    },
}


def plot_fractional_cosmic_variance(
    results: dict[str, dict[str, tuple]],
    thresholds: List[float],
    z_ranges: dict[str, tuple[float, float]],
    figsize: tuple[float, float] = (7, 5),
    reference: Optional[dict[str, dict]] = None,
) -> plt.Figure:
    """Errorbar plot of sigma_CV vs M_UV threshold, both models, both z-ranges.

    Plots sigma_CV = sqrt(sigma_CV^2), matching the literature convention
    (e.g. Robertson 2010, Somerville et al. 2004, Weibel et al. 2025's
    b_g,CV = sigma_CV/sigma_DM) where sigma_CV itself is the fractional rms
    — not the squared quantity computed internally by
    `bootstrap_fractional_cosmic_variance`/`fractional_cosmic_variance`.
    Percentiles commute with the (monotonic) sqrt, so taking the sqrt of the
    median/16th/84th-percentile of sigma_CV^2 is exact, not an approximation.

    Central value per (model, z-range, threshold) is the median of the
    bootstrap sigma_CV^2 distribution (`bootstrap_fractional_cosmic_variance`);
    error bars are its 16th/84th percentiles — i.e. the realistic
    survey-to-survey scatter on the cosmic-variance estimate itself for a
    `group_size`-pointing survey.

    Parameters
    ----------
    results : dict
        results[model][zrange_label] = (means, varis, mean, mean_sq, sigma_cv2_trials),
        as produced by run_cosmic_variance.py / save_cosmic_variance.
    thresholds : list of float
        MUV thresholds, used for the x-axis.
    z_ranges : dict
        zrange_label -> (z_lo, z_hi), used for the legend labels.
    figsize : tuple of float
        Figure size in inches, passed to `plt.subplots`.
    reference : dict, optional
        Literature comparison points (already in sigma_CV, not squared) to
        overlay, keyed by legend label. Each value is a dict with keys
        "values", "lo", "hi" (arrays matching `thresholds`), "color",
        "marker", and "offset" (x-axis nudge to avoid overlapping markers).
        See `PANORAMIC_SIGMA_CV` for a ready-made example (Weibel et al.
        2025, arXiv:2512.14212).
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.array(thresholds)

    for model, by_zrange in results.items():
        for zrange_label, values in by_zrange.items():
            _, _, _, _, sigma_cv2_trials = values
            med2 = np.nanmedian(sigma_cv2_trials, axis=0)
            lo2, hi2 = np.nanpercentile(sigma_cv2_trials, [16, 84], axis=0)
            med, lo, hi = np.sqrt(med2), np.sqrt(lo2), np.sqrt(hi2)
            zlo, zhi = z_ranges[zrange_label]
            color = _ZRANGE_SHADE[model][zrange_label]
            offset = _ZRANGE_X_OFFSET.get(zrange_label, 0.0)
            ax.errorbar(
                x + offset, med,
                yerr=[med - lo, hi - med],
                fmt=_MODEL_MARKER.get(model, 'o'), color=color, capsize=3, lw=2, markersize=6,
                label=f"{_MODEL_LABEL.get(model, model)}, ${zlo}<z<{zhi}$",
            )

    if reference is not None:
        for label, ref in reference.items():
            values = np.array(ref["values"])
            lo = np.array(ref["lo"])
            hi = np.array(ref["hi"])
            offset = ref.get("offset", 0.0)
            ax.errorbar(
                x + offset, values,
                yerr=[values - lo, hi - values],
                fmt=ref.get("marker", "s"), color=ref.get("color", "black"),
                capsize=3, lw=2, markersize=8, label=label,
            )

    ax.set_xlabel(r"$M_{\rm UV}$ threshold", fontsize=14)
    ax.set_ylabel(r"$\sigma_{\rm CV}$", fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig