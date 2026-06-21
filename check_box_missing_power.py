#!/usr/bin/env python
"""
check_box_missing_power.py
---------------------------
Quantify how much of the *true* cosmic variance (linear-theory dark-matter
density-field variance, smoothed on the survey-volume scale) a single
periodic box of side `box_len_mpc` can ever represent, independent of
resolution, realization count, or galaxy-formation model.

Motivation
----------
cosmic_variance.py measures sigma_CV empirically from one 512 cMpc periodic
box. A periodic box's Fourier representation only contains discrete modes
k_n = (2*pi/L) * n for integer n != 0 -- modes longer than the box (k < 2*pi/L)
are *structurally* absent, not just unresolved. If the survey volumes we
care about (LOS depths ~368-596 Mpc, comparable to the box itself) draw a
meaningful fraction of their true variance from those missing large-scale
modes, then *every* model built on this box (fiducial, stochastic, pre-JWST)
would underpredict sigma_CV relative to literature values derived from larger
volumes/analytic linear theory (e.g. Weibel et al. 2025's sigma_DM, which is
computed from the linear power spectrum, not measured in a finite box).

This script computes, for each z-range already used in run_cosmic_variance.py:
    sigma2_true : continuum linear-theory variance (all k, in particular k -> 0)
    sigma2_box  : discrete-lattice variance achievable from a box of side
                  box_len_mpc (excludes the k < 2*pi/box_len_mpc modes that
                  don't exist in a periodic box this size)
for an effective top-hat sphere matching the *requested* (not depth-truncated)
survey volume -- i.e. comparing against the survey volume PANORAMIC/
UniverseMachine actually refer to, not the volume our box can literally
simulate.

Linear P(k): BBKS/Sugiyama-corrected CDM transfer function (Bardeen, Bond,
Kaiser & Szalay 1986; Sugiyama 1995 Gamma correction), normalized to
sigma8 (Planck 2018: sigma8=0.8111, ns=0.9649), growth factor via the
Carroll, Press & Turner (1992) fitting formula. This is a diagnostic-level
P(k) (no BAO wiggles, no exact Boltzmann solver) -- accurate to a few percent
in shape, which is more than enough to assess an order-unity missing-power
effect; swap in camb/colossus if the result is borderline and needs refining.
"""

import numpy as np
from scipy.integrate import quad
from astropy.cosmology import Planck18 as cosmo

from cosmic_variance import PointingConfig, comoving_depth_mpc, footprint_side_mpc

BOX_LEN_MPC = 512.0
BOX_REDSHIFT = 10.5
Z_RANGES = {
    "z9p2_10p9": (9.2, 10.9),
    "z8p6_11p3": (8.6, 11.3),
}
FOV_AREA_ARCMIN2 = 9.7

# Planck 2018 (TT,TE,EE+lowE+lensing) primordial/structure parameters --
# not carried by astropy's FLRW objects, which only hold background params.
SIGMA8 = 0.8111
NS = 0.9649

OM0 = cosmo.Om0
OB0 = cosmo.Ob0
H0_H = cosmo.h


# ---------------------------------------------------------------------------
# Linear growth factor (Carroll, Press & Turner 1992 fitting formula)
# ---------------------------------------------------------------------------

def _growth_g(z: float) -> float:
    """CPT92 g(z) = D(z)*(1+z), for flat Om0+OL0=1 cosmology."""
    a3 = (1 + z) ** 3
    e2 = OM0 * a3 + (1 - OM0)
    om_z = OM0 * a3 / e2
    ol_z = (1 - OM0) / e2
    return 2.5 * om_z / (om_z ** (4 / 7) + (1 + om_z / 2) * (1 + ol_z / 70))


def growth_factor(z: float) -> float:
    """D(z), normalized to D(0)=1."""
    return (_growth_g(z) / (1 + z)) / _growth_g(0.0)


# ---------------------------------------------------------------------------
# BBKS / Sugiyama linear transfer function and P(k)
# ---------------------------------------------------------------------------

def _shape_gamma() -> float:
    """Sugiyama (1995) baryon-corrected shape parameter Gamma = Om0*h*exp(...)."""
    return OM0 * H0_H * np.exp(-OB0 * (1 + np.sqrt(2 * H0_H) / OM0))


_GAMMA = _shape_gamma()


def transfer_function(k_mpc: np.ndarray) -> np.ndarray:
    """BBKS CDM transfer function. k_mpc: physical wavenumber [1/Mpc]."""
    k_mpc = np.asarray(k_mpc, dtype=float)
    q = (k_mpc / H0_H) / _GAMMA  # k converted to h/Mpc units, then / Gamma
    q = np.where(q <= 0, 1e-12, q)
    return (np.log(1 + 2.34 * q) / (2.34 * q)) * (
        1 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4
    ) ** -0.25


def _unnormalized_pk(k_mpc: np.ndarray) -> np.ndarray:
    return k_mpc ** NS * transfer_function(k_mpc) ** 2


def _tophat_window(x: np.ndarray) -> np.ndarray:
    """Spherical top-hat window in k-space, W(x) with x = k*R."""
    x = np.asarray(x, dtype=float)
    x = np.where(x == 0, 1e-12, x)
    return 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3


def _sigma2_tophat_unnormalized(R_mpc: float, k_max: float = 50.0) -> float:
    integrand = lambda k: k ** 2 * _unnormalized_pk(k) * _tophat_window(k * R_mpc) ** 2
    val, _ = quad(integrand, 1e-6, k_max, limit=400)
    return val / (2 * np.pi ** 2)


# Normalize P(k) = A * k^ns * T(k)^2 so that sigma(R=8/h Mpc, z=0) = sigma8.
_R8_MPC = 8.0 / H0_H
_SIGMA2_8_UNNORM = _sigma2_tophat_unnormalized(_R8_MPC)
_PK_NORM = SIGMA8 ** 2 / _SIGMA2_8_UNNORM


def power_spectrum(k_mpc: np.ndarray, z: float) -> np.ndarray:
    """Linear matter power spectrum P(k, z) [Mpc^3], normalized to sigma8 at z=0."""
    d = growth_factor(z)
    return _PK_NORM * _unnormalized_pk(k_mpc) * d ** 2


# ---------------------------------------------------------------------------
# True (continuum) vs box-achievable (discrete lattice) variance
#
# The survey "window" is the actual anisotropic pencil-beam (footprint_side x
# footprint_side x depth), NOT a spherical stand-in of equal volume -- that
# matters a lot here: the tiny transverse footprint gives a *broad* window in
# (kx, ky) (not very sensitive to missing low-k power), while the long LOS
# depth gives a *narrow* window in kz, peaked right around k ~ 2*pi/depth --
# i.e. exactly the scale where the box's own k_min = 2*pi/box_len_mpc bites.
# An equal-volume sphere averages over all directions and mostly misses this.
# ---------------------------------------------------------------------------

def _sinc(x: np.ndarray) -> np.ndarray:
    """sin(x)/x, with sinc(0) = 1 (not numpy's pi-scaled convention)."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = np.sin(x[nz]) / x[nz]
    return out


def _composite_gl_grid(k_max: float, period: float, points_per_period: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre nodes/weights over [-k_max, k_max], using
    panels narrow enough (a quarter of `period`) to resolve an oscillation
    with that period -- a single fixed-order GL rule across the whole range
    silently aliases once the integrand oscillates more than ~order(n_gl)
    times across the domain (exactly what sinc^2(kz*Lz/2) does here, with
    Lz ~ 400-600 Mpc giving hundreds of periods within a few Mpc^-1)."""
    panel_width = period / 4
    n_panels = max(1, int(np.ceil(k_max / panel_width)))
    edges = np.linspace(0.0, k_max, n_panels + 1)
    nodes_1, weights_1 = np.polynomial.legendre.leggauss(8)
    half = edges[:-1, None] + (edges[1:, None] - edges[:-1, None]) * (nodes_1[None, :] + 1) / 2
    halfw = (edges[1:, None] - edges[:-1, None]) / 2 * weights_1[None, :]
    pos_nodes, pos_weights = half.ravel(), halfw.ravel()
    # Mirror to negative side (integrand here is always even in each axis individually).
    nodes = np.concatenate([-pos_nodes[::-1], pos_nodes])
    weights = np.concatenate([pos_weights[::-1], pos_weights])
    return nodes, weights


def sigma2_true_aniso(Lx: float, Ly: float, Lz: float, z: float,
                       k_max: float = 4.0, n_perp: int = 120) -> float:
    """Continuum pencil-beam-window variance via 3D quadrature (no missing
    power -- includes the full k -> 0 limit).

    The transverse (kx, ky) window oscillates slowly (period ~2*pi/footprint_side,
    a handful of periods across k_max) so a fixed-order Gauss-Legendre grid
    resolves it fine. The line-of-sight (kz) window oscillates with period
    ~2*pi/depth -- hundreds of periods across k_max, since depth ~ 400-600 Mpc
    -- so it needs the composite (panelled) quadrature from `_composite_gl_grid`.
    Looped over kz panels (vectorized over the kx-ky plane per panel point) to
    keep peak memory bounded.
    """
    nodes_perp, weights_perp = np.polynomial.legendre.leggauss(n_perp)
    k_perp = nodes_perp * k_max
    w_perp = weights_perp * k_max
    KX, KY = np.meshgrid(k_perp, k_perp, indexing="ij")
    WXY = np.outer(w_perp, w_perp)
    wx2 = _sinc(KX * Lx / 2) ** 2
    wy2 = _sinc(KY * Ly / 2) ** 2

    kz_nodes, kz_weights = _composite_gl_grid(k_max, period=2 * np.pi / Lz)
    total = 0.0
    for kz_val, kz_w in zip(kz_nodes, kz_weights):
        kmag = np.sqrt(KX ** 2 + KY ** 2 + kz_val ** 2)
        wz2 = _sinc(kz_val * Lz / 2) ** 2
        total += kz_w * np.sum(power_spectrum(kmag, z) * wx2 * wy2 * wz2 * WXY)
    return total / (2 * np.pi) ** 3


def sigma2_missing_aniso(Lx: float, Ly: float, Lz: float, z: float,
                          k_min: float, n_gl: int = 200) -> float:
    """Continuum variance contributed by modes with |k| < k_min -- the part
    of the true variance that is *structurally* inaccessible to any single
    periodic box of side `box_len_mpc = 2*pi/k_min`, regardless of
    resolution or how many realizations are drawn from it.

    (Deliberately NOT computed as a discrete sum over the box's own k-lattice:
    when the survey window has a feature (e.g. the LOS sinc^2, width ~
    2*pi/depth) narrower than the lattice spacing 2*pi/box_len_mpc, sampling
    it *on* that coarse lattice aliases badly -- it can way overcount a sharp
    peak landing near a single grid point. Splitting the smooth continuum
    integral at k_min sidesteps that entirely.)
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_gl)
    k = nodes * k_min
    w = weights * k_min
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    WX, WY, WZ = np.meshgrid(w, w, w, indexing="ij")
    kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    inside = kmag < k_min
    integrand = np.zeros_like(kmag)
    integrand[inside] = (
        power_spectrum(kmag[inside], z)
        * _sinc(KX[inside] * Lx / 2) ** 2 * _sinc(KY[inside] * Ly / 2) ** 2
        * _sinc(KZ[inside] * Lz / 2) ** 2
    )
    total = np.sum(integrand * WX * WY * WZ)
    return total / (2 * np.pi) ** 3


def main():
    cfg = PointingConfig(box_len_mpc=BOX_LEN_MPC, fov_area_arcmin2=FOV_AREA_ARCMIN2)
    footprint_side = footprint_side_mpc(BOX_REDSHIFT, cfg.fov_area_arcmin2)

    print(f"Cosmology: Om0={OM0:.4f}  Ob0={OB0:.4f}  h={H0_H:.4f}  "
          f"sigma8={SIGMA8}  ns={NS}  (Planck 2018)")
    print(f"Box side: {BOX_LEN_MPC:.1f} Mpc  ->  k_min = {2 * np.pi / BOX_LEN_MPC:.5f} Mpc^-1")
    print(f"Footprint side at z={BOX_REDSHIFT}: {footprint_side:.3f} Mpc\n")

    header = f"  {'z-range':<12} {'depth[Mpc]':>11} {'sigma_true':>11} {'sigma_box':>10} {'missing %':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    k_min = 2 * np.pi / BOX_LEN_MPC
    for label, (zlo, zhi) in Z_RANGES.items():
        depth = comoving_depth_mpc(zlo, zhi)  # requested (not box-truncated) depth -- the
        # true survey volume PANORAMIC/UniverseMachine refer to, regardless of whether our
        # box can simulate all of it.
        s2_true = sigma2_true_aniso(footprint_side, footprint_side, depth, BOX_REDSHIFT, k_max=6.0, n_perp=200)
        s2_missing = sigma2_missing_aniso(footprint_side, footprint_side, depth, BOX_REDSHIFT, k_min)
        s2_box = s2_true - s2_missing
        missing_pct = 100 * s2_missing / s2_true
        print(
            f"  {label:<12} {depth:>11.1f} "
            f"{np.sqrt(s2_true):>11.5f} {np.sqrt(max(s2_box, 0.0)):>10.5f} {missing_pct:>9.1f}%"
        )


if __name__ == "__main__":
    main()