#!/usr/bin/env python
"""
plot_posterior_gain.py
-----------------------
Summarize the inference "gain" going UVLF-only -> d1-only -> full-MLP (all
neighbors), using two physically interpretable derived quantities instead of
the raw (Muv_add, sigmaUV_a, sigmaUV_b) inference parameters:

  1. sigma_UV evaluated at M_UV = -20        (SIGMA_UV_TARGET_MUV)
  2. M_UV predicted at log(Mh) = 11          (MUV_AT_LOGMH_TARGET)

Both use the same z=10 Muv-Mh table (via uvlf.Mason15) that infer_nre.py's
UVLF likelihood is built on, so all three posteriors are compared on
self-consistent footing.

Produces:
  - posterior_gain_corner.pdf   overlaid raw-parameter corner plot, 3 colors
  - posterior_gain_derived.pdf  2-panel histogram of the two derived quantities
  - printed medians / 68% CI / contraction ratios for both quantities

Run on the cluster (needs the actual posterior_samples_*.npy files and corner/hmf).

Usage
-----
    python plot_posterior_gain.py
"""

import numpy as np
import matplotlib.pyplot as plt
import corner

from uvlf import Mason15

# ---------------------------------------------------------------------------
# Inputs — edit these paths if yours differ
# ---------------------------------------------------------------------------
SAMPLES = {
    "UVLF-only":     "/groups/astro/ivannik/projects/Neighbors/UVLF_only/posterior_samples_N0_uvlfonly.npy",
    "d1-only":       "/groups/astro/ivannik/projects/Neighbors/nre_model_d1_capped_only_ang/posterior_samples_d1_N1.npy",
    "full-MLP":      "/groups/astro/ivannik/projects/Neighbors/nre_model_capped_only_ang/posterior_samples_N1.npy",
    "full-MLP+UVLF": "/groups/astro/ivannik/projects/Neighbors/nre_model_capped_only_ang/posterior_samples_N1_uvlf.npy",
}
OUTPUT_DIR = "/groups/astro/ivannik/projects/Neighbors/UVLF_only"

SIGMA_UV_TARGET_MUV  = -20.0   # evaluate sigma_UV(M_h) at the halo mass where median M_UV = this
MUV_AT_LOGMH_TARGET  = 11.0    # evaluate the (shifted) median M_UV-Mh relation at this log(Mh)

# palette — dataviz skill categorical slots 1-3 (validated all-pairs safe, both modes)
# for the combined full-MLP+UVLF case, black is used deliberately rather than a 4th
# categorical hue: it's achromatic, so it can't collide/CVD-confuse with the other
# three, and reads as "the combined/reference result" rather than another series.
COLOR_UVLF = "#1baf7a"   # aqua  -- weakest baseline
COLOR_D1   = "#eb6834"   # orange
COLOR_MLP  = "#2a78d6"   # blue  -- richest single-probe info
COLOR_MLP_UVLF = "#0b0b0b"   # black -- full-MLP + UVLF combined
COLORS = {
    "UVLF-only": COLOR_UVLF,
    "d1-only": COLOR_D1,
    "full-MLP": COLOR_MLP,
    "full-MLP+UVLF": COLOR_MLP_UVLF,
}

PARAM_LABELS = [r"$M_{\rm UV,add}$", r"$\sigma_{\rm UV,a}$", r"$\sigma_{\rm UV,b}$"]
PARAM_RANGE  = [(-1.5, 2.0), (-1.0, 1.5), (0.0, 3.0)]

# ---------------------------------------------------------------------------
# Load posteriors
# ---------------------------------------------------------------------------
posteriors = {name: np.load(path) for name, path in SAMPLES.items()}
for name, s in posteriors.items():
    print(f"{name}: {s.shape[0]} samples")

# ---------------------------------------------------------------------------
# Muv-Mh median relation (z=10, dust-attenuated, no scatter) -> inversion table
# ---------------------------------------------------------------------------
mason15 = Mason15(z=10.0)
muv_mh_dict = mason15.Muv_Mh_dict   # {logMh: [Muv, Muv_dust]}

_logMh_arr = np.array(sorted(muv_mh_dict.keys()))
_muv_arr   = np.array([muv_mh_dict[lm][1] for lm in _logMh_arr])  # dust-attenuated median

# np.interp needs an ascending x-array; sort by Muv for the M_UV -> logMh direction
_order_by_muv = np.argsort(_muv_arr)
_muv_sorted_asc    = _muv_arr[_order_by_muv]
_logMh_sorted_by_muv = _logMh_arr[_order_by_muv]


def sigma_uv_at_fixed_muv(samples, muv_target):
    """For each posterior sample (Muv_add, sigmaUV_a, sigmaUV_b), find the halo
    mass where the *shifted* median relation hits muv_target, then evaluate
    sigma_UV(Mh) = sigmaUV_a*(logMh-12) + sigmaUV_b there."""
    Muv_add, sigmaUV_a, sigmaUV_b = samples[:, 0], samples[:, 1], samples[:, 2]
    target_unshifted = muv_target - Muv_add          # invert the shift first
    logMh = np.interp(target_unshifted, _muv_sorted_asc, _logMh_sorted_by_muv)
    sigma = sigmaUV_a * (logMh - 12) + sigmaUV_b
    return np.maximum(sigma, 1e-3)                    # same floor as uvlf.pMuv_Mh


def muv_at_fixed_logMh(samples, logMh_target):
    """M_UV predicted at a fixed halo mass: table lookup (interpolated) + shift."""
    Muv_add = samples[:, 0]
    muv_med = np.interp(logMh_target, _logMh_arr, _muv_arr)  # logMh_arr already ascending
    return muv_med + Muv_add


derived_sigma = {name: sigma_uv_at_fixed_muv(s, SIGMA_UV_TARGET_MUV) for name, s in posteriors.items()}
derived_muv   = {name: muv_at_fixed_logMh(s, MUV_AT_LOGMH_TARGET)   for name, s in posteriors.items()}

# ---------------------------------------------------------------------------
# Printed summary: medians, 68% CI, contraction ratios
# ---------------------------------------------------------------------------
def summarize(d, label):
    print(f"\n{label}:")
    widths = {}
    for name, vals in d.items():
        lo, med, hi = np.percentile(vals, [16, 50, 84])
        widths[name] = hi - lo
        print(f"  {name:10s}: {med:+.3f}  (68% CI [{lo:+.3f}, {hi:+.3f}], width {hi - lo:.3f})")
    if "UVLF-only" in widths:
        for name in widths:
            if name != "UVLF-only":
                print(f"  contraction {name} vs UVLF-only: {widths['UVLF-only'] / widths[name]:.2f}x")
    if "d1-only" in widths and "full-MLP" in widths:
        print(f"  contraction full-MLP vs d1-only:       {widths['d1-only'] / widths['full-MLP']:.2f}x")
    if "full-MLP" in widths and "full-MLP+UVLF" in widths:
        print(f"  contraction full-MLP+UVLF vs full-MLP: {widths['full-MLP'] / widths['full-MLP+UVLF']:.2f}x")


summarize(derived_sigma, f"sigma_UV(M_UV={SIGMA_UV_TARGET_MUV:.1f})")
summarize(derived_muv,   f"M_UV(log Mh={MUV_AT_LOGMH_TARGET:.1f})")

# ---------------------------------------------------------------------------
# Figure 1: overlaid raw-parameter corner plot
# ---------------------------------------------------------------------------
fig = None
for name, s in posteriors.items():
    fig = corner.corner(
        s, labels=PARAM_LABELS, fig=fig, color=COLORS[name],
        quantiles=[0.16, 0.5, 0.84] if fig is None else None,
        bins=40, smooth=1.0, range=PARAM_RANGE, levels=[0.68, 0.95],
        plot_datapoints=False, plot_density=False, fill_contours=True,
        label_kwargs={"fontsize": 13},
    )

legend_handles = [plt.Line2D([], [], color=COLORS[name], lw=2.5, label=name)
                   for name in SAMPLES]
fig.legend(handles=legend_handles, loc="upper right", fontsize=12, bbox_to_anchor=(1.0, 1.0))
fig.suptitle("Posterior gain: UVLF-only $\\to$ d1-only $\\to$ full MLP $\\to$ full MLP + UVLF",
             fontsize=13, y=1.02)
fig.savefig(f"{OUTPUT_DIR}/posterior_gain_corner.pdf", bbox_inches="tight")
print(f"\nSaved: {OUTPUT_DIR}/posterior_gain_corner.pdf")

# ---------------------------------------------------------------------------
# Figure 2: derived-quantity comparison, 2 panels
# ---------------------------------------------------------------------------
fig2, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for name in SAMPLES:
    axes[0].hist(derived_sigma[name], bins=50, histtype="step", lw=2.2,
                 color=COLORS[name], density=True, label=name)
    axes[1].hist(derived_muv[name], bins=50, histtype="step", lw=2.2,
                 color=COLORS[name], density=True, label=name)

axes[0].set_xlabel(rf"$\sigma_{{\rm UV}}(M_{{\rm UV}}={SIGMA_UV_TARGET_MUV:.0f})$", fontsize=13)
axes[1].set_xlabel(rf"$M_{{\rm UV}}(\log M_h={MUV_AT_LOGMH_TARGET:.0f})$", fontsize=13)
for ax in axes:
    ax.set_ylabel("posterior density", fontsize=12)
    ax.legend(fontsize=10.5, framealpha=0.7)

fig2.tight_layout()
fig2.savefig(f"{OUTPUT_DIR}/posterior_gain_derived.pdf", bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/posterior_gain_derived.pdf")