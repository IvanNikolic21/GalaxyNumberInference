#!/usr/bin/env python
"""
plot_panoramic_comparison.py
-----------------------------
Plot our sigma_CV(M_UV) predictions against the PANORAMIC z~10 cosmic
variance measurements (Weibel et al. 2025, arXiv:2512.14212), for a direct
side-by-side comparison.

Loads the Gamma/NB fit summary saved by run_cosmic_variance.py (no
recomputation) and overlays the paper's reported sigma_CV values
(PANORAMIC_SIGMA_CV in cosmic_variance.py). Both are sigma_CV =
sqrt(sigma_CV^2), not the squared quantity — see the note in
cosmic_variance.py for why.

fiducial/stochastic use the 2 cMpc-resolution box's fit at M_UV<-20.5 (where
the original 1 cMpc box's halo-discreteness dip in the UVLF lands) and the
original (1 cMpc) box's fit at the other thresholds, but are plotted/labeled
as plain "fiducial"/"stochastic" — the box swap is an internal data-quality
choice, not something worth surfacing in the legend.

Usage
-----
    python plot_panoramic_comparison.py --n-realizations 100
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from cosmic_variance import (
    PointingConfig,
    load_gamma_summary,
    gamma_fit_to_reference,
    plot_fractional_cosmic_variance,
    PANORAMIC_SIGMA_CV,
)
from run_cosmic_variance import (
    CACHE_DIR, OUTPUT_DIR, Z_RANGES, PREJWST_MUV_PATH, PREJWST_N_REALIZATIONS,
    ALT_N_REALIZATIONS, PREJWST_MODEL, ALT_MODEL_FIDUCIAL, ALT_MODEL_STOCHASTIC,
)

# M_UV<-20.5 is where the 1 cMpc box's halo-mass-quantization dip lands (see
# the UVLF discreteness discussion) -- swap in the 2 cMpc box's fit there only.
SWAP_AT_2CMPC = {-20.5}


def _splice_fits(primary: dict, alt: dict, swap_at: set, thresholds) -> dict:
    """Per-threshold fit dict, using `alt` at `swap_at` thresholds and
    `primary` everywhere else."""
    return {t: (alt[t] if t in swap_at else primary[t]) for t in thresholds}


def parse_args():
    p = argparse.ArgumentParser(description="Plot our sigma_CV vs PANORAMIC")
    p.add_argument("--n-realizations", type=int, default=100)
    p.add_argument("--n-trials", type=int, default=2000)
    p.add_argument("--group-size", type=int, default=28)
    p.add_argument("--fov-area-arcmin2", type=float, default=9.7)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PointingConfig(group_size=args.group_size, n_trials=args.n_trials)

    fov_tag = f"fov{args.fov_area_arcmin2:g}"
    pj_tag = f"pj-{PREJWST_MUV_PATH.stem}-real{PREJWST_N_REALIZATIONS}"
    alt_tag = f"alt2cmpc-real{ALT_N_REALIZATIONS}"
    gamma_summary_path = (
        CACHE_DIR
        / f"gamma_summary_real{args.n_realizations}_trials{args.n_trials}_g{args.group_size}_{fov_tag}_{pj_tag}_{alt_tag}.npz"
    )
    if not gamma_summary_path.exists():
        raise FileNotFoundError(
            f"No Gamma/NB fit summary at {gamma_summary_path} — run run_cosmic_variance.py "
            "with matching --n-realizations/--n-trials/--group-size/--fov-area-arcmin2 first."
        )

    gamma_fits = load_gamma_summary(gamma_summary_path)

    reference = dict(PANORAMIC_SIGMA_CV)
    for model, alt_model in (("fiducial", ALT_MODEL_FIDUCIAL), ("stochastic", ALT_MODEL_STOCHASTIC)):
        for zrange_label, (zlo, zhi) in Z_RANGES.items():
            merged_fits = _splice_fits(
                gamma_fits[model][zrange_label], gamma_fits[alt_model][zrange_label],
                SWAP_AT_2CMPC, cfg.thresholds,
            )
            label, entry = gamma_fit_to_reference(model, zlo, zhi, zrange_label, merged_fits, cfg.thresholds)
            reference[label] = entry
    for zrange_label, (zlo, zhi) in Z_RANGES.items():
        label, entry = gamma_fit_to_reference(
            PREJWST_MODEL, zlo, zhi, zrange_label, gamma_fits[PREJWST_MODEL][zrange_label], cfg.thresholds,
        )
        reference[label] = entry

    fig = plot_fractional_cosmic_variance({}, cfg.thresholds, Z_RANGES, reference=reference)

    out_path = OUTPUT_DIR / f"sigma_cv_vs_panoramic_real{args.n_realizations}_trials{args.n_trials}_g{args.group_size}_{fov_tag}_{pj_tag}_{alt_tag}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()