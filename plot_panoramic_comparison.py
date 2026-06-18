#!/usr/bin/env python
"""
plot_panoramic_comparison.py
-----------------------------
Plot our sigma_CV(M_UV) predictions against the PANORAMIC z~10 cosmic
variance measurements (Weibel et al. 2025, arXiv:2512.14212), for a direct
side-by-side comparison.

Loads an existing cache from run_cosmic_variance.py (no recomputation) and
overlays the paper's reported sigma_CV values (PANORAMIC_SIGMA_CV in
cosmic_variance.py) on the same axes. Both are sigma_CV = sqrt(sigma_CV^2),
not the squared quantity — see the note in cosmic_variance.py for why.

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
    load_cosmic_variance,
    plot_fractional_cosmic_variance,
    PANORAMIC_SIGMA_CV,
)
from run_cosmic_variance import (
    CACHE_DIR, OUTPUT_DIR, Z_RANGES, PREJWST_MUV_PATH, PREJWST_N_REALIZATIONS,
)


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
    cache_path = (
        CACHE_DIR
        / f"cosmic_variance_real{args.n_realizations}_trials{args.n_trials}_g{args.group_size}_{fov_tag}_{pj_tag}.npz"
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cache at {cache_path} — run run_cosmic_variance.py with matching "
            "--n-realizations/--n-trials/--group-size/--fov-area-arcmin2 first."
        )

    results = load_cosmic_variance(cache_path)
    fig = plot_fractional_cosmic_variance(
        results, cfg.thresholds, Z_RANGES, reference=PANORAMIC_SIGMA_CV,
    )

    out_path = OUTPUT_DIR / f"sigma_cv_vs_panoramic_real{args.n_realizations}_trials{args.n_trials}_g{args.group_size}_{fov_tag}_{pj_tag}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()