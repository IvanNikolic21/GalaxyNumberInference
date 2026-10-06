#!/usr/bin/env python
"""
sbc_diagnostic_plots.py
-------------------------
Two diagnostics for a failed SBC rank-uniformity check:

1. "posterior vs truth" scatter (3 panels, one per parameter): posterior
   median + 68% C.I. for each test truth, plotted against the true value.
   If the model is actually tracking theta, points should scatter around
   the y=x line. If the posterior is collapsing to roughly the same place
   regardless of the true input (the suspected failure mode here), points
   will cluster at a similar y-value despite the x-value (truth) varying.

2. Individual corner plots (one per test truth, up to --n-examples) with
   the true theta marked -- same corner.corner() style/ranges as
   infer_nre_d1.py's own plot, for a direct visual read per truth.

Usage
-----
    python sbc_diagnostic_plots.py --sbc-dir /groups/astro/ivannik/projects/Neighbors/sbc_multibox4 \\
        --n-examples 9
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

PARAM_NAMES = ["Muv_add", "sigmaUV_a", "sigmaUV_b"]
PARAM_LABELS = [r"$M_{\rm UV,add}$", r"$\sigma_{\rm UV,a}$", r"$\sigma_{\rm UV,b}$"]
RANGE = [(-1.5, 2.0), (-1.0, 1.5), (0.0, 3.0)]  # same as infer_nre_d1.py's corner plot


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sbc-dir", type=Path, required=True,
                   help="Same --output-dir passed to run_sbc_one_truth.py.")
    p.add_argument("--n-examples", type=int, default=9,
                   help="Number of individual corner plots to make (first N non-skipped truths).")
    return p.parse_args()


def load_truth_and_posterior(sbc_dir, idx):
    rank_path = sbc_dir / "ranks" / f"rank_truth{idx:03d}.npz"
    if not rank_path.exists():
        return None
    rank_data = np.load(rank_path)
    if bool(rank_data["skipped"]):
        return None
    theta = rank_data["theta"]

    infer_dir = sbc_dir / "infer" / f"truth{idx:03d}"
    candidates = sorted(infer_dir.glob("posterior_samples_*.npy"))
    if not candidates:
        return None
    posterior = np.load(candidates[-1])
    return theta, posterior


def main():
    args = parse_args()
    rank_files = sorted((args.sbc_dir / "ranks").glob("rank_truth*.npz"))
    indices = [int(f.stem.replace("rank_truth", "")) for f in rank_files]

    loaded = []
    for idx in indices:
        result = load_truth_and_posterior(args.sbc_dir, idx)
        if result is not None:
            loaded.append((idx, *result))
    print(f"Loaded {len(loaded)} / {len(indices)} non-skipped truth+posterior pairs.")

    # --- Diagnostic 1: posterior vs truth scatter ---------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for p, (ax, name, label) in enumerate(zip(axes, PARAM_NAMES, PARAM_LABELS)):
        truths = np.array([theta[p] for _, theta, _ in loaded])
        meds   = np.array([np.median(post[:, p]) for _, _, post in loaded])
        los    = np.array([np.percentile(post[:, p], 16) for _, _, post in loaded])
        his    = np.array([np.percentile(post[:, p], 84) for _, _, post in loaded])

        ax.errorbar(truths, meds, yerr=[meds - los, his - meds], fmt="o", alpha=0.7, capsize=3)
        lo, hi = RANGE[p]
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (perfect tracking)")
        ax.set_xlabel(f"true {label}")
        ax.set_ylabel(f"posterior median {label}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        if p == 0:
            ax.legend(fontsize=9)
    fig.suptitle(f"Posterior vs. truth ({len(loaded)} test truths) -- "
                 f"points hugging a horizontal line instead of y=x means the\n"
                 f"posterior isn't tracking the true parameter value")
    fig.tight_layout()
    out1 = args.sbc_dir / "diagnostic_posterior_vs_truth.pdf"
    fig.savefig(out1)
    print(f"Saved: {out1}")

    # --- Diagnostic 2: individual corner plots -------------------------------
    for idx, theta, posterior in loaded[:args.n_examples]:
        fig = corner.corner(
            posterior,
            labels=PARAM_LABELS,
            truths=list(theta),
            truth_color="red",
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 13},
            quantiles=[0.16, 0.5, 0.84],
            bins=40,
            smooth=1.0,
            range=RANGE,
            levels=[0.68, 0.95],
            color="black",
            plot_datapoints=False,
            plot_density=False,
            fill_contours=True,
        )
        fig.suptitle(f"Truth #{idx:03d}", y=1.02)
        out = args.sbc_dir / f"diagnostic_corner_truth{idx:03d}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
