#!/usr/bin/env python
"""
plot_sbc_rank_ecdf.py
-----------------------
Companion to run_sbc_aggregate.py's rank-histogram plot: the other classic
SBC diagnostic, normalized rank on x vs. empirical CDF on y, compared to
the diagonal (uniform), with a distribution-free (DKW inequality) confidence
band. This is the more commonly-recognized SBC visualization (used by
default in sbi/arviz-style tooling); run_sbc_aggregate.py's histogram tests
the same null hypothesis on the same rank data -- its KS D-statistic per
parameter is exactly the max vertical deviation this plot shows visually.

Usage
-----
    python plot_sbc_rank_ecdf.py --sbc-dir /groups/astro/ivannik/projects/Neighbors/sbc
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARAM_NAMES  = ["Muv_add", "sigmaUV_a", "sigmaUV_b"]
PARAM_LABELS = [r"$M_{\rm UV,add}$", r"$\sigma_a$", r"$\sigma_b$"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sbc-dir", type=Path, required=True,
                   help="Same --output-dir passed to run_sbc_one_truth.py -- expects a 'ranks' "
                        "subdirectory full of rank_truth*.npz files.")
    p.add_argument("--confidence", type=float, default=0.95,
                   help="Confidence level for the DKW band. Default 0.95.")
    return p.parse_args()


def main():
    args = parse_args()
    rank_files = sorted((args.sbc_dir / "ranks").glob("rank_truth*.npz"))
    if not rank_files:
        raise SystemExit(f"No rank_truth*.npz files found in {args.sbc_dir / 'ranks'}")

    ranks, n_thins = [], []
    for f in rank_files:
        d = np.load(f)
        if bool(d["skipped"]):
            continue
        ranks.append(d["rank"])
        n_thins.append(int(d["n_thin"]))
    if not ranks:
        raise SystemExit("Every truth was skipped -- nothing to plot.")

    ranks = np.array(ranks)
    n_thins = np.array(n_thins)
    normalized = ranks / n_thins[:, np.newaxis]
    n = len(normalized)

    # DKW inequality: distribution-free (1-alpha) confidence band for a uniform ECDF,
    # half-width = sqrt(ln(2/alpha) / (2n)) -- same at every x, no simulation needed.
    alpha = 1.0 - args.confidence
    eps = np.sqrt(np.log(2 / alpha) / (2 * n))

    plt.style.use("seaborn-v0_8-ticks")
    plt.rcParams.update({"font.size": 13, "xtick.top": True, "ytick.right": True,
                         "xtick.direction": "in", "ytick.direction": "in"})
    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(4.3 * len(PARAM_NAMES), 4.3), sharey=True)

    x_diag = np.linspace(0, 1, 200)
    for i, (name, label, ax) in enumerate(zip(PARAM_NAMES, PARAM_LABELS, axes)):
        x = np.sort(normalized[:, i])
        y = np.arange(1, n + 1) / n
        ax.step(np.concatenate([[0], x, [1]]), np.concatenate([[0], y, [1]]),
                where="post", color="#2171b5", lw=2, label="empirical rank CDF")
        ax.plot(x_diag, x_diag, "k--", lw=1.2, label="ideal (uniform)")
        ax.fill_between(x_diag, np.clip(x_diag - eps, 0, 1), np.clip(x_diag + eps, 0, 1),
                         color="gray", alpha=0.25, label=f"{int(args.confidence*100)}% DKW band")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("normalized rank")
        ax.set_title(label, fontsize=13)
        if i == 0:
            ax.set_ylabel("empirical CDF")
            ax.legend(fontsize=9, frameon=False, loc="upper left")

    fig.suptitle(f"SBC rank ECDF vs. diagonal ({n} test truths)", fontsize=13)
    fig.tight_layout()
    out = args.sbc_dir / "sbc_rank_ecdf.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
