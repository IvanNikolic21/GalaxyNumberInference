#!/usr/bin/env python
"""
run_sbc_aggregate.py
---------------------
Aggregate per-truth rank statistics from run_sbc_one_truth.py into the
actual SBC calibration check: pooled rank histograms per parameter, plus a
KS test against the expected discrete-uniform distribution (Talts et al.
2018). Run once all SLURM array tasks from run_sbc_slurm.sh have finished.

Usage
-----
    python run_sbc_aggregate.py --sbc-dir /groups/astro/ivannik/projects/Neighbors/sbc
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kstest, uniform

PARAM_NAMES = ["Muv_add", "sigmaUV_a", "sigmaUV_b"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sbc-dir", type=Path, required=True,
                   help="Same --output-dir passed to run_sbc_one_truth.py -- expects a 'ranks' "
                        "subdirectory full of rank_truth*.npz files.")
    return p.parse_args()


def main():
    args = parse_args()
    rank_files = sorted((args.sbc_dir / "ranks").glob("rank_truth*.npz"))
    if not rank_files:
        raise SystemExit(f"No rank_truth*.npz files found in {args.sbc_dir / 'ranks'}")

    ranks, n_thins, skipped = [], [], 0
    for f in rank_files:
        d = np.load(f)
        if bool(d["skipped"]):
            skipped += 1
            continue
        ranks.append(d["rank"])
        n_thins.append(int(d["n_thin"]))

    if not ranks:
        raise SystemExit("Every truth was skipped -- nothing to aggregate.")
    if len(set(n_thins)) > 1:
        print(f"WARNING: n_thin varies across truths ({set(n_thins)}) -- some truths had fewer "
              f"posterior samples than requested. Uniformity test below normalizes by each "
              f"truth's own n_thin (rank/n_thin in [0,1]), which is still valid, but the raw "
              f"rank histogram mixes different bin counts -- read the normalized KS test, not "
              f"the histogram bin edges, as the primary result.")

    ranks = np.array(ranks)          # shape (M, 3)
    n_thins = np.array(n_thins)      # shape (M,)
    n_used = len(ranks)
    print(f"Aggregated {n_used} truths ({skipped} skipped -- zero usable environments or "
          f"failed inference).\n")

    # Normalize each truth's rank to [0, 1] by its own n_thin, since n_thin can
    # vary (see warning above) -- under correct calibration this should be
    # Uniform(0, 1) regardless of n_thin.
    normalized = ranks / n_thins[:, np.newaxis]

    plt.style.use("seaborn-v0_8-ticks")
    plt.rcParams.update({"font.size": 13, "xtick.top": True, "ytick.right": True,
                         "xtick.direction": "in", "ytick.direction": "in"})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    print(f"{'parameter':<12}  {'KS stat':>10}  {'p-value':>10}  verdict")
    print("-" * 55)
    for i, (name, ax) in enumerate(zip(PARAM_NAMES, axes)):
        stat, pvalue = kstest(normalized[:, i], uniform.cdf)
        verdict = "OK (p>0.05)" if pvalue > 0.05 else "MISCALIBRATED (p<=0.05)"
        print(f"{name:<12}  {stat:>10.4f}  {pvalue:>10.3e}  {verdict}")

        ax.hist(normalized[:, i], bins=10, range=(0, 1), color="#6baed6",
                edgecolor="white", density=True)
        ax.axhline(1.0, color="k", ls="--", lw=1, label="expected (uniform)")
        ax.set_xlabel(f"normalized rank: {name}")
        ax.set_title(f"p={pvalue:.3f}")
        if i == 0:
            ax.set_ylabel("density")
            ax.legend(fontsize=9, frameon=False)

    fig.suptitle(f"SBC rank uniformity ({n_used} test truths, "
                 f"drawn from the UVLF-only posterior)")
    fig.tight_layout()
    out_path = args.sbc_dir / "sbc_rank_uniformity.pdf"
    fig.savefig(out_path)
    print(f"\nSaved plot: {out_path}")

    np.savez(args.sbc_dir / "sbc_ranks_aggregated.npz",
             ranks=ranks, n_thins=n_thins, normalized=normalized)


if __name__ == "__main__":
    main()