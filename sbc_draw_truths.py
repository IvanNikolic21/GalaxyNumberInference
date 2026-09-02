#!/usr/bin/env python
"""
sbc_draw_truths.py
-------------------
Draw a fixed, reproducible set of SBC test truths from the UVLF-only
posterior (item 4 of the NRE to-do list) -- NOT a uniform prior draw, so
SBC test truths are concentrated where the observed UVLF says they should
be, rather than wasted on wildly UVLF-inconsistent parts of the prior
(e.g. sigma_b=2.4) the way earlier anecdotal spot-checks were.

The UVLF-only posterior samples are MCMC output (correlated, 32-walker
emcee chain flattened), so a random subsample without replacement is used
rather than literal thinning-by-stride -- adequate given n_truths (~40) is
far smaller than the effective sample size of the ~48000 raw draws.

Run this ONCE to fix the truth list, then point run_sbc_one_truth.py's
--truths-file / --truth-index at it (e.g. one per SLURM array task).

Usage
-----
    python sbc_draw_truths.py \\
        --uvlf-posterior /groups/astro/ivannik/projects/Neighbors/UVLF_only_true/posterior_samples_N0_uvlfonly.npy \\
        --n-truths 40 --seed 42 \\
        --output /groups/astro/ivannik/projects/Neighbors/sbc/sbc_truths.npy
"""
import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uvlf-posterior", type=Path, required=True)
    p.add_argument("--n-truths", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    samples = np.load(args.uvlf_posterior)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(samples), size=args.n_truths, replace=False)
    truths = samples[idx]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, truths)

    names = ["Muv_add", "sigmaUV_a", "sigmaUV_b"]
    print(f"Drew {args.n_truths} SBC test truths from {args.uvlf_posterior}")
    print(f"Saved to {args.output}\n")
    print(f"{'idx':>4}  " + "  ".join(f"{n:>10}" for n in names))
    for i, row in enumerate(truths):
        print(f"{i:>4}  " + "  ".join(f"{v:>10.3f}" for v in row))


if __name__ == "__main__":
    main()