#!/usr/bin/env python
"""
evaluate_halo_mass_model.py
-----------------------------
Quick sanity check for the trained halo-mass network: pull a batch of
(environment, theta, true log Mh) examples straight from the existing
database files, run them through the model (single forward pass, no MCMC
-- this isn't a ratio-estimator), and plot predicted (mu +/- sigma) vs
true log(Mh). No literal "corner plot" applies here (only one inferred
quantity), so this is the direct equivalent: the same kind of
predicted-vs-truth diagnostic used for the NRE SBC check, just without
needing posterior sampling.

Usage
-----
    python evaluate_halo_mass_model.py \\
        --model-dir /groups/astro/ivannik/projects/Neighbors/halo_mass_model_multibox4 \\
        --database-dir /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed1955 \\
        --n-examples 500 --output /groups/astro/ivannik/projects/Neighbors/halo_mass_model_multibox4/eval_vs_truth.pdf
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_halo_mass import load_halo_mass_model, build_input, mixture_outputs, mixture_mean_std


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--database-dir", type=Path, nargs='+', required=True)
    p.add_argument("--n-examples", type=int, default=500,
                   help="Random examples to sample across all given database dirs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Shared loader: handles architecture, --use-bright-muv and mixture heads.
    model, cfg = load_halo_mass_model(args.model_dir)

    # Gather candidate (file, env_index) pairs across all database dirs, then
    # randomly sample --n-examples of them -- avoids loading every file in full.
    files = []
    for db_dir in args.database_dir:
        files.extend(sorted(Path(db_dir).glob("nre_*.npz")))
    rng.shuffle(files)

    true_logmh = []
    pred_mu = []
    pred_sigma = []

    for path in files:
        if len(true_logmh) >= args.n_examples:
            break
        try:
            data = np.load(path)
            if 'bright_logmh' not in data.files:
                continue
            if cfg['use_bright_muv'] and 'bright_muv' not in data.files:
                continue
            coords, offsets = data['coords'], data['offsets']
            params = data['params']
            bright_logmh = data['bright_logmh']
            bright_muv = data['bright_muv'] if cfg['use_bright_muv'] else None
        except Exception:
            continue

        n_envs = len(offsets) - 1
        if n_envs == 0:
            continue
        idx = rng.integers(0, n_envs)
        env = coords[offsets[idx]:offsets[idx + 1]]
        if len(env) == 0:
            continue

        x_in = build_input(env, params, None if bright_muv is None else bright_muv[idx], cfg)

        with torch.no_grad():
            # Predictive mean/std (= mu, sigma for a single-Gaussian model).
            mu, sigma = mixture_mean_std(*mixture_outputs(model, x_in))
        true_logmh.append(float(bright_logmh[idx]))
        pred_mu.append(float(mu[0]))
        pred_sigma.append(float(sigma[0]))

    true_logmh = np.array(true_logmh)
    pred_mu    = np.array(pred_mu)
    pred_sigma = np.array(pred_sigma)
    print(f"Evaluated {len(true_logmh)} examples.")
    print(f"  RMS error (true - mu): {np.sqrt(np.mean((true_logmh - pred_mu)**2)):.3f} dex")
    print(f"  Mean predicted sigma:  {pred_sigma.mean():.3f} dex")
    print(f"  Fraction within predicted 1-sigma: "
          f"{np.mean(np.abs(true_logmh - pred_mu) < pred_sigma):.1%}  (expect ~68% if calibrated)")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(true_logmh, pred_mu, yerr=pred_sigma, fmt='o', alpha=0.4, markersize=3,
                elinewidth=0.5, capsize=0)
    lo = min(true_logmh.min(), pred_mu.min())
    hi = max(true_logmh.max(), pred_mu.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='y = x (perfect tracking)')
    ax.set_xlabel(r"true $\log_{10}(M_h/M_\odot)$")
    ax.set_ylabel(r"predicted $\log_{10}(M_h/M_\odot)$ ($\mu \pm \sigma$)")
    ax.legend(fontsize=9)
    ax.set_title(f"Halo-mass model: predicted vs. true ({len(true_logmh)} examples)")
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
