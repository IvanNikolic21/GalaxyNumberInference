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

from train_nre import env_to_array, normalize_params
from train_halo_mass import HaloMassNetwork


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

    config = np.load(args.model_dir / "model_config.npz")
    norm   = np.load(args.model_dir / "normalization.npz")
    param_min, param_max = norm['param_min'], norm['param_max']
    hidden_dims  = list(config['hidden_dims'])
    dropout      = float(config['dropout'])
    input_dim    = int(config['input_dim'])
    only_angular = bool(int(config['only_angular']))
    summary_mode = bool(int(config['summary_mode'])) if 'summary_mode' in config else False

    model = HaloMassNetwork(input_dim, hidden_dims, dropout)
    model.load_state_dict(torch.load(args.model_dir / "halo_mass_best.pt", map_location="cpu"))
    model.eval()

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
            coords, offsets = data['coords'], data['offsets']
            params = data['params']
            bright_logmh = data['bright_logmh']
        except Exception:
            continue

        n_envs = len(offsets) - 1
        if n_envs == 0:
            continue
        idx = rng.integers(0, n_envs)
        env = coords[offsets[idx]:offsets[idx + 1]]
        if len(env) == 0:
            continue

        flat, n_norm, n, dists = env_to_array(env, summary_mode=summary_mode, only_angular=only_angular)
        x = torch.from_numpy(np.concatenate([flat, n_norm]))
        theta = torch.from_numpy(normalize_params(params, param_min, param_max))
        x_in = torch.cat([x, theta]).unsqueeze(0).float()

        with torch.no_grad():
            mu, log_sigma = model(x_in)
        true_logmh.append(float(bright_logmh[idx]))
        pred_mu.append(float(mu[0]))
        pred_sigma.append(float(torch.exp(log_sigma[0])))

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
