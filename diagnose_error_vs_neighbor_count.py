#!/usr/bin/env python
"""
diagnose_error_vs_neighbor_count.py
--------------------------------------
Tests the "information floor" hypothesis raised after v8 (Deep Sets): three
structurally different fixes -- LDS loss reweighting, beta-NLL optimization
fix, and a permutation-invariant architecture -- all left RMS error and the
shrinkage-to-the-mean bias pattern essentially unchanged (v7 vs v8: RMS
0.361 dex in BOTH). That pattern is more consistent with the model being
starved of INFORMATION (too few real neighbors in the input) than with it
being under-fit or badly encoded.

This script checks that directly: for a sample of examples from the
database, it records each one's prediction error alongside its TRUE
neighbor count -- both the truncated count n (how many of the
MAX_NEIGHBORS=10 input slots are actually filled with real data; this is
exactly what the network's phi/pooling sees) and the uncapped count n_total
(how many neighbors actually existed before truncation, recoverable exactly
from the n_norm feature since it was stored as n_total/200.0 in float32) --
then reports:
  1. Correlation (Pearson r and Spearman rho) between |error| and each of
     n, n_total, across the whole sample.
  2. Binned mean |error| vs n and vs n_total, plotted.
  3. The SAME 8 true-logMh bins log_per_bin_metrics uses during training,
     but now also showing each bin's mean n and mean n_total -- a direct
     confound check: are the sparse/tail mass bins (where bias is worst)
     ALSO the low-neighbor-count examples? If so, no amount of reweighting
     or architecture change targeting "rare mass bin" will reduce those
     examples' irreducible error, because the real deficiency is
     information (how many real neighbors existed), not which mass bin
     they happen to fall in.

Usage
-----
    python diagnose_error_vs_neighbor_count.py \\
        --model-dir /groups/astro/ivannik/projects/Neighbors/halo_mass_model_multibox4_v8 \\
        --database-dir /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed1955 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed2027 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed3142 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed4242 \\
        --n-examples 2000 \\
        --output /groups/astro/ivannik/projects/Neighbors/halo_mass_model_multibox4_v8/error_vs_neighbor_count.pdf
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_nre import env_to_array, normalize_params, N_PARAMS
from train_halo_mass import HaloMassNetwork, HaloMassSetNetwork


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--database-dir", type=Path, nargs='+', required=True)
    p.add_argument("--n-examples", type=int, default=2000,
                   help="Larger than evaluate_halo_mass_model.py's default (500) -- this script "
                        "needs enough examples per neighbor-count AND per mass bin to make the "
                        "binned means meaningful, not just enough for one overall RMS number.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mass-bins", type=int, default=8,
                   help="Same bin count as train_halo_mass.py's --diag-bins default usage, for a "
                        "directly comparable table.")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def _spearman(x, y):
    # Spearman rho = Pearson correlation of the RANKS -- avoids needing
    # scipy just for this one diagnostic. np.argsort(np.argsort(x)) is the
    # standard trick for turning values into their 0-indexed ranks without
    # an external dependency.
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


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
    architecture = str(config['architecture']) if 'architecture' in config else "mlp"

    if architecture == "deepset":
        phi_dims      = list(config['phi_dims'])
        pooling       = str(config['pooling'])
        max_neighbors = int(config['max_neighbors'])
        n_feat        = (input_dim - 1 - N_PARAMS) // max_neighbors
        model = HaloMassSetNetwork(max_neighbors, n_feat, phi_dims, hidden_dims, dropout, pooling=pooling)
    else:
        model = HaloMassNetwork(input_dim, hidden_dims, dropout)
    model.load_state_dict(torch.load(args.model_dir / "halo_mass_best.pt", map_location="cpu"))
    model.eval()

    files = []
    for db_dir in args.database_dir:
        files.extend(sorted(Path(db_dir).glob("nre_*.npz")))
    rng.shuffle(files)

    true_logmh, pred_mu, n_truncated, n_total_list = [], [], [], []

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
            mu, _ = model(x_in)

        true_logmh.append(float(bright_logmh[idx]))
        pred_mu.append(float(mu[0]))
        n_truncated.append(int(n))
        # n_norm was computed as n_total/200.0 in float32 -- round-trips
        # exactly for any realistic integer n_total, so this recovers the
        # UNCAPPED neighbor count without needing env_to_array to return it
        # directly (it only returns the truncated n).
        n_total_list.append(int(round(float(n_norm[0]) * 200.0)))

    true_logmh  = np.array(true_logmh)
    pred_mu     = np.array(pred_mu)
    n_truncated = np.array(n_truncated)
    n_total     = np.array(n_total_list)
    abs_err     = np.abs(true_logmh - pred_mu)

    print(f"Evaluated {len(true_logmh)} examples.")
    print(f"  n (truncated, 0-{n_truncated.max()}) range: [{n_truncated.min()}, {n_truncated.max()}]")
    print(f"  n_total (uncapped) range: [{n_total.min()}, {n_total.max()}]")
    print()
    print("Correlation between |prediction error| and neighbor count "
          "(negative = fewer neighbors -> bigger errors, supporting the information-floor hypothesis):")
    print(f"  |error| vs n (truncated):   Pearson r={np.corrcoef(abs_err, n_truncated)[0,1]:+.3f}  "
          f"Spearman rho={_spearman(abs_err, n_truncated):+.3f}")
    print(f"  |error| vs n_total:         Pearson r={np.corrcoef(abs_err, n_total)[0,1]:+.3f}  "
          f"Spearman rho={_spearman(abs_err, n_total):+.3f}")
    print()

    # --- confound check: do the same (sparse) mass bins that show the worst
    # bias/error also have systematically lower neighbor counts? ---
    edges = np.linspace(true_logmh.min(), true_logmh.max() + 1e-6, args.mass_bins + 1)
    bin_idx = np.clip(np.digitize(true_logmh, edges) - 1, 0, args.mass_bins - 1)
    print(f"Per-mass-bin table ({args.mass_bins} bins, same scheme as training's log_per_bin_metrics):")
    print(f"  {'bin range':<16}{'n_ex':>6}{'mean|err|':>11}{'mean n':>9}{'mean n_tot':>12}")
    for b in range(args.mass_bins):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        print(f"  [{edges[b]:5.2f},{edges[b+1]:5.2f})"
              f"{m.sum():6d}{abs_err[m].mean():11.3f}{n_truncated[m].mean():9.2f}{n_total[m].mean():12.1f}")

    # --- figure: binned mean |error| vs n and vs n_total ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, counts, label, bins in [
        (axes[0], n_truncated, "n (truncated, 0-10 -- what the network actually sees)",
         np.arange(-0.5, n_truncated.max() + 1.5, 1)),
        (axes[1], n_total, "n_total (uncapped true neighbor count)",
         np.linspace(n_total.min(), n_total.max() + 1e-6, 15)),
    ]:
        bidx = np.clip(np.digitize(counts, bins) - 1, 0, len(bins) - 2)
        centers, means, sems = [], [], []
        for i in range(len(bins) - 1):
            m = bidx == i
            if m.sum() < 3:
                continue
            centers.append(0.5 * (bins[i] + bins[i + 1]))
            means.append(abs_err[m].mean())
            sems.append(abs_err[m].std() / np.sqrt(m.sum()))
        ax.scatter(counts, abs_err, s=8, alpha=0.15, color="tab:blue")
        ax.errorbar(centers, means, yerr=sems, fmt='o-', color="tab:red", capsize=3,
                    label="binned mean |error| +/- SEM")
        ax.set_xlabel(label)
        ax.set_ylabel(r"|true $-$ predicted| $\log_{10}(M_h/M_\odot)$")
        ax.legend(fontsize=8)

    fig.suptitle(f"Prediction error vs. neighbor count ({len(true_logmh)} examples, {architecture})")
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
