#!/usr/bin/env python
"""
diagnose_uvlf_stratified_neighbor_count.py
----------------------------------------------
Follow-up to diagnose_error_vs_neighbor_count.py: that script found low-mass
"bright" hosts have ~100 true neighbors on average, high-mass hosts have
~11-12 -- the OPPOSITE of what real halo clustering would predict (more
massive halos should be MORE clustered, not less). The mechanism traced to
the code: build_nre_database.py selects both the "bright" host AND its
"faint" neighbors from the SAME theta-dependent MUV realization
(muvs = load_muv_catalog(...)), and since sigmaUV(Mh) grows at low mass in
this box (sigmaUV_a*(logMh-12)+sigmaUV_b, confirmed sigma_a<=0 and
logMh<12 always for this box), a theta that boosts a low-mass halo past
M_UV,0=-21.5 via large scatter ALSO smears many other nearby halos across
the M_UV,lim=-18.5 faint window by the same large scatter.

But Ivan's objection is the right one: we are NOT conditioning the training
prior on whether theta actually reproduces the observed UVLF. A theta that
needs huge scatter to make a low-mass halo this bright would also badly
overpredict the bright-end UVLF -- i.e. it may be a physically implausible
corner of the (uniform) prior, in which case the neighbor-count-vs-host-mass
trend found before could be an artifact of including such points, not
something that would survive in a UVLF-consistent universe.

This script re-examines that trend CONTINUOUSLY STRATIFIED by each
example's parent theta's UVLF log-likelihood (no hard plausibility cutoff,
per Ivan's explicit request) -- reusing the EXACT same Mason15/Observations
UVLF log-likelihood already used by infer_nre.py's --use-uvlf path and
check_uvlf_consistency.py, not a new implausibility criterion invented here.

One approximation inherited from the existing pipeline, not introduced by
this script: Mason15's Muv_Mh_dict is only ever built for z=10.0 (see
uvlf.py's Mason15.__init__), and check_uvlf_consistency.py itself calls
Mason15(z=10.0) even though it was analyzing the z=10.5 problem truth that
motivated it -- this script follows that exact same established precedent
(z=10.0 Mason15 for the z=10.5 database) rather than inventing a different
answer to a question the rest of the pipeline has already settled the same
way.

Usage
-----
    python diagnose_uvlf_stratified_neighbor_count.py \\
        --model-dir /groups/astro/ivannik/projects/Neighbors/halo_mass_model_multibox4_v8 \\
        --database-dir /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed1955 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed2027 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed3142 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed4242 \\
        --n-examples 2000 \\
        --output /groups/astro/ivannik/projects/Neighbors/halo_mass_model_multibox4_v8/uvlf_stratified.pdf
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
from uvlf import Mason15, Observations


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--database-dir", type=Path, nargs='+', required=True)
    p.add_argument("--n-examples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mass-bins", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def _spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _weighted_mean(values, weights):
    return float(np.sum(values * weights) / np.sum(weights))


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

    # UVLF log-likelihood machinery -- IDENTICAL to check_uvlf_consistency.py
    # and infer_nre.py's --use-uvlf path, not reimplemented differently here.
    print("Building Mason15(z=10.0) model and loading the observed UVLF ...")
    mason15 = Mason15(z=10.0)
    obs = Observations(ang=False, uvlf=True)
    muv_obs, phi_obs, (sig_p, sig_m) = obs.get_obs_uvlf_z10_Donnan24()
    muv_grid = np.linspace(-25, -13, 100)

    # hmf.MassFunction (inside mason15.mf) doesn't depend on theta -- built
    # once above and reused for every theta below. Only Muv-dependent parts
    # of calculate_UVLF change per theta, so cache each distinct theta's
    # log-likelihood rather than recomputing it once per sampled example
    # (many examples share the same parent theta/file).
    ll_cache = {}

    def uvlf_loglike(muv_add, sa, sb):
        key = (round(float(muv_add), 8), round(float(sa), 8), round(float(sb), 8))
        if key in ll_cache:
            return ll_cache[key]
        phi_pred_grid = mason15.calculate_UVLF(muv_add, sa, sb, Muv_grid=muv_grid)
        phi_pred = np.interp(muv_obs, muv_grid, phi_pred_grid)
        sigma = np.where(phi_pred >= phi_obs, sig_p, sig_m)
        ll = -0.5 * np.sum(((phi_obs - phi_pred) / sigma) ** 2)
        ll_cache[key] = ll
        return ll

    files = []
    for db_dir in args.database_dir:
        files.extend(sorted(Path(db_dir).glob("nre_*.npz")))
    rng.shuffle(files)

    true_logmh, pred_mu, n_truncated, n_total_list, loglikes = [], [], [], [], []

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

        ll = uvlf_loglike(params[0], params[1], params[2])

        true_logmh.append(float(bright_logmh[idx]))
        pred_mu.append(float(mu[0]))
        n_truncated.append(int(n))
        n_total_list.append(int(round(float(n_norm[0]) * 200.0)))
        loglikes.append(ll)

    true_logmh  = np.array(true_logmh)
    pred_mu     = np.array(pred_mu)
    n_truncated = np.array(n_truncated)
    n_total     = np.array(n_total_list)
    loglikes    = np.array(loglikes)
    abs_err     = np.abs(true_logmh - pred_mu)

    print(f"\nEvaluated {len(true_logmh)} examples across {len(ll_cache)} distinct theta points.")
    print(f"  UVLF log-L range: [{loglikes.min():.1f}, {loglikes.max():.1f}]  "
          f"(less negative = more consistent with Donnan+24)")
    print()
    print("Correlation between UVLF log-L and each quantity (continuous, no cutoff):")
    print(f"  log-L vs n_total:        Pearson r={np.corrcoef(loglikes, n_total)[0,1]:+.3f}  "
          f"Spearman rho={_spearman(loglikes, n_total):+.3f}")
    print(f"  log-L vs true log(Mh):   Pearson r={np.corrcoef(loglikes, true_logmh)[0,1]:+.3f}  "
          f"Spearman rho={_spearman(loglikes, true_logmh):+.3f}")
    print(f"  log-L vs |error|:        Pearson r={np.corrcoef(loglikes, abs_err)[0,1]:+.3f}  "
          f"Spearman rho={_spearman(loglikes, abs_err):+.3f}")
    print()

    # Continuous UVLF-plausibility weight per example: softmax-style,
    # exp(ll - ll_max) so the best-fitting theta in the sample gets weight 1
    # and everything else is downweighted by how much worse its log-L is --
    # no hard cutoff, exactly as requested.
    weights = np.exp(loglikes - loglikes.max())

    edges = np.linspace(true_logmh.min(), true_logmh.max() + 1e-6, args.mass_bins + 1)
    bin_idx = np.clip(np.digitize(true_logmh, edges) - 1, 0, args.mass_bins - 1)

    print(f"Per-mass-bin table ({args.mass_bins} bins) -- UNWEIGHTED vs UVLF-log-L-WEIGHTED "
          f"mean n_total, and each bin's own mean log-L:")
    print(f"  {'bin range':<16}{'n_ex':>6}{'mean n_tot':>12}{'UVLF-wtd n_tot':>16}{'mean log-L':>12}")
    bin_mean_ll = []
    for b in range(args.mass_bins):
        m = bin_idx == b
        if m.sum() == 0:
            bin_mean_ll.append(np.nan)
            continue
        wtd = _weighted_mean(n_total[m], weights[m])
        mean_ll = loglikes[m].mean()
        bin_mean_ll.append(mean_ll)
        print(f"  [{edges[b]:5.2f},{edges[b+1]:5.2f})"
              f"{m.sum():6d}{n_total[m].mean():12.1f}{wtd:16.1f}{mean_ll:12.1f}")

    # --- figure ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sc = axes[0].scatter(loglikes, n_total, c=true_logmh, cmap="viridis", s=10, alpha=0.5)
    axes[0].set_xlabel("UVLF log-likelihood of parent theta")
    axes[0].set_ylabel("n_total (uncapped true neighbor count)")
    cb = fig.colorbar(sc, ax=axes[0])
    cb.set_label(r"true $\log_{10}(M_h/M_\odot)$")

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    unweighted_means = [n_total[bin_idx == b].mean() if (bin_idx == b).sum() > 0 else np.nan
                        for b in range(args.mass_bins)]
    weighted_means = [_weighted_mean(n_total[bin_idx == b], weights[bin_idx == b])
                       if (bin_idx == b).sum() > 0 else np.nan for b in range(args.mass_bins)]
    axes[1].plot(bin_centers, unweighted_means, 'o-', label="unweighted mean n_total")
    axes[1].plot(bin_centers, weighted_means, 's--', label="UVLF-log-L-weighted mean n_total")
    axes[1].set_xlabel(r"true $\log_{10}(M_h/M_\odot)$ (bin center)")
    axes[1].set_ylabel("mean n_total")
    axes[1].legend(fontsize=8)

    axes[2].bar(bin_centers, bin_mean_ll, width=(edges[1] - edges[0]) * 0.8)
    axes[2].set_xlabel(r"true $\log_{10}(M_h/M_\odot)$ (bin center)")
    axes[2].set_ylabel("mean UVLF log-L of contributing theta")

    fig.suptitle(f"UVLF-stratified neighbor-count check ({len(true_logmh)} examples, {architecture})")
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
