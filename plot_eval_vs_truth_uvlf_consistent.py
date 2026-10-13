#!/usr/bin/env python
"""
plot_eval_vs_truth_uvlf_consistent.py
-----------------------------------------
Same predicted-vs-true log(Mh) diagnostic as evaluate_halo_mass_model.py,
but restricted to examples whose PARENT THETA is consistent with the
observed UVLF (Donnan+24) to within --max-sigma -- a HARD cut this time
(unlike diagnose_uvlf_stratified_neighbor_count.py's continuous weighting),
per Ivan's explicit request, to see what the predicted-vs-true plot looks
like once the astrophysically implausible majority of the training prior
(found there: UVLF log-L ranging from -1.5 to -56,069,751.6 across just
2000 sampled theta -- i.e. nearly the WHOLE uniform prior is wildly
UVLF-inconsistent) is excluded rather than just downweighted.

Converts each theta's UVLF chi-square-like log-likelihood into an
EQUIVALENT GAUSSIAN SIGNIFICANCE via the standard chi-square-to-sigma
convention used throughout physics (e.g. "a 5-sigma detection"):
    chi2  = -2 * log_likelihood
    p     = chi2.sf(chi2, df=n_uvlf_points) / 2   # TWO-sided equivalent tail prob.
    sigma = norm.isf(p)                           # equivalent Gaussian Z-score
The /2 matters and is verified exactly against the dof=1 case: chi2 with 1
degree of freedom is literally Z^2 for a single Gaussian deviate Z, so
chi2=25 must map to EXACTLY sigma=5 -- chi2.sf(25, df=1) = 2*norm.sf(5)
(a two-sided tail), so halving it before norm.isf() is what recovers 5.0
exactly (omitting the /2 gives 4.865, a few percent off).
With n_uvlf_points = 7 (the Donnan+24 z~10 UVLF data vector length, see
uvlf.py's get_obs_uvlf_z10_Donnan24) -- theta is EVALUATED against this
data, not fit to it, so no degrees-of-freedom reduction for fitted
parameters is applied. This is the same log-likelihood used by
infer_nre.py's --use-uvlf path and check_uvlf_consistency.py, just
converted to a sigma for an intuitive --max-sigma cutoff instead of a raw
log-L number.

Because the prior is so overwhelmingly UVLF-inconsistent, most database
files are rejected before their (expensive) coords/offsets/bright_logmh
arrays are even loaded -- only 'params' is read first (np.load is lazy per-
array) to decide pass/fail cheaply. Passing files may be rare enough that
one bright-galaxy environment per file isn't enough to reach --n-examples,
so up to --max-envs-per-theta environments are drawn from each PASSING
file (unlike evaluate_halo_mass_model.py's one-per-file sampling).

Usage
-----
    python plot_eval_vs_truth_uvlf_consistent.py \\
        --model-dir /groups/astro/ivannik/projects/Neighbors/halo_mass_model_multibox4_v8 \\
        --database-dir /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed1955 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed2027 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed3142 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed4242 \\
        --n-examples 500 --max-sigma 5.0 \\
        --output /groups/astro/ivannik/projects/Neighbors/halo_mass_model_multibox4_v8/eval_vs_truth_uvlf5sigma.pdf
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm
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
    p.add_argument("--n-examples", type=int, default=500,
                   help="Target number of examples in the plot (same default as "
                        "evaluate_halo_mass_model.py, for a directly comparable figure).")
    p.add_argument("--max-sigma", type=float, default=5.0,
                   help="Hard cutoff: keep only theta whose UVLF equivalent-Gaussian-sigma "
                        "is <= this. 'Broad agreement' per Ivan's request -- 3.0 is stricter, "
                        "5.0 (default) is looser.")
    p.add_argument("--max-envs-per-theta", type=int, default=50,
                   help="Cap on how many environments to draw from one PASSING file, since "
                        "passing files may be rare enough that one-per-file (like "
                        "evaluate_halo_mass_model.py) can't reach --n-examples.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    config = np.load(args.model_dir / "model_config.npz")
    norm_  = np.load(args.model_dir / "normalization.npz")
    param_min, param_max = norm_['param_min'], norm_['param_max']
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

    print("Building Mason15(z=10.0) model and loading the observed UVLF ...")
    mason15 = Mason15(z=10.0)
    obs = Observations(ang=False, uvlf=True)
    muv_obs, phi_obs, (sig_p, sig_m) = obs.get_obs_uvlf_z10_Donnan24()
    muv_grid = np.linspace(-25, -13, 100)
    dof = len(muv_obs)
    print(f"  UVLF data vector length (degrees of freedom): {dof}")

    ll_cache = {}

    def theta_sigma(muv_add, sa, sb):
        # Equivalent-Gaussian-sigma for this theta's UVLF fit -- see the
        # module docstring above for the chi2 -> p -> sigma conversion.
        key = (round(float(muv_add), 8), round(float(sa), 8), round(float(sb), 8))
        if key in ll_cache:
            return ll_cache[key]
        phi_pred_grid = mason15.calculate_UVLF(muv_add, sa, sb, Muv_grid=muv_grid)
        phi_pred = np.interp(muv_obs, muv_grid, phi_pred_grid)
        sigma_obs = np.where(phi_pred >= phi_obs, sig_p, sig_m)
        ll = -0.5 * np.sum(((phi_obs - phi_pred) / sigma_obs) ** 2)
        chi2_val = -2.0 * ll
        # /2: chi2.sf is a ONE-SIDED tail probability, but a chi-square is
        # itself the two-sided-squared sum of Gaussian deviates, so the
        # standard "N-sigma" convention (e.g. "a 5-sigma result") is TWO-
        # SIDED -- verified against the exact dof=1 case, where chi2=25
        # must map to exactly 5.0 (since chi2 with 1 dof is literally Z^2):
        # chi2.sf(25, 1) = 2*norm.sf(5), so halving it first before norm.isf
        # is what recovers 5.0 exactly; without the /2 it comes out as 4.865.
        p = chi2.sf(chi2_val, df=dof) / 2.0
        p = np.clip(p, 0.0, 1.0 - 1e-16)  # avoid norm.isf(1.0) = -inf for a suspiciously-perfect fit
        sigma_equiv = max(0.0, float(norm.isf(p)))
        ll_cache[key] = sigma_equiv
        return sigma_equiv

    files = []
    for db_dir in args.database_dir:
        files.extend(sorted(Path(db_dir).glob("nre_*.npz")))
    rng.shuffle(files)

    true_logmh, pred_mu, pred_sigma = [], [], []
    n_theta_scanned = 0
    n_theta_passed  = 0

    for path in files:
        if len(true_logmh) >= args.n_examples:
            break
        try:
            data = np.load(path)
            if 'bright_logmh' not in data.files:
                continue
            params = data['params']  # cheap: reads only this one small array from the .npz
        except Exception:
            continue

        n_theta_scanned += 1
        sigma_equiv = theta_sigma(params[0], params[1], params[2])
        if sigma_equiv > args.max_sigma:
            continue  # reject BEFORE loading coords/offsets/bright_logmh -- the expensive arrays
        n_theta_passed += 1

        try:
            coords, offsets = data['coords'], data['offsets']
            bright_logmh = data['bright_logmh']
        except Exception:
            continue

        n_envs = len(offsets) - 1
        if n_envs == 0:
            continue
        env_indices = rng.permutation(n_envs)[:args.max_envs_per_theta]

        for idx in env_indices:
            if len(true_logmh) >= args.n_examples:
                break
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

    print(f"\nScanned {n_theta_scanned} theta points; {n_theta_passed} "
          f"({n_theta_passed / max(n_theta_scanned, 1):.2%}) passed the <={args.max_sigma}-sigma "
          f"UVLF cut.")

    if len(true_logmh) == 0:
        print("No examples passed the cut -- nothing to plot. Try a larger --max-sigma, or more "
              "--database-dir / a larger file pool.")
        return

    true_logmh  = np.array(true_logmh)
    pred_mu     = np.array(pred_mu)
    pred_sigma  = np.array(pred_sigma)

    print(f"Evaluated {len(true_logmh)} examples (target was {args.n_examples}).")
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
    ax.set_title(f"Halo-mass model, UVLF-consistent theta only "
                 f"($\\leq${args.max_sigma}$\\sigma$, {len(true_logmh)} examples)")
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
