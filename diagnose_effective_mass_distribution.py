#!/usr/bin/env python
"""
diagnose_effective_mass_distribution.py
--------------------------------------------
Follow-up to the v9 result: the UVLF-consistent-only eval plot
(eval_vs_truth_uvlf5sigma_v9.pdf) showed essentially the SAME residual
shrinkage-to-the-mean shape as v8's, despite --uvlf-reweight training the
network to focus almost entirely on the 14-ish UVLF-consistent theta
(18% of 77 scanned). Ivan's read: UVLF reweighting didn't fix the real
problem, it just moved where the bias sits.

Before concluding the architecture needs more work, this checks a cheaper,
more specific hypothesis: --reweight-by-mass's LDS smoothing is computed
over the GLOBAL true-logMh histogram (dominated by the 82% UVLF-
IMPLAUSIBLE majority of examples), not over the mass distribution WITHIN
the UVLF-consistent subset specifically. If those ~14 surviving theta
themselves produce a mass distribution skewed toward the high end (plausible,
since low-scatter/UVLF-consistent models mostly only make genuinely massive
halos bright in the first place), LDS and UVLF-reweighting might not
compose the way train_halo_mass.py's design assumed -- the network could
still have trained on an effectively mass-imbalanced distribution WITHIN
the astrophysically relevant region, reproducing the same residual shape
without any architecture explanation needed.

This builds the EXACT SAME HaloMassDataset the v9 training run built (same
--reweight-by-mass/--uvlf-reweight/etc. settings -> identical self.weights,
self.logmh, self.params), then separately flags which examples belong to a
UVLF-consistent theta (same chi2->sigma conversion as
plot_eval_vs_truth_uvlf_consistent.py), and compares, WITHIN that subset
only:
  1. The UNWEIGHTED true-logMh histogram (what the astrophysics alone gives).
  2. The ACTUAL COMBINED-WEIGHT-weighted histogram (what the network
     effectively trained on, LDS x catalog-count x UVLF x clipping, all of
     it, since this literally re-runs HaloMassDataset.__init__).
If (2) is still as skewed as (1), LDS isn't flattening the physically
relevant region at all -- confirming Ivan's instinct that the residual bias
isn't about UVLF-consistency weighting per se, just not where Ivan might
expect (a composition/scope mismatch between the two reweighting schemes,
rather than needing a different architecture).

Usage
-----
    python diagnose_effective_mass_distribution.py \\
        --database-dir /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed1955 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed2027 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed3142 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed4242 \\
        --output /groups/astro/ivannik/projects/Neighbors/halo_mass_model_multibox4_v9/effective_mass_distribution.pdf
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_halo_mass import HaloMassDataset
from uvlf import Mason15, Observations


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database-dir", type=Path, nargs='+', required=True)
    p.add_argument("--output", type=Path, required=True)
    # Everything below defaults to the EXACT v9 training settings, so a
    # no-override run reconstructs v9's actual effective training weights.
    p.add_argument("--reweight-alpha", type=float, default=0.5)
    p.add_argument("--reweight-bins", type=int, default=30)
    p.add_argument("--lds-kernel-sigma", type=float, default=2.0)
    p.add_argument("--uvlf-sigma-scale", type=float, default=5.0)
    p.add_argument("--max-weight-ratio", type=float, default=10.0)
    p.add_argument("--max-sigma", type=float, default=5.0,
                   help="Threshold (equivalent UVLF sigma) for calling an example's theta "
                        "'UVLF-consistent' for THIS diagnostic's stratification -- same default "
                        "as plot_eval_vs_truth_uvlf_consistent.py's --max-sigma.")
    p.add_argument("--mass-bins", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()

    log_prefix = "[diagnose_effective_mass_distribution]"
    print(f"{log_prefix} Computing parameter normalization ...")
    all_params = []
    for db_dir in args.database_dir:
        for path in sorted(Path(db_dir).glob("nre_*.npz")):
            try:
                all_params.append(np.load(path)['params'])
            except Exception:
                continue
    all_params = np.array(all_params)
    param_min = all_params.min(axis=0)
    param_max = all_params.max(axis=0)

    print(f"{log_prefix} Building HaloMassDataset with v9's exact settings "
          f"(this re-runs the same data-loading + reweighting as training) ...")
    dataset = HaloMassDataset(
        database_dirs=args.database_dir,
        param_min=param_min, param_max=param_max,
        max_per_catalog=0,
        summary_mode=False, only_angular=True,
        weight_by_catalog_count=True,
        reweight_by_mass=True,
        reweight_alpha=args.reweight_alpha,
        reweight_bins=args.reweight_bins,
        lds_kernel_sigma=args.lds_kernel_sigma,
        uvlf_reweight=True,
        uvlf_sigma_scale=args.uvlf_sigma_scale,
        max_weight_ratio=args.max_weight_ratio,
    )

    logmh   = np.array(dataset.logmh)
    weights = np.array(dataset.weights)
    params  = np.array(dataset.params)  # (N, 3): Muv_add, sigmaUV_a, sigmaUV_b per example

    print(f"{log_prefix} Flagging UVLF-consistent examples (<= {args.max_sigma} sigma, "
          f"same conversion as plot_eval_vs_truth_uvlf_consistent.py) ...")
    mason15 = Mason15(z=10.0)
    obs = Observations(ang=False, uvlf=True)
    muv_obs, phi_obs, (sig_p, sig_m) = obs.get_obs_uvlf_z10_Donnan24()
    muv_grid = np.linspace(-25, -13, 100)
    dof = len(muv_obs)
    sigma_cache = {}

    def theta_sigma(muv_add, sa, sb):
        key = (round(float(muv_add), 8), round(float(sa), 8), round(float(sb), 8))
        if key in sigma_cache:
            return sigma_cache[key]
        phi_pred_grid = mason15.calculate_UVLF(muv_add, sa, sb, Muv_grid=muv_grid)
        phi_pred = np.interp(muv_obs, muv_grid, phi_pred_grid)
        sigma_obs = np.where(phi_pred >= phi_obs, sig_p, sig_m)
        ll = -0.5 * np.sum(((phi_obs - phi_pred) / sigma_obs) ** 2)
        p = chi2.sf(-2.0 * ll, df=dof) / 2.0
        p = np.clip(p, 0.0, 1.0 - 1e-16)
        sigma_equiv = max(0.0, float(norm.isf(p)))
        sigma_cache[key] = sigma_equiv
        return sigma_equiv

    # Only compute once per DISTINCT theta, then broadcast -- same caching
    # pattern as every other UVLF diagnostic in this project.
    unique_params, inverse = np.unique(np.round(params, 8), axis=0, return_inverse=True)
    unique_sigmas = np.array([theta_sigma(*p_) for p_ in unique_params])
    example_sigmas = unique_sigmas[inverse]
    is_consistent = example_sigmas <= args.max_sigma

    n_total = len(logmh)
    n_consistent = int(is_consistent.sum())
    print(f"\n{log_prefix} {n_consistent}/{n_total} examples ({n_consistent/n_total:.2%}) "
          f"belong to a <= {args.max_sigma}-sigma UVLF-consistent theta "
          f"({len(unique_params[unique_sigmas <= args.max_sigma])}/{len(unique_params)} "
          f"distinct theta).")

    logmh_sub   = logmh[is_consistent]
    weights_sub = weights[is_consistent]

    edges = np.linspace(logmh_sub.min(), logmh_sub.max() + 1e-6, args.mass_bins + 1)
    bin_idx = np.clip(np.digitize(logmh_sub, edges) - 1, 0, args.mass_bins - 1)

    unweighted_counts = np.bincount(bin_idx, minlength=args.mass_bins).astype(float)
    weighted_counts   = np.zeros(args.mass_bins)
    for b in range(args.mass_bins):
        weighted_counts[b] = weights_sub[bin_idx == b].sum()

    unweighted_frac = unweighted_counts / unweighted_counts.sum()
    weighted_frac   = weighted_counts / weighted_counts.sum()

    print(f"\n{log_prefix} Within the UVLF-consistent subset ({args.mass_bins} bins):")
    print(f"  {'bin range':<16}{'n_ex':>7}{'raw frac':>10}{'weighted frac':>15}")
    for b in range(args.mass_bins):
        print(f"  [{edges[b]:5.2f},{edges[b+1]:5.2f})"
              f"{int(unweighted_counts[b]):7d}{unweighted_frac[b]:10.1%}{weighted_frac[b]:15.1%}")

    # Concentration measure: ratio of the most- to least-represented bin's
    # fraction -- 1.0 would mean perfectly flat; large values mean still
    # skewed despite the combined reweighting.
    nz = weighted_frac[weighted_frac > 0]
    conc_weighted   = float(nz.max() / nz.min()) if len(nz) > 1 else float('nan')
    nz_u = unweighted_frac[unweighted_frac > 0]
    conc_unweighted = float(nz_u.max() / nz_u.min()) if len(nz_u) > 1 else float('nan')
    print(f"\n  Max/min bin fraction ratio -- raw: {conc_unweighted:.1f}x   "
          f"after LDS+UVLF reweighting: {conc_weighted:.1f}x")
    print(f"  (close to 1x would mean the reweighting successfully flattened the mass "
          f"distribution WITHIN the physically relevant region; a large ratio here means it "
          f"did not, regardless of how well it worked on the full, mostly-implausible prior.)")

    fig, ax = plt.subplots(figsize=(7, 5))
    width = (edges[1] - edges[0]) * 0.4
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers - width / 2, unweighted_frac, width=width, label="raw (unweighted) fraction", alpha=0.7)
    ax.bar(centers + width / 2, weighted_frac, width=width, label="LDS+UVLF-weighted fraction", alpha=0.7)
    ax.set_xlabel(r"true $\log_{10}(M_h/M_\odot)$")
    ax.set_ylabel("fraction of examples / effective training weight")
    ax.set_title(f"Effective mass distribution within UVLF-consistent "
                 f"theta ($\\leq${args.max_sigma}$\\sigma$, n={n_consistent})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"\n{log_prefix} Saved: {args.output}")


if __name__ == "__main__":
    main()
