#!/usr/bin/env python
"""
diagnose_heldout_calibration.py
-------------------------------
Held-out calibration test for a trained halo-mass model: is the remaining
"S-shape" in the predicted-vs-true plots a BUG / underfit, or the expected
behaviour of a posterior-mean estimator that only has limited information?

Why the existing plots can't answer that
----------------------------------------
evaluate_halo_mass_model.py and plot_eval_vs_truth_uvlf_consistent.py plot
predicted mu against TRUE log Mh. The network's mu approximates the
posterior mean E[log Mh | x, theta] (Gaussian-NLL regression; Bishop 2006,
"Pattern Recognition and Machine Learning", Sec. 1.5.5 -- the squared-loss
/ Gaussian-likelihood optimum is the conditional mean). A posterior mean is
unbiased in only ONE direction:
    E[ true | mu ] = mu        (holds for any well-trained posterior mean)
    E[ mu | true ] = true      (does NOT hold unless x pins down true exactly)
The second one, which is what a scatter of mu against true visualises, is
pulled toward the bulk of the training prior whenever the data are not
fully informative -- regression dilution / "regression to the mean"
(Galton 1886; in astronomy the same effect as Eddington bias, Eddington
1913, MNRAS 73, 359). So an S-shape in mu-vs-true is EXPECTED, and changing
the loss, architecture or reweighting only moves where it points.

This script performs the two tests that DO distinguish "bug" from
"information limit", on the validation split ONLY (the existing eval
scripts draw from the same databases the model trained on, so ~80% of their
points are training examples):

  1. Reverse binning: bin by PREDICTED mu, plot the mean TRUE log Mh in each
     bin. A well-trained posterior mean lies on y = x here.
  2. PIT (probability integral transform; Dawid 1984, J. R. Stat. Soc. A
     147, 278; Gneiting, Balabdaoui & Raftery 2007, J. R. Stat. Soc. B 69,
     243): u = F(true), the model's predicted CDF evaluated at the truth --
     Phi((true - mu) / sigma) for a single Gaussian, sum_k pi_k
     Phi((true - mu_k) / sigma_k) for a mixture (--n-components > 1). If
     the predicted distribution is the true conditional one, u ~ U(0, 1). A hump means sigma
     is too large, a U shape means it is too small, a slope means mu is
     biased. Plus the coverage curves -- central (fraction of true values
     inside the central q-interval, |u - 0.5| < q/2) and one-sided (u < q)
     vs q -- the cumulative version of the same test. The central curve
     alone is blind to SKEW (an excess in one tail cancels a deficit in the
     other, as happened for v10), which the one-sided curve catches --
     the regression analogue of the SBC rank checks used for the NRE
     (Talts et al. 2018, arXiv:1804.06788).

Two populations, one row of panels each:
  * "val, training-weighted": every validation example, weighted by the
    SAME per-example weights the model trained with (catalog-count x LDS x
    UVLF taper x clipping, rebuilt here by re-running HaloMassDataset with
    the training flags). The network approximates the posterior mean under
    THIS weighted distribution, so this is the strict self-consistency
    test: if it fails, the network is underfitting / buggy.
  * "val, UVLF <= max-sigma, unweighted": the validation examples whose
    parent theta passes the hard UVLF cut, equally weighted -- the
    physically relevant population, same cut as
    plot_eval_vs_truth_uvlf_consistent.py. Small departures here are
    expected because this distribution differs from the training one.

Reproducing the split
---------------------
train_halo_mass.py splits with torch.utils.data.random_split seeded by
--seed, over a dataset whose ORDER depends only on the file list and (if
--max-per-catalog > 0) the global NumPy RNG seeded by the same --seed. So
the split is reproduced exactly as long as (a) the same --database-dir list
in the same order, (b) the same --max-per-catalog, --val-frac and
--train-seed are passed, and (c) no new database files were patched since
training. The script prints the dataset size and the train/val counts;
check them against the "Total (environment, log Mh) pairs" and
"Train: ... Val: ..." lines in the training log (v10: 200370 pairs).

Usage (v10 settings)
--------------------
    python diagnose_heldout_calibration.py \\
        --model-dir /groups/astro/ivannik/projects/Neighbors/halo_mass_model_multibox4_v10 \\
        --database-dir /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed1955 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed2027 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed3142 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed4242 \\
        --max-per-catalog 0 --weight-by-catalog-count \\
        --reweight-by-mass --reweight-alpha 0.5 --reweight-bins 30 --lds-kernel-sigma 2.0 \\
        --uvlf-reweight --uvlf-sigma-scale 5.0 --max-weight-ratio 10.0 \\
        --output /groups/astro/ivannik/projects/Neighbors/halo_mass_model_multibox4_v10/heldout_calibration_v10.pdf
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_halo_mass import (HaloMassDataset, load_halo_mass_model, mixture_outputs,
                             mixture_mean_std, mixture_cdf)
from uvlf_consistency import uvlf_sigma_per_theta

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--database-dir", type=Path, nargs='+', required=True,
                   help="SAME directories, in the SAME order, as the training run.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output PDF. A .npz with the per-example val predictions is saved next to it.")
    # --- flags that must match the training run (defaults = train_halo_mass.py's defaults) ---
    p.add_argument("--train-seed", type=int, default=42,
                   help="The --seed the model was TRAINED with (fixes the train/val split).")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-per-catalog", type=int, default=200)
    p.add_argument("--weight-by-catalog-count", action="store_true")
    p.add_argument("--reweight-by-mass", action="store_true")
    p.add_argument("--reweight-alpha", type=float, default=0.5)
    p.add_argument("--reweight-bins", type=int, default=30)
    p.add_argument("--lds-kernel-sigma", type=float, default=2.0)
    p.add_argument("--uvlf-reweight", action="store_true")
    p.add_argument("--uvlf-sigma-scale", type=float, default=5.0)
    p.add_argument("--max-weight-ratio", type=float, default=10.0)
    # --- evaluation-only options ---
    p.add_argument("--max-sigma", type=float, default=5.0,
                   help="Hard UVLF cut (equivalent Gaussian sigma) for the physically-relevant subset.")
    p.add_argument("--n-examples", type=int, default=0,
                   help="Random subsample of the val split to evaluate (0 = all of it).")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the --n-examples subsample only (independent of --train-seed).")
    p.add_argument("--n-bins", type=int, default=12, help="Bins in predicted mu for the reverse-binning panel.")
    p.add_argument("--batch-size", type=int, default=4096)
    return p.parse_args()


def weighted_mean_se(x, w):
    # Weighted mean and its standard error, using Kish's effective sample size
    # n_eff = (sum w)^2 / sum w^2 (Kish 1965, "Survey Sampling") so that a bin
    # dominated by a few heavily-weighted examples gets an honestly large error bar.
    wsum = w.sum()
    mean = np.sum(w * x) / wsum
    var = np.sum(w * (x - mean) ** 2) / wsum
    n_eff = wsum ** 2 / np.sum(w ** 2)
    return mean, np.sqrt(var / n_eff), n_eff


def calibration_row(axes, true, mu, pit, w, n_bins, label):
    """Fill one row of 4 panels; returns a dict of summary numbers for the log."""
    w = w / w.sum()
    n_eff_total = 1.0 / np.sum(w ** 2)

    # (a) Forward binning (bin by TRUE): the S-shape view, kept for comparison.
    ax = axes[0]
    edges = np.linspace(true.min(), true.max() + 1e-6, n_bins + 1)
    idx = np.clip(np.digitize(true, edges) - 1, 0, n_bins - 1)
    cen, m, e = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() < 10:
            continue
        mean, se, _ = weighted_mean_se(mu[sel], w[sel])
        cen.append(0.5 * (edges[b] + edges[b + 1])); m.append(mean); e.append(se)
    lo, hi = min(true.min(), mu.min()), max(true.max(), mu.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.errorbar(cen, m, yerr=e, fmt='o-', ms=4, capsize=2)
    ax.set_xlabel(r"true $\log M_h$ (bin)")
    ax.set_ylabel(r"$\langle \mu \rangle$ in bin")
    ax.set_title(f"{label}\nbinned by TRUE (S-shape expected)", fontsize=9)

    # (b) Reverse binning (bin by PREDICTED mu): the actual bias test.
    ax = axes[1]
    edges = np.linspace(mu.min(), mu.max() + 1e-6, n_bins + 1)
    idx = np.clip(np.digitize(mu, edges) - 1, 0, n_bins - 1)
    cen, m, e, devs, dev_w = [], [], [], [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() < 10:
            continue
        mean_true, se, _ = weighted_mean_se(true[sel], w[sel])
        mean_mu, _, _ = weighted_mean_se(mu[sel], w[sel])
        # Compare against the bin's mean PREDICTION, not the bin centre, so
        # within-bin spread of mu can't masquerade as a bias.
        cen.append(mean_mu); m.append(mean_true); e.append(se)
        devs.append(mean_true - mean_mu); dev_w.append(w[sel].sum())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.errorbar(cen, m, yerr=e, fmt='o-', ms=4, capsize=2, color='C1')
    ax.set_xlabel(r"$\langle \mu \rangle$ in bin of predicted $\mu$")
    ax.set_ylabel(r"$\langle$ true $\log M_h \rangle$ in bin")
    ax.set_title("binned by PREDICTED (should lie on y = x)", fontsize=9)

    # (c) PIT histogram, weighted. Under perfect calibration each of the
    # n_pit bins holds probability p = 1/n_pit; the shaded band is the
    # +/-2-sigma binomial scatter of that, using n_eff.
    ax = axes[2]
    n_pit = 20
    hist, pedges = np.histogram(pit, bins=n_pit, range=(0, 1), weights=w)
    p_bin = 1.0 / n_pit
    dens = hist / p_bin
    band = 2.0 * np.sqrt(p_bin * (1 - p_bin) / n_eff_total) / p_bin
    ax.axhspan(1 - band, 1 + band, color='0.85')
    ax.axhline(1.0, color='k', lw=1, ls='--')
    ax.stairs(dens, pedges, color='C2', lw=1.5)
    ax.set_xlabel(r"PIT $= F_{\rm pred}(\mathrm{true})$")
    ax.set_ylabel("density")
    ax.set_title("PIT (flat = calibrated)", fontsize=9)

    # (d) Coverage curves, both from the PIT so they hold for mixtures too:
    # central q-interval (|PIT - 0.5| < q/2; for a single Gaussian this is
    # exactly mu +/- Phi^-1((1+q)/2) sigma) and one-sided lower q-quantile
    # (PIT < q), which exposes skew the central curve averages away.
    ax = axes[3]
    q = np.linspace(0.05, 0.95, 19)
    cov = np.array([np.sum(w * (np.abs(pit - 0.5) < qq / 2)) for qq in q])
    cov_lo = np.array([np.sum(w * (pit < qq)) for qq in q])
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.plot(q, cov, 'o-', ms=3, color='C3', label='central interval')
    ax.plot(q, cov_lo, 's-', ms=3, color='C4', label='one-sided (PIT < q)')
    ax.legend(fontsize=8)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("coverage (on diagonal = calibrated)", fontsize=9)

    # Weighted PIT ECDF vs uniform: Kolmogorov-Smirnov-style max distance.
    order = np.argsort(pit)
    ks = np.max(np.abs(np.cumsum(w[order]) - pit[order]))
    devs, dev_w = np.array(devs), np.array(dev_w)
    return dict(
        n=len(true), n_eff=n_eff_total,
        rms=np.sqrt(np.sum(w * (true - mu) ** 2)),
        cov68=np.sum(w * (np.abs(pit - 0.5) < 0.6827 / 2)),
        cov95=np.sum(w * (np.abs(pit - 0.5) < 0.9545 / 2)),
        reverse_bias_mean=np.sum(dev_w * np.abs(devs)) / dev_w.sum(),
        reverse_bias_max=np.max(np.abs(devs)),
        pit_ks=ks,
    )


def main():
    args = parse_args()

    # Shared loader (train_halo_mass.load_halo_mass_model): architecture,
    # mixture head, bright-M_UV input and the training normalization bounds.
    model, cfg = load_halo_mass_model(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Rebuild the training dataset EXACTLY as train_halo_mass.py's main() did:
    # same global NumPy seed before construction (matters only for
    # --max-per-catalog > 0 subsampling), same flags (so self.weights is the
    # weight each example actually trained with).
    np.random.seed(args.train_seed)
    dataset = HaloMassDataset(
        database_dirs=args.database_dir,
        param_min=cfg['param_min'], param_max=cfg['param_max'],
        max_per_catalog=args.max_per_catalog,
        summary_mode=cfg['summary_mode'], only_angular=cfg['only_angular'],
        use_bright_muv=cfg['use_bright_muv'],
        max_neighbors=cfg['max_neighbors'],
        weight_by_catalog_count=args.weight_by_catalog_count,
        reweight_by_mass=args.reweight_by_mass,
        reweight_alpha=args.reweight_alpha,
        reweight_bins=args.reweight_bins,
        lds_kernel_sigma=args.lds_kernel_sigma,
        uvlf_reweight=args.uvlf_reweight,
        uvlf_sigma_scale=args.uvlf_sigma_scale,
        max_weight_ratio=args.max_weight_ratio,
    )
    # No random-shift augmentation at evaluation time (training applied it
    # to val too, since the split shares one dataset instance; we don't).
    dataset.augment = False

    n_val = int(len(dataset) * args.val_frac)
    n_train = len(dataset) - n_val
    _, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.train_seed))
    val_idx = np.array(val_set.indices)
    log.info(f"Dataset size {len(dataset)}  ->  Train: {n_train}  Val: {n_val}  "
             f"(must match the training log's 'Train: ... Val: ...' line)")

    if args.n_examples > 0 and args.n_examples < len(val_idx):
        val_idx = np.random.default_rng(args.seed).choice(val_idx, args.n_examples, replace=False)
        log.info(f"Subsampled {len(val_idx)} val examples (--seed {args.seed}).")

    # Inference over the held-out examples, batched.
    loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)
    true, mu, sigma, pit, w = [], [], [], [], []
    with torch.no_grad():
        for x, target, weight in loader:
            out = mixture_outputs(model, x.to(device))
            m, sd = mixture_mean_std(*out)  # predictive mean/std (= mu, sigma if K = 1)
            mu.append(m.cpu().numpy()); sigma.append(sd.cpu().numpy())
            pit.append(mixture_cdf(*out, target.to(device)).cpu().numpy())
            true.append(target.numpy()); w.append(weight.numpy())
    true, mu, sigma, pit, w = (np.concatenate(a).astype(np.float64) for a in (true, mu, sigma, pit, w))
    params = np.array([dataset.params[i] for i in val_idx])

    log.info("Computing UVLF consistency of each val example's theta ...")
    uvlf_sig = uvlf_sigma_per_theta(params)
    consistent = uvlf_sig <= args.max_sigma
    log.info(f"{consistent.sum()}/{len(consistent)} val examples pass UVLF <= {args.max_sigma} sigma "
             f"({len(np.unique(params[consistent], axis=0))} distinct theta)")

    # Two populations; drop zero-weight examples from the weighted one
    # (they carry no information about the training distribution).
    rows = [("val, training-weighted", w > 0, w),
            (f"val, UVLF $\\leq${args.max_sigma}$\\sigma$, unweighted", consistent, np.ones_like(w))]
    fig, axes = plt.subplots(len(rows), 4, figsize=(17, 4.3 * len(rows)))
    for r, (label, sel, wt) in enumerate(rows):
        if sel.sum() < 50:
            log.warning(f"'{label}': only {sel.sum()} examples, skipping.")
            continue
        s = calibration_row(axes[r], true[sel], mu[sel], pit[sel], wt[sel], args.n_bins, label)
        mean_sigma = np.sum(wt[sel] * sigma[sel]) / wt[sel].sum()
        log.info(f"[{label}]  n={s['n']}  n_eff={s['n_eff']:.0f}  RMS={s['rms']:.3f}  "
                 f"<sigma>={mean_sigma:.3f}  cov68={s['cov68']:.1%}  cov95={s['cov95']:.1%}")
        log.info(f"    reverse-binning |<true>-<mu>|: weighted mean {s['reverse_bias_mean']:.3f} dex, "
                 f"max {s['reverse_bias_max']:.3f} dex   PIT KS distance {s['pit_ks']:.3f}")
    fig.suptitle(f"Held-out calibration: {args.model_dir.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output)
    log.info(f"Saved: {args.output}")

    # Raw per-example results, so further checks don't need another inference pass.
    npz_path = args.output.with_suffix(".npz")
    np.savez(npz_path, true_logmh=true, mu=mu, sigma=sigma, pit=pit, train_weight=w,
             params=params, uvlf_sigma=uvlf_sig, dataset_index=val_idx)
    log.info(f"Saved: {npz_path}")


if __name__ == "__main__":
    main()
