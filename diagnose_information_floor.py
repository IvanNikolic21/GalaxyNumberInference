#!/usr/bin/env python
"""
diagnose_information_floor.py
-----------------------------
How much can ANY estimator learn about a bright galaxy's halo mass from a
given set of observables? A model-free estimate of the information floor,
to compare against the trained network (v11: RMS 0.243 dex on held-out,
UVLF-consistent theta).

Motivation
----------
v11 added the bright galaxy's own M_UV as an input and improved the
held-out RMS only from 0.262 to 0.243 dex, while diagnose_heldout_
calibration.py showed the network is a well-calibrated posterior mean. That
leaves two explanations: (a) the observables simply don't constrain log Mh
better than ~0.24 dex (an information floor), or (b) the network fails to
extract information that is there. This script decides between them
without any neural network, by fitting simple, transparent regressions of
true log Mh on different feature sets and measuring their out-of-sample RMS.

The minimum-RMS predictor of log Mh given features x is the conditional
mean E[log Mh | x] (e.g. Hastie, Tibshirani & Friedman 2009, "The Elements
of Statistical Learning", Sec. 2.4), and its RMS is the floor. Any fitted
estimator can only do WORSE than that, so every number printed here is an
UPPER BOUND on the floor for that feature set:
  * if the network's RMS is already at or below these bounds, the network is
    not the bottleneck -- the observables are;
  * if a simple regression clearly BEATS the network on the same features,
    the network is failing to use them.
Comparing feature sets measures what each observable adds -- e.g.
theta + M_UV vs. theta + M_UV + environment quantifies the information the
environment carries beyond the galaxy's own luminosity.

Two estimators are fitted per feature set:
  * k-nearest-neighbour regression (mean log Mh of the k nearest training
    points in standardized feature space; Hastie et al. 2009, Sec. 2.3.2).
    Nonparametric, but degrades as the number of features grows (curse of
    dimensionality), so the bound is loosest for the large feature sets.
  * quadratic least squares (all linear + pairwise + squared terms, with a
    small ridge penalty; Hastie et al. 2009, Sec. 3.4.1). Copes with many
    features but cannot capture sharper structure.
Report the smaller of the two as the bound.

Cross-validation by BOX
-----------------------
The same halo is "the bright galaxy" under many nearby theta in the same box
(the vertical stripes in the predicted-vs-true plots), with an almost
identical environment each time. A random train/test split would let the
regression find a near-copy of each test example and report a falsely low
floor. Instead every fold holds out one whole box (--database-dir entry) and
fits on the others; the boxes are independent realizations, so nothing
leaks. (The network's own random_split does NOT do this, so its held-out
RMS may be slightly optimistic by comparison.)

Environment summaries (PROJECTED, i.e. dx, dy only, matching --only-angular
and real imaging data -- or 3D with --use-dz, see below):
  n_tot        number of stored faint neighbours (capped at 200 by
               build_nre_database.py's --max-env-neighbors -- the fraction of
               environments at the cap is printed, since the cap discards
               information for the richest fields)
  n(<R)        neighbour counts within projected R = 1, 2, 4 cMpc
  d_k          projected distance to the k = 1, 3, 10 nearest neighbours
               (k-NN density estimates; e.g. Dressler 1980, ApJ 236, 351)
  M_UV,brightest  magnitude of the brightest faint neighbour
  log L(<2 Mpc)   log summed neighbour UV luminosity within 2 cMpc

Forecast mode: run on databases built deeper (build_nre_database.py
--faint-limit, with --max-uvlf-sigma so only the relevant theta are built,
and their bright galaxies are the SAME as in the -18.5 database) and with
--use-dz, to measure how the floor improves with survey depth and with
line-of-sight information -- "what data would it take to weigh the halo".
NOTE the distance-based summaries (d_k, n(<R)) mean different things at
different depths; compare floors across depths, not individual features.

Population: every environment whose theta passes the hard UVLF cut
(<= --max-sigma, same conversion as plot_eval_vs_truth_uvlf_consistent.py),
equally weighted -- the same population as the bottom row of
diagnose_heldout_calibration.py, so the numbers are directly comparable.

Usage
-----
    python diagnose_information_floor.py \\
        --database-dir /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed1955 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed2027 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed3142 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed4242 \\
        --output /groups/astro/ivannik/projects/Neighbors/information_floor.npz
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RADII_MPC = (1.0, 2.0, 4.0)        # projected count-in-cylinder radii
KNN_ORDERS = (1, 3, 10)            # k for the k-th-nearest-neighbour distances
FLUX_RADIUS_MPC = 2.0              # radius for the summed neighbour luminosity


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database-dir", type=Path, nargs='+', required=True,
                   help="One directory per independent box; each is held out in turn.")
    p.add_argument("--use-dz", action="store_true",
                   help="Use 3D distances (dx, dy, dz) for every environment summary instead of "
                        "projected ones -- an upper bound on what perfect line-of-sight "
                        "information (e.g. spectroscopic redshifts of every neighbour, no "
                        "redshift-space distortions: halo positions are in real space) would add.")
    p.add_argument("--faint-limit", type=float, default=-18.5,
                   help="Faint limit the databases were built with (build_nre_database.py "
                        "--faint-limit); sets the 'no neighbour' magnitude and the luminosity zero point.")
    p.add_argument("--env-cap", type=int, default=200,
                   help="Neighbour storage cap the databases were built with (--max-env-neighbors).")
    p.add_argument("--max-sigma", type=float, default=5.0,
                   help="Hard UVLF cut (equivalent Gaussian sigma) on theta.")
    p.add_argument("--knn-k", type=int, nargs='+', default=[25, 100],
                   help="k values for k-NN regression (results for each are printed).")
    p.add_argument("--ridge", type=float, default=1e-3,
                   help="Ridge penalty for the quadratic fit (in standardized units).")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional .npz with the per-environment features, so further "
                        "feature sets can be tried without re-reading the databases.")
    return p.parse_args()


def env_summaries(env, faint_limit=-18.5, use_dz=False):
    """Environment summaries for one environment (N, 4) = (dx, dy, dz, MUV):
    projected distances by default, 3D ones with use_dz."""
    if len(env) == 0:
        d = np.array([])
        muv = np.array([])
    else:
        d = np.linalg.norm(env[:, :3], axis=1) if use_dz else np.hypot(env[:, 0], env[:, 1])
        order = np.argsort(d)
        d, muv = d[order], env[order, 3]
    # A missing k-th neighbour gets a distance just beyond the farthest one
    # stored (or 20 cMpc if there are none) -- a monotone "very sparse" value
    # rather than NaN, so it can be standardized like every other feature.
    far = (d[-1] * 1.5) if len(d) else 20.0
    dk = [d[k - 1] if len(d) >= k else far for k in KNN_ORDERS]
    counts = [np.sum(d < r) for r in RADII_MPC]
    brightest = muv.min() if len(muv) else faint_limit
    flux = np.sum(10 ** (-0.4 * (muv[d < FLUX_RADIUS_MPC] - faint_limit)))
    return [len(env), *counts, *dk, brightest, np.log10(1.0 + flux)]


ENV_NAMES = (["n_tot"] + [f"n(<{r:g})" for r in RADII_MPC] + [f"d_{k}" for k in KNN_ORDERS]
             + ["M_UV,brightest", "log L(<2)"])


def load(args):
    # Pass 1: read only each file's theta (cheap), so the UVLF cut can be
    # computed in one batch -- uvlf_sigma_per_theta builds the Mason15 model
    # once per call.
    files, box_of_file, thetas = [], [], []
    for b, db_dir in enumerate(args.database_dir):
        for path in sorted(Path(db_dir).glob("nre_*.npz")):
            try:
                thetas.append(np.load(path)['params'])
            except Exception:
                continue
            files.append(path)
            box_of_file.append(b)
    # Imported here, not at module level, so plot_information_floor_comparison.py
    # can reuse this module's regression helpers without the uvlf/hmf stack.
    from uvlf_consistency import uvlf_sigma_per_theta
    sig = uvlf_sigma_per_theta(np.array(thetas))
    keep = sig <= args.max_sigma
    log.info(f"{keep.sum()}/{len(files)} files pass UVLF <= {args.max_sigma} sigma")

    # Pass 2: per-environment features for the UVLF-consistent files only.
    y, theta, muv, envf, box = [], [], [], [], []
    n_skipped = 0
    for path, b, th, ok in zip(files, box_of_file, thetas, keep):
        if not ok:
            continue
        d = np.load(path)
        if 'bright_logmh' not in d.files or 'bright_muv' not in d.files:
            n_skipped += 1
            continue
        coords, offsets = d['coords'], d['offsets']
        for i in range(len(offsets) - 1):
            env = coords[offsets[i]:offsets[i + 1]]
            if len(env) == 0:
                continue  # same rule as HaloMassDataset
            y.append(d['bright_logmh'][i]); muv.append(d['bright_muv'][i])
            theta.append(th); envf.append(env_summaries(env, args.faint_limit, args.use_dz)); box.append(b)
    if n_skipped:
        log.warning(f"{n_skipped} UVLF-consistent files lack bright_logmh/bright_muv -- skipped")
    return (np.array(y, float), np.array(theta, float), np.array(muv, float),
            np.array(envf, float), np.array(box))


def standardize(X, train):
    m, s = X[train].mean(0), X[train].std(0)
    return (X - m) / np.where(s > 0, s, 1.0)


def quad_design(X):
    # [1, x_i, x_i * x_j (i <= j)] -- every linear, squared and pairwise term.
    n, p = X.shape
    iu = np.triu_indices(p)
    return np.hstack([np.ones((n, 1)), X, (X[:, :, None] * X[:, None, :])[:, iu[0], iu[1]]])


def cv_rms(X, y, box, method, k=None, ridge=1e-3):
    """Out-of-sample RMS, holding out one box at a time; returns pooled RMS."""
    resid = np.empty_like(y)
    for b in np.unique(box):
        test, train = box == b, box != b
        Z = standardize(X, train)
        if method == "knn":
            _, idx = cKDTree(Z[train]).query(Z[test], k=k)
            resid[test] = y[test] - y[train][idx].mean(axis=1)
        else:
            A = quad_design(Z)
            At = A[train]
            coef = np.linalg.solve(At.T @ At + ridge * len(At) * np.eye(A.shape[1]), At.T @ y[train])
            resid[test] = y[test] - A[test] @ coef
    return np.sqrt(np.mean(resid ** 2))


def main():
    args = parse_args()
    y, theta, muv, envf, box = load(args)
    log.info(f"{len(y)} UVLF-consistent environments across {len(np.unique(box))} boxes "
             f"(per box: {np.bincount(box).tolist()})")
    log.info(f"std of true log Mh (the 'theta-free' prior width): {y.std():.3f} dex; "
             f"environments at the {args.env_cap}-neighbour storage cap: {np.mean(envf[:, 0] >= args.env_cap):.1%}")
    log.info(f"Distances: {'3D (with dz)' if args.use_dz else 'projected'};  faint limit {args.faint_limit}")

    feature_sets = {
        "theta":                   theta,
        "theta + M_UV":            np.column_stack([theta, muv]),
        "theta + n_tot":           np.column_stack([theta, envf[:, :1]]),
        "theta + M_UV + n_tot":    np.column_stack([theta, muv, envf[:, :1]]),
        "theta + env":             np.column_stack([theta, envf]),
        "theta + M_UV + env":      np.column_stack([theta, muv, envf]),
    }
    log.info("Out-of-sample RMS of true log Mh, cross-validated by box "
             "(upper bounds on the information floor; network v11: 0.243 dex):")
    results = {}
    for name, X in feature_sets.items():
        r = {f"knn{k}": cv_rms(X, y, box, "knn", k=k) for k in args.knn_k}
        r["quad"] = cv_rms(X, y, box, "quad", ridge=args.ridge)
        best = min(r.values())
        results[name] = best
        log.info(f"  {name:<24s} ({X.shape[1]:2d} features)  "
                 + "  ".join(f"{m}={v:.3f}" for m, v in r.items())
                 + f"   -> bound {best:.3f}  (variance explained {1 - best**2 / y.var():.0%})")

    if args.output is not None:
        np.savez(args.output, true_logmh=y, theta=theta, bright_muv=muv, env_features=envf,
                 env_feature_names=np.array(ENV_NAMES), box=box)
        log.info(f"Saved features: {args.output}")


if __name__ == "__main__":
    main()
