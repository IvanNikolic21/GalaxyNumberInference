#!/usr/bin/env python
"""
plot_information_floor_comparison.py
------------------------------------
Visual report of the information-floor test (diagnose_information_floor.py)
against the trained halo-mass network (train_halo_mass.py, v11): predicted
vs true log Mh of the bright galaxy for every method, on the SAME galaxies.

Methods compared (all predict E[log Mh | x], the minimum-RMS estimate;
Hastie, Tibshirani & Friedman 2009, Sec. 2.4):
  * k-NN regression       mean of the k nearest training points in
                          standardized feature space (Hastie+09, Sec. 2.3.2)
  * quadratic ridge       all linear + squared + pairwise terms (Sec. 3.4.1)
  * gradient boosting     histogram gradient-boosted trees (Friedman 2001,
                          Ann. Stat. 29, 1189; scikit-learn's
                          HistGradientBoostingRegressor, after LightGBM, Ke+
                          2017) -- the most flexible of the three, so the
                          tightest model-free upper bound on the floor
  * network               v11's posterior mean (mixture mean)
The three regressions use the floor script's features: theta + the bright
galaxy's own M_UV + 9 projected environment summaries. They are fitted
with cross-validation BY BOX (one box held out per fold), so no halo is
ever in both the fit and the test set. The network was trained on its own
80/20 random_split; only its validation examples are plotted, and the
regressions are scored on exactly those examples too.

Matching network <-> floor rows
-------------------------------
The held-out npz (diagnose_heldout_calibration.py) stores theta and true
log Mh per validation example but not the box; the floor npz stores both.
A row is matched on (theta, log Mh) rounded to 1e-6: halo masses are
continuous (not particle-quantized) at that precision, so a match is the
same galaxy. Equal keys (rare) are paired off one-to-one, never reused.
The plotted population is therefore: theta passing the FLOOR script's UVLF
cut AND in the network's validation split (~20% of the floor rows).

Why one panel shows both binnings
---------------------------------
For a posterior-mean estimator mu, E[true | mu] = mu (calibration), while
E[mu | true] is pulled toward the population mean wherever the data are not
fully informative (regression to the mean; e.g. Stigler 1997, Stat. Methods
Med. Res. 6, 103). Panel (a) draws both on the same axes:
  * lines + bands: median and 16-84% of mu in bins of TRUE log Mh -- the
    S-shape; how flat it is shows how little information there is;
  * markers: mean TRUE log Mh in bins of mu, plotted at (<true>, mu) --
    for an unbiased estimator they sit on y = x.
So "S-shaped lines, diagonal markers" is the signature of an unbiased but
information-limited estimator, and any method whose markers leave the
diagonal is biased. Panel (b) shows where along the mass range each method
gains, as the RMS of (mu - true) in bins of true log Mh.

Usage
-----
    python plot_information_floor_comparison.py \\
        --floor-npz information_floor/floor_faint18p5_projected.npz \\
        --network-npz heldout_calibration_v11.npz \\
        --output information_floor_comparison_v11.pdf
"""
import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Reuse the floor script's exact preprocessing, so the regressions here are
# the ones whose RMS diagnose_information_floor.py prints.
from diagnose_information_floor import standardize, quad_design

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# One colour per method; the network in black, so it reads as "the result"
# and the regressions as references.
STYLE = {
    "knn":     dict(label="$k$-NN ($k$=100)",    color="#1f77b4", marker="s"),
    "quad":    dict(label="quadratic ridge",     color="#2ca02c", marker="^"),
    "gbt":     dict(label="gradient boosting",   color="#d62728", marker="D"),
    "network": dict(label="network (v11)",       color="black",   marker="o"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--floor-npz", type=Path, required=True,
                   help="Per-environment features saved by diagnose_information_floor.py --output.")
    p.add_argument("--network-npz", type=Path, required=True,
                   help="Per-example predictions saved by diagnose_heldout_calibration.py.")
    p.add_argument("--output", type=Path, default=Path("information_floor_comparison.pdf"))
    p.add_argument("--knn-k", type=int, default=100)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--no-gbt", action="store_true",
                   help="Skip gradient boosting (e.g. where scikit-learn is not installed).")
    p.add_argument("--bin-width", type=float, default=0.15, help="dex, for both binnings.")
    p.add_argument("--min-per-bin", type=int, default=30)
    return p.parse_args()


def cv_predictions(X, y, box, knn_k, ridge, use_gbt):
    """Out-of-sample predictions of every regression, one box held out per fold."""
    preds = {"knn": np.empty_like(y), "quad": np.empty_like(y)}
    if use_gbt:
        from sklearn.ensemble import HistGradientBoostingRegressor
        preds["gbt"] = np.empty_like(y)
    for b in np.unique(box):
        test, train = box == b, box != b
        Z = standardize(X, train)
        _, idx = cKDTree(Z[train]).query(Z[test], k=knn_k, workers=-1)
        preds["knn"][test] = y[train][idx].mean(axis=1)
        A = quad_design(Z)
        At = A[train]
        coef = np.linalg.solve(At.T @ At + ridge * len(At) * np.eye(A.shape[1]), At.T @ y[train])
        preds["quad"][test] = A[test] @ coef
        if use_gbt:
            # Trees are invariant to monotone feature rescaling -> raw X.
            # Hyperparameters fixed (not tuned on the test box); early
            # stopping uses an internal 10% split of the TRAINING boxes only.
            gbt = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                                max_leaf_nodes=63, random_state=0)
            preds["gbt"][test] = gbt.fit(X[train], y[train]).predict(X[test])
        log.info(f"  fold: box {b} held out ({test.sum()} environments)")
    return preds


def match_network(floor, net):
    """Network prediction per floor row (NaN where the row is not in the network's val split)."""
    def keys(theta, logmh):
        return map(tuple, np.round(np.column_stack([theta, logmh]), 6))
    rows = defaultdict(list)
    for i, k in enumerate(keys(floor["theta"], floor["true_logmh"])):
        rows[k].append(i)
    mu = np.full(len(floor["true_logmh"]), np.nan)
    for j, k in enumerate(keys(net["params"], net["true_logmh"])):
        if rows.get(k):                       # pop: each floor row used at most once
            mu[rows[k].pop()] = net["mu"][j]
    return mu


def binned(x, v, edges, min_n, stat):
    """stat(v) in bins of x; NaN where a bin has fewer than min_n points."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        out.append(stat(v[s]) if s.sum() >= min_n else np.full(np.shape(stat(np.zeros(1))), np.nan))
    return np.array(out)


def main():
    args = parse_args()
    floor = np.load(args.floor_npz)
    net = np.load(args.network_npz)
    y, box = floor["true_logmh"], floor["box"]
    X = np.column_stack([floor["theta"], floor["bright_muv"], floor["env_features"]])

    mu_net = match_network(floor, net)
    val = ~np.isnan(mu_net)
    log.info(f"{val.sum()} of {len(y)} floor environments are in the network's validation split "
             f"({val.mean():.1%}; expect ~20% for an 80/20 split)")
    # The network's 0.243 dex (diagnose_heldout_calibration.py) is quoted on
    # that script's own UVLF<=5sigma selection; print it next to the matched
    # value, since the two selections were found to differ (2026-10-23).
    own = net["uvlf_sigma"] <= 5.0
    log.info(f"network RMS on diagnose_heldout_calibration's own UVLF<=5sigma val rows "
             f"(n={own.sum()}): {np.sqrt(np.mean((net['mu'][own] - net['true_logmh'][own]) ** 2)):.3f}")

    log.info("Cross-validated regressions (by box) on theta + M_UV + 9 env summaries:")
    preds = cv_predictions(X, y, box, args.knn_k, args.ridge, use_gbt=not args.no_gbt)
    preds["network"] = mu_net

    yv = y[val]
    lo = np.floor(np.percentile(yv, 0.5) / args.bin_width) * args.bin_width
    hi = np.ceil(np.percentile(yv, 99.5) / args.bin_width) * args.bin_width
    edges = np.arange(lo, hi + args.bin_width / 2, args.bin_width)
    centers = 0.5 * (edges[1:] + edges[:-1])

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw=dict(width_ratios=[1.25, 1]))

    # Background: where the population actually lives (the network's mu vs
    # true), so the eye weighs each bin by how many galaxies it holds.
    ax.hist2d(yv, mu_net[val], bins=[np.linspace(lo, hi, 60)] * 2, cmap="Greys", cmin=1,
              norm="log", alpha=0.35, rasterized=True)
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.8, ls=":")

    rms_total = {}
    for name, st in STYLE.items():
        if name not in preds:
            continue
        mu = preds[name][val]
        rms_total[name] = np.sqrt(np.mean((mu - yv) ** 2))
        lab = f"{st['label']}: RMS {rms_total[name]:.3f} dex"
        # (1) E[mu | true]: median + 16-84% of mu in bins of true -> the S-shape.
        q = binned(yv, mu, edges, args.min_per_bin, lambda v: np.percentile(v, [16, 50, 84]))
        ax.plot(centers, q[:, 1], color=st["color"], lw=1.8, label=lab)
        ax.fill_between(centers, q[:, 0], q[:, 2], color=st["color"], alpha=0.10, lw=0)
        # (2) E[true | mu]: mean true in bins of mu, drawn at (<true>, mu bin
        # centre) -> on y = x if the method is unbiased.
        mt = binned(mu, yv, edges, args.min_per_bin, np.mean)
        ax.plot(mt, centers, ls="none", marker=st["marker"], ms=5, mfc="none", mec=st["color"], mew=1.2)
        # Panel (b): conditional RMS in bins of true log Mh.
        r = binned(yv, mu - yv, edges, args.min_per_bin, lambda v: np.sqrt(np.mean(v ** 2)))
        axr.plot(centers, r, color=st["color"], lw=1.8, marker=st["marker"], ms=4, mfc="none",
                 label=st["label"])

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"true $\log_{10} M_{\rm h}$ of the bright galaxy")
    ax.set_ylabel(r"predicted $\log_{10} M_{\rm h}$")
    ax.set_title("(a) lines: median & 16-84% of prediction at fixed TRUE mass\n"
                 "     markers: mean true mass at fixed PREDICTION (unbiased = on diagonal)",
                 fontsize=9, loc="left")
    ax.legend(fontsize=8, loc="upper left", frameon=False)

    # Panel (b) background: the true-mass distribution of the plotted galaxies.
    axh = axr.twinx()
    axh.hist(yv, bins=edges, color="0.85", zorder=0)
    axh.set_yticks([]); axh.set_ylabel("")
    axr.set_zorder(axh.get_zorder() + 1); axr.patch.set_visible(False)
    axr.axhline(np.std(yv), color="0.4", ls=":", lw=0.8)
    axr.text(edges[0] + 0.05, np.std(yv) * 1.02, "prior width (std of true mass)", fontsize=7, color="0.4")
    axr.set_xlim(lo, hi); axr.set_ylim(0, None)
    axr.set_xlabel(r"true $\log_{10} M_{\rm h}$")
    axr.set_ylabel(r"RMS(predicted $-$ true) [dex]")
    axr.set_title("(b) scatter at fixed true mass (grey: number of galaxies)", fontsize=9, loc="left")
    axr.legend(fontsize=8, frameon=False)

    fig.suptitle(f"Same {val.sum()} held-out galaxies, UVLF-consistent $\\theta$ "
                 f"(regressions: theta + own $M_{{UV}}$ + env summaries, CV by box)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    log.info(f"Saved: {args.output}")
    for name, r in rms_total.items():
        log.info(f"  {STYLE[name]['label']:<20s} RMS {r:.3f} dex")


if __name__ == "__main__":
    main()
