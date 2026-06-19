

#!/usr/bin/env python
"""
run_cosmic_variance.py
-----------------------
Estimate the cosmic (field-to-field) variance of galaxy number counts in
NIRCam-sized pointings, for the fiducial and stochastic models, built from
the single z=10.5 coeval box.

Two assumed lightcone depths are evaluated, both centered such that the box
is treated as representative of the whole range:
    z9p2_10p9 :  9.2 < z < 10.9   (depth ~368 cMpc, fits inside the box)
    z8p6_11p3 :  8.6 < z < 11.3   (depth ~596 cMpc, truncated to one box
                                    length — see cosmic_variance.effective_los_depth)

Usage
-----
    python run_cosmic_variance.py
    python run_cosmic_variance.py --n-realizations 50 --n-trials 5000
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from galaxy_neighbors import load_halo_catalog
from cosmic_variance import (
    PointingConfig,
    comoving_depth_mpc,
    footprint_side_mpc,
    effective_los_depth,
    build_pointing_pool,
    bootstrap_group_stats,
    bootstrap_fractional_cosmic_variance,
    summarize_group_stats,
    pool_moments,
    summarize_pool_moments,
    plot_fractional_cosmic_variance,
    save_cosmic_variance,
    load_cosmic_variance,
    GammaCVConfig,
    fit_sigma_cv_mcmc,
    summarize_gamma_fits,
    gamma_fit_to_reference,
)
from run_analysis import REDSHIFT_CONFIGS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redshift ranges to evaluate — both lightcones are built from the z=10.5 box
# ---------------------------------------------------------------------------
BOX_REDSHIFT = 10.5

Z_RANGES = {
    "z9p2_10p9": (9.2, 10.9),
    "z8p6_11p3": (8.6, 11.3),
}

# Extra model: a UniverseMachine-matched ("pre-JWST") MUV catalog, built on
# the same z=10.5 halo positions as fiducial/stochastic, but with much lower
# abundance per M_UV threshold (~30-40x lower mean count/pointing than
# fiducial/stochastic).
PREJWST_MODEL = "prejwst"
PREJWST_MUV_PATH = Path("/lustre/astro/ivannik/catalog_preJWST_30.h5")
PREJWST_N_REALIZATIONS = 30

# All three models are plotted using a Gamma/NB MCMC fit
# (cosmic_variance.fit_sigma_cv_mcmc) rather than the naive bootstrap median.
# We switched fiducial/stochastic over too after direct comparison showed
# the bootstrap median collapsing to ~0 at rare thresholds (M_UV<-20.5) where
# the true (full-pool, method-of-moments) sigma_CV is ~0.7 -- the exact
# small-N zero-count pathology bootstrap_fractional_cosmic_variance's
# docstring warns about. Using the same large-sample Gamma/NB fit for every
# model keeps the comparison apples-to-apples. This is computed fresh every
# run rather than cached (cheap relative to catalog loading) and is
# unaffected by --skip-gamma-fit, which only toggles a diagnostic print.
GAMMA_PLOTTED_MODELS = {"fiducial", "stochastic", PREJWST_MODEL}

CACHE_DIR = Path("/groups/astro/ivannik/projects/Neighbors/cache/cosmic_variance")
OUTPUT_DIR = Path("/groups/astro/ivannik/projects/Neighbors/cosmic_variance_plots")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Cosmic variance of NIRCam pointings (fiducial vs stochastic)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n-realizations", type=int, default=100,
        help="Number of MUV realizations to load per model.",
    )
    p.add_argument(
        "--n-trials", type=int, default=2000,
        help="Number of bootstrap trials (surveys of --group-size pointings).",
    )
    p.add_argument(
        "--group-size", type=int, default=28,
        help="Number of pointings per bootstrap survey draw.",
    )
    p.add_argument(
        "--fov-area-arcmin2", type=float, default=PointingConfig().fov_area_arcmin2,
        help="On-sky area of one pointing [arcmin^2]. Default: NIRCam's two "
             "modules combined (~9.7). Use ~4.84 for a single module.",
    )
    p.add_argument(
        "--force-recompute", action="store_true",
        help="Ignore existing cache and recompute from scratch.",
    )
    p.add_argument(
        "--gamma-fit-group-size", type=int, default=28,
        help="Number of pointings in the realistic-survey-sized sample used "
             "for the Gamma/Negative-Binomial MCMC sigma_CV fit (Weibel et al. "
             "2025 Section 2.4.2 method). A second, larger-sample fit "
             "(--gamma-fit-max-fullpool-size) is always also run, as a "
             "precision cross-check.",
    )
    p.add_argument(
        "--gamma-fit-max-fullpool-size", type=int, default=5000,
        help="Cap on the precision-cross-check sample size for the Gamma/NB "
             "fit (randomly subsampled from the pool if larger). Runtime "
             "scales with this number, so keep it in the low thousands.",
    )
    p.add_argument(
        "--prejwst-gamma-sample-size", type=int, default=20000,
        help="Sample size for the pre-JWST model's plotted Gamma/NB fit "
             "(capped at its pool size). Its mean count per pointing is far "
             "lower than fiducial/stochastic, so the NB likelihood carries "
             "much less information per pointing — it needs a much bigger "
             "sample than --group-size to converge (5000 pointings already "
             "matches the full-pool estimate well for fiducial's abundance; "
             "pre-JWST's abundance is ~30-40x lower, hence the bigger "
             "default here). Increase further (up to its full pool size) "
             "for max precision at the cost of runtime.",
    )
    p.add_argument(
        "--skip-gamma-fit", action="store_true",
        help="Skip printing the diagnostic 28-pointing single-draw Gamma/NB "
             "fit table for fiducial/stochastic. Does NOT skip the actual "
             "computation: the 5000-pointing precision-check fit always runs "
             "and is what's plotted for ALL THREE models now (the bootstrap "
             "median was shown to collapse to ~0 at rare thresholds like "
             "M_UV<-20.5 -- see the module-level comment near GAMMA_PLOTTED_MODELS).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_gamma_fits(
    pool, cfg: PointingConfig, gamma_group_size: int, seed: int = 0, max_fullpool_size: int = 5000,
) -> tuple[dict, dict]:
    """Gamma/NB MCMC fit on (a) one realistic-survey-sized draw, (b) a large precision-check sample.

    The NB log-likelihood is evaluated over the full sample at every one of
    `n_walkers * n_steps` MCMC steps, so runtime scales with sample size —
    using the literal full pool (which can be ~1e5-1e6 pointings) would take
    hours. `max_fullpool_size` caps the "precision cross-check" sample via a
    random subsample, which is already far more than needed for a tight
    posterior (a few thousand points is enough to pin down sigma_CV to ~1%,
    per the n=1000 vs n=28 comparison already verified in development).

    Returns
    -------
    fits_subsample, fits_fullpool : dict
        threshold -> result dict from `fit_sigma_cv_mcmc`, for the
        gamma_group_size-sized draw and the (possibly subsampled) larger
        precision-check sample respectively.
    """
    rng = np.random.default_rng(seed)
    n_pool = pool.shape[0]
    idx = rng.choice(n_pool, size=min(gamma_group_size, n_pool), replace=False)
    subsample = pool[idx]

    fullpool_idx = rng.choice(n_pool, size=min(max_fullpool_size, n_pool), replace=False)
    fullpool_sample = pool[fullpool_idx]

    gamma_cfg = GammaCVConfig()
    fits_subsample = {}
    fits_fullpool = {}
    for k, threshold in enumerate(cfg.thresholds):
        fits_subsample[threshold] = fit_sigma_cv_mcmc(subsample[:, k], gamma_cfg)
        fits_fullpool[threshold] = fit_sigma_cv_mcmc(fullpool_sample[:, k], gamma_cfg)
    return fits_subsample, fits_fullpool


def _print_summaries(results: dict, cfg: PointingConfig) -> None:
    for model, by_zrange in results.items():
        for zrange_label, (means, varis, mean, mean_sq, sigma_cv2_trials) in by_zrange.items():
            zlo, zhi = Z_RANGES[zrange_label]
            print(f"\n{'=' * 78}\nmodel={model}  z-range=({zlo}, {zhi})  [{zrange_label}]")
            print(summarize_group_stats(means, varis, cfg.thresholds))
            print(f"\n  Per-pointing moments & Poisson/cosmic variance decomposition:")
            print(summarize_pool_moments(mean, mean_sq, cfg.thresholds, group_size=cfg.group_size))


def _save_plot(
    results: dict, cfg: PointingConfig, plot_path: Path, reference: dict | None = None,
) -> None:
    fig = plot_fractional_cosmic_variance(results, cfg.thresholds, Z_RANGES, reference=reference)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    log.info(f"Saved plot: {plot_path}")


def _cache_is_complete(results: dict, expected_models: set[str]) -> bool:
    """Reject caches written by an older schema (missing slots -> None) or a
    partial run (missing model/z-range keys), so a stale .npz triggers a
    recompute instead of crashing deep inside summarize_pool_moments."""
    if set(results.keys()) != expected_models:
        return False
    for by_zrange in results.values():
        if set(by_zrange.keys()) != set(Z_RANGES.keys()):
            return False
        for values in by_zrange.values():
            if any(v is None for v in values):
                return False
    return True


def main():
    args = parse_args()
    z_cfg = REDSHIFT_CONFIGS[BOX_REDSHIFT]

    cfg = PointingConfig(
        group_size=args.group_size,
        n_trials=args.n_trials,
        fov_area_arcmin2=args.fov_area_arcmin2,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fov_tag = f"fov{args.fov_area_arcmin2:g}"
    pj_tag = f"pj-{PREJWST_MUV_PATH.stem}-real{PREJWST_N_REALIZATIONS}"
    # Bumped when the meaning of cfg.thresholds changes (e.g. cumulative ->
    # differential bins) so old caches with a different counting convention
    # are never silently reused -- they'd pass _cache_is_complete's shape/key
    # checks despite meaning something different.
    bins_tag = "diffbins"
    cache_path = CACHE_DIR / f"cosmic_variance_real{args.n_realizations}_trials{args.n_trials}_g{args.group_size}_{fov_tag}_{pj_tag}_{bins_tag}.npz"
    plot_path = OUTPUT_DIR / f"sigma_cv_vs_Muv_real{args.n_realizations}_trials{args.n_trials}_g{args.group_size}_{fov_tag}_{pj_tag}_{bins_tag}.pdf"

    expected_models = {"fiducial", "stochastic", PREJWST_MODEL}

    results = None
    if cache_path.exists() and not args.force_recompute:
        log.info(f"Cache found, loading: {cache_path}")
        loaded = load_cosmic_variance(cache_path)
        if _cache_is_complete(loaded, expected_models):
            results = loaded
        else:
            log.warning(
                f"Cache at {cache_path} is incomplete or from an older script version "
                "(missing fields) — recomputing from scratch."
            )

    recompute_bootstrap = results is None
    if results is None:
        results = {"fiducial": {}, "stochastic": {}, PREJWST_MODEL: {}}
    else:
        log.info(
            "Cache has the bootstrap/moments results already — rebuilding pools "
            "anyway, since every model's plotted point always needs a fresh "
            "Gamma/NB fit (not cached, see GAMMA_PLOTTED_MODELS comment above)."
        )

    log.info(f"Loading halo catalog: {z_cfg.halo_catalog_path}")
    halo_coords, _ = load_halo_catalog(z_cfg.halo_catalog_path)
    log.info(f"  {len(halo_coords)} halos with M > 0")

    footprint_side = footprint_side_mpc(BOX_REDSHIFT, cfg.fov_area_arcmin2)
    log.info(f"NIRCam footprint side at z={BOX_REDSHIFT}: {footprint_side:.3f} Mpc")

    muv_paths = {
        "fiducial": z_cfg.muv_fiducial_path,
        "stochastic": z_cfg.muv_stochastic_path,
        PREJWST_MODEL: PREJWST_MUV_PATH,
    }
    n_realizations_per_model = {
        "fiducial": args.n_realizations,
        "stochastic": args.n_realizations,
        PREJWST_MODEL: PREJWST_N_REALIZATIONS,
    }

    gamma_ref: dict[str, dict] = {}

    for zrange_label, (zlo, zhi) in Z_RANGES.items():
        depth_requested = comoving_depth_mpc(zlo, zhi)
        depth, truncated = effective_los_depth(depth_requested, cfg.box_len_mpc)
        log.info(
            f"[{zrange_label}] z=({zlo}, {zhi})  requested depth={depth_requested:.1f} Mpc  "
            f"effective depth={depth:.1f} Mpc"
            + ("  [TRUNCATED]" if truncated else "")
        )

        for model, muv_path in muv_paths.items():
            n_real = n_realizations_per_model[model]
            log.info(f"  Building pointing pool: model={model}, n_realizations={n_real} ...")
            pool = build_pointing_pool(
                halo_coords, muv_path, n_real, cfg, depth, footprint_side,
            )
            log.info(f"    pool shape: {pool.shape}")

            if recompute_bootstrap:
                means, varis = bootstrap_group_stats(pool, cfg)
                mean, mean_sq = pool_moments(pool)
                sigma_cv2_trials = bootstrap_fractional_cosmic_variance(pool, cfg)
                results[model][zrange_label] = (means, varis, mean, mean_sq, sigma_cv2_trials)

            if model == PREJWST_MODEL:
                pj_sample_size = min(args.prejwst_gamma_sample_size, pool.shape[0])
                log.info(
                    f"  Gamma/NB MCMC fit (plotted estimate): model={model}, "
                    f"sample_size={pj_sample_size} (of pool={pool.shape[0]}) ..."
                )
                rng = np.random.default_rng(0)
                idx = rng.choice(pool.shape[0], size=pj_sample_size, replace=False)
                sample = pool[idx]
                gamma_cfg = GammaCVConfig()
                fits = {
                    threshold: fit_sigma_cv_mcmc(sample[:, k], gamma_cfg)
                    for k, threshold in enumerate(cfg.thresholds)
                }
                print(f"\n  Gamma/NB MCMC fit — {model}, {zrange_label} (one {pj_sample_size}-pointing draw, plotted):")
                print(summarize_gamma_fits(fits, cfg.thresholds))
                label, entry = gamma_fit_to_reference(model, zlo, zhi, zrange_label, fits, cfg.thresholds)
                gamma_ref[label] = entry
            else:
                fullpool_n = min(args.gamma_fit_max_fullpool_size, pool.shape[0])
                log.info(
                    f"  Gamma/NB MCMC fit: model={model}, "
                    f"group_size={args.gamma_fit_group_size} + {fullpool_n}-pointing precision check (plotted) ..."
                )
                fits_sub, fits_full = _run_gamma_fits(
                    pool, cfg, args.gamma_fit_group_size,
                    max_fullpool_size=args.gamma_fit_max_fullpool_size,
                )
                if not args.skip_gamma_fit:
                    print(
                        f"\n  Gamma/NB MCMC fit — {model}, {zrange_label} "
                        f"(one {args.gamma_fit_group_size}-pointing draw):"
                    )
                    print(summarize_gamma_fits(fits_sub, cfg.thresholds))
                    print(f"\n  Gamma/NB MCMC fit — {model}, {zrange_label} ({fullpool_n}-pointing precision check, plotted):")
                    print(summarize_gamma_fits(fits_full, cfg.thresholds))
                label, entry = gamma_fit_to_reference(model, zlo, zhi, zrange_label, fits_full, cfg.thresholds)
                gamma_ref[label] = entry

    _print_summaries(results, cfg)
    if recompute_bootstrap:
        save_cosmic_variance(cache_path, results)
    plot_results = {model: by_zrange for model, by_zrange in results.items() if model not in GAMMA_PLOTTED_MODELS}
    _save_plot(plot_results, cfg, plot_path, reference=gamma_ref)
    log.info("All done.")


if __name__ == "__main__":
    main()