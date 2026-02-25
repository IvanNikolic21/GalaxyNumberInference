#!/usr/bin/env python
"""
run_analysis.py
---------------
End-to-end launch script for the galaxy neighbor / d1s pipeline.

Run interactively:
    python run_analysis.py --redshift 10.5 --muv-realizations 10

Submit via SLURM:
    sbatch run_slurm.sh
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from galaxy_neighbors import RedshiftConfig, AnalysisConfig, run_neighbor_analysis
from galaxy_d1s import D1sConfig, load_or_compute_d1s, plot_d1s_grid

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
# Redshift registry  —  add / edit paths here
# ---------------------------------------------------------------------------
# Shared base path — z=8, z=12, z=14 all use seed 1955; z=10.5 uses 1952
_CACHE_BASE = "/lustre/astro/ivannik/21cmFAST_cache/d12b21e80b7885d62d31717c2c2d8421"
_HASH       = "ffa852ccaa39d8f82951cc98ff798ab4"

REDSHIFT_CONFIGS = {
    8.0: RedshiftConfig(
        redshift=8.0,
        halo_catalog_path=Path(f"{_CACHE_BASE}/1955/{_HASH}/8.0000/HaloCatalog.h5"),
        muv_fiducial_path=Path("/lustre/astro/ivannik/catalog_fiducial_bigger_z8.h5"),
        muv_stochastic_path=Path("/lustre/astro/ivannik/catalog_stoch_bigger_z8.h5"),
    ),
    10.5: RedshiftConfig(
        redshift=10.5,
        halo_catalog_path=Path(f"{_CACHE_BASE}/1952/{_HASH}/10.5000/HaloCatalog.h5"),
        muv_fiducial_path=Path("/lustre/astro/ivannik/catalog_fiducial_bigger_new_save.h5"),
        muv_stochastic_path=Path("/lustre/astro/ivannik/catalog_stoch_bigger_new3.h5"),
    ),
    12.0: RedshiftConfig(
        redshift=12.0,
        halo_catalog_path=Path(f"{_CACHE_BASE}/1955/{_HASH}/12.0000/HaloCatalog.h5"),
        muv_fiducial_path=Path("/lustre/astro/ivannik/catalog_fiducial_bigger_z12.h5"),
        muv_stochastic_path=Path("/lustre/astro/ivannik/catalog_stoch_bigger_z12.h5"),
    ),
    14.0: RedshiftConfig(
        redshift=14.0,
        halo_catalog_path=Path(f"{_CACHE_BASE}/1955/{_HASH}/14.0000/HaloCatalog.h5"),
        muv_fiducial_path=Path("/lustre/astro/ivannik/catalog_fiducial_bigger_z14.h5"),
        muv_stochastic_path=Path("/lustre/astro/ivannik/catalog_stoch_bigger_z14.h5"),
    ),
}

# Sensible default number of realizations per redshift, reflecting catalog sizes.
# Used in the SLURM script comments and as documentation — override via CLI as needed.
DEFAULT_REALIZATIONS = {
    8.0:  1,
    10.5: 1,    # has named files (new_save / new3) rather than a large stack
    12.0: 50,
    14.0: 200,
}

AVAILABLE_REDSHIFTS = sorted(REDSHIFT_CONFIGS.keys())

# ---------------------------------------------------------------------------
# Analysis configuration  —  change magnitude grids here
# ---------------------------------------------------------------------------
analysis_cfg = AnalysisConfig(
    bright_limits         = [-20.5, -20.75, -21.0, -21.25, -21.5, -21.75, -22.0],
    faint_limits          = [-17.0, -17.1, -17.2, -17.3, -17.4, -17.5, -17.6, -17.7,
                             -17.8, -17.9, -18.0, -18.1, -18.2, -18.3, -18.4, -18.5,
                             -18.6, -18.7, -18.8, -18.9, -19.0, -19.1, -19.2],
    preselect_faint_limit = -17.0,
    survey_area_arcmin2   = 12.24,
)

d1s_cfg = D1sConfig(
    min_neighbors = 2,
    n_bins        = 5,
    plot_d_max    = 8.0,
    bw_fid        = 0.18,
    bw_stoc       = 0.11,
)

# Root directories  —  each redshift gets its own subfolder automatically
CACHE_ROOT  = Path("/groups/astro/ivannik/projects/Neighbors/cache")
OUTPUT_ROOT = Path("/groups/astro/ivannik/projects/Neighbors")

PLOT_FILENAMES = {
    "M21.5": "d1s_Muv0_m21p5.pdf",
    "M21":   "d1s_Muv0_m21.pdf",
    "M22":   "d1s_Muv0_m22.pdf",
}

# ---------------------------------------------------------------------------
# Cache filename helper
# ---------------------------------------------------------------------------

def make_cache_name(model: str, muv_index, n_realizations) -> str:
    """Build a descriptive .npz filename encoding the realization selection.

    Examples
    --------
    make_cache_name("fiducial", 3,      None) -> "d1s_fiducial_idx3.npz"
    make_cache_name("fiducial", [0,1,2],None) -> "d1s_fiducial_idx0-1-2.npz"
    make_cache_name("fiducial", 0,      10  ) -> "d1s_fiducial_real10.npz"
    """
    if n_realizations is not None:
        tag = f"real{n_realizations}"
    elif isinstance(muv_index, list):
        tag = "idx" + "-".join(str(i) for i in muv_index)
    else:
        tag = f"idx{muv_index}"
    return f"d1s_{model}_{tag}.npz"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Galaxy neighbor d1s pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--redshift", type=float, required=True,
        choices=AVAILABLE_REDSHIFTS, metavar="Z",
        help=f"Redshift snapshot to analyse. Available: {AVAILABLE_REDSHIFTS}",
    )

    real_group = p.add_mutually_exclusive_group()
    real_group.add_argument(
        "--muv-index", type=int, nargs="+", default=[0], metavar="I",
        help="One or more realization indices to load and concatenate.",
    )
    real_group.add_argument(
        "--muv-realizations", type=int, metavar="N",
        help="Load the first N realizations ([:N]) and concatenate.",
    )

    p.add_argument(
        "--force-recompute", action="store_true",
        help="Ignore existing cache and recompute d1s from scratch.",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip plotting (useful for quick cache-building runs).",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    z_cfg = REDSHIFT_CONFIGS[args.redshift]
    log.info(f"Redshift: {z_cfg.redshift}  ({z_cfg.label})")

    # Resolve realization selection
    if args.muv_realizations is not None:
        muv_index      = 0
        n_realizations = args.muv_realizations
    else:
        muv_index      = args.muv_index if len(args.muv_index) > 1 else args.muv_index[0]
        n_realizations = None

    # Redshift-specific subdirectories
    cache_dir  = CACHE_ROOT  / z_cfg.cache_subdir
    output_dir = OUTPUT_ROOT / z_cfg.cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_fid  = cache_dir / make_cache_name("fiducial",   muv_index, n_realizations)
    cache_stoc = cache_dir / make_cache_name("stochastic", muv_index, n_realizations)
    log.info(f"Cache dir:  {cache_dir}")
    log.info(f"Output dir: {output_dir}")

    # ------------------------------------------------------------------
    # Part 1: neighbor search
    # ------------------------------------------------------------------
    need_part1 = args.force_recompute or not (cache_fid.exists() and cache_stoc.exists())
    results_fid = results_stoc = None

    if need_part1:
        log.info("Part 1: running neighbor search ...")
        t0 = time.perf_counter()
        results_fid, results_stoc = run_neighbor_analysis(
            redshift_cfg   = z_cfg,
            analysis_cfg   = analysis_cfg,
            muv_index      = muv_index,
            n_realizations = n_realizations,
        )
        log.info(f"Part 1 done in {time.perf_counter() - t0:.1f}s")
    else:
        log.info("Part 1: cache found, skipping neighbor search.")

    # ------------------------------------------------------------------
    # Part 2: compute / load d1s
    # ------------------------------------------------------------------
    log.info("Part 2: computing / loading d1s ...")
    t0 = time.perf_counter()
    d1s_fid = load_or_compute_d1s(
        path=cache_fid, results=results_fid, cfg=analysis_cfg, redshift_cfg=z_cfg,
        d1s_cfg=d1s_cfg, force_recompute=args.force_recompute,
    )
    d1s_stoc = load_or_compute_d1s(
        path=cache_stoc, results=results_stoc, cfg=analysis_cfg, redshift_cfg=z_cfg,
        d1s_cfg=d1s_cfg, force_recompute=args.force_recompute,
    )
    log.info(f"Part 2 done in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Part 3: plots
    # ------------------------------------------------------------------
    if args.no_plots:
        log.info("--no-plots set, skipping figures.")
        return

    plt.style.use("seaborn-v0_8-ticks")
    plt.rcParams.update({
        "font.size": 16, "xtick.top": True, "ytick.right": True,
        "xtick.direction": "in", "ytick.direction": "in",
    })

    log.info("Part 3: generating plots ...")
    for bright_key in analysis_cfg.bright_names:
        fig = plot_d1s_grid(
            d1s_fid=d1s_fid, d1s_stoc=d1s_stoc,
            cfg=analysis_cfg, d1s_cfg=d1s_cfg,
            bright_key=bright_key,
            n_cols=2,
            redshift_label=z_cfg.redshift,
        )
        fname = PLOT_FILENAMES.get(bright_key, f"d1s_{bright_key}.pdf")
        out_path = output_dir / fname
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved: {out_path}")

    log.info("All done.")


if __name__ == "__main__":
    main()
