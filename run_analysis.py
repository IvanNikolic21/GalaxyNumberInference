#!/usr/bin/env python
"""
run_analysis.py
---------------
End-to-end launch script for the galaxy neighbor / d1s pipeline.

Run interactively:
    python run_analysis.py

Or submit via SLURM:
    sbatch run_slurm.sh
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for cluster nodes
import matplotlib.pyplot as plt

from galaxy_neighbors import AnalysisConfig, run_neighbor_analysis
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
# Paths  —  edit these
# ---------------------------------------------------------------------------
HALO_CATALOG = Path(
    "/lustre/astro/ivannik/21cmFAST_cache/"
    "d12b21e80b7885d62d31717c2c2d8421/1952/"
    "ffa852ccaa39d8f82951cc98ff798ab4/10.5000/HaloCatalog.h5"
)
MUV_FIDUCIAL  = Path("/lustre/astro/ivannik/catalog_fiducial_bigger_new_save.h5")
MUV_STOCH     = Path("/lustre/astro/ivannik/catalog_stoch_bigger_new3.h5")

OUTPUT_DIR    = Path("/groups/astro/ivannik/projects/Neighbors")
CACHE_DIR     = OUTPUT_DIR / "cache"

# ---------------------------------------------------------------------------
# Configuration  —  change magnitude grids here, nothing else needs touching
# ---------------------------------------------------------------------------
cfg = AnalysisConfig(
    bright_limits        = [-22.0, -21.5, -21.0],
    faint_limits         = [-17.3, -17.4, -17.5, -17.6, -17.7, -17.8, -17.9, -18.0, -18.1,  -18.2, -18.3, -18.4, -18.5, -18.6, -18.7, -18.8],
    preselect_faint_limit= -17.3,
    redshift             = 10.5,
    survey_area_arcmin2  = 12.24,
)

d1s_cfg = D1sConfig(
    min_neighbors = 2,
    n_bins        = 5,
    plot_d_max    = 8.0,
    bw_fid        = 0.18,
    bw_stoc       = 0.11,
)

# Output filenames keyed by bright_key
PLOT_FILENAMES = {
    "M21.5": "d1s_Muv0_m21p5.pdf",
    "M21":   "d1s_Muv0_m21.pdf",
    "M22":   "d1s_Muv0_m22.pdf",
}

# ---------------------------------------------------------------------------
# Cache filename helper
# ---------------------------------------------------------------------------

def make_cache_name(model: str, muv_index, n_realizations) -> str:
    """Build a descriptive cache filename encoding the realization selection.

    Examples
    --------
    make_cache_name("fiducial", 3,    None) -> "d1s_fiducial_idx3.npz"
    make_cache_name("fiducial", [0,1,2], None) -> "d1s_fiducial_idx0-1-2.npz"
    make_cache_name("fiducial", 0,    10  ) -> "d1s_fiducial_real10.npz"
    """
    if n_realizations is not None:
        tag = f"real{n_realizations}"
    elif isinstance(muv_index, list):
        tag = "idx" + "-".join(str(i) for i in muv_index)
    else:
        tag = f"idx{muv_index}"
    return f"d1s_{model}_{tag}.npz"


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Galaxy neighbor d1s pipeline")

    # Realization selection — mutually exclusive
    real_group = p.add_mutually_exclusive_group()
    real_group.add_argument(
        "--muv-index", type=int, nargs="+", default=[0], metavar="I",
        help=(
            "One or more realization indices to load and concatenate. "
            "E.g. --muv-index 0  or  --muv-index 0 1 2 3  (default: 0)"
        ),
    )
    real_group.add_argument(
        "--muv-realizations", type=int, metavar="N",
        help=(
            "Load the first N realizations ([:N]) and concatenate. "
            "Mutually exclusive with --muv-index."
        ),
    )

    p.add_argument(
        "--force-recompute", action="store_true",
        help="Ignore existing cache files and recompute d1s from scratch.",
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

    # Resolve realization selection into the two kwargs for run_neighbor_analysis
    if args.muv_realizations is not None:
        muv_index = 0              # ignored downstream
        n_realizations = args.muv_realizations
    else:
        muv_index = args.muv_index if len(args.muv_index) > 1 else args.muv_index[0]
        n_realizations = None


    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cache_fid  = CACHE_DIR / f"d1s_fiducial_idx{args.muv_index}.npz"
    cache_stoc = CACHE_DIR / f"d1s_stochastic_idx{args.muv_index}.npz"

    log.info(f"Cache files: {cache_fid.name}  |  {cache_stoc.name}")
    # ------------------------------------------------------------------
    # Part 1: neighbor search  (skipped if both caches already exist)
    # ------------------------------------------------------------------
    need_part1 = args.force_recompute or not (cache_fid.exists() and cache_stoc.exists())

    results_fid = results_stoc = None
    if need_part1:
        log.info("Part 1: running neighbor search ...")
        t0 = time.perf_counter()
        results_fid, results_stoc = run_neighbor_analysis(
            fiducial_halo_path   = HALO_CATALOG,
            fiducial_muv_path    = MUV_FIDUCIAL,
            stochastic_halo_path = HALO_CATALOG,
            stochastic_muv_path  = MUV_STOCH,
            config               = cfg,
            muv_index            = args.muv_index,
            n_realizations=n_realizations,
        )
        log.info(f"Part 1 done in {time.perf_counter() - t0:.1f}s")
    else:
        log.info("Part 1: both cache files found, skipping neighbor search.")

    # ------------------------------------------------------------------
    # Part 2: compute / load d1s
    # ------------------------------------------------------------------
    log.info("Part 2: computing / loading d1s ...")
    t0 = time.perf_counter()

    d1s_fid = load_or_compute_d1s(
        path=cache_fid, results=results_fid, cfg=cfg, d1s_cfg=d1s_cfg,
        force_recompute=args.force_recompute,
    )
    d1s_stoc = load_or_compute_d1s(
        path=cache_stoc, results=results_stoc, cfg=cfg, d1s_cfg=d1s_cfg,
        force_recompute=args.force_recompute,
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
    for bright_key in cfg.bright_names:
        fig = plot_d1s_grid(
            d1s_fid=d1s_fid, d1s_stoc=d1s_stoc,
            cfg=cfg, d1s_cfg=d1s_cfg,
            bright_key=bright_key,
            n_cols=2,
            redshift_label=10.5,
        )
        fname = PLOT_FILENAMES.get(bright_key, f"d1s_{bright_key}.pdf")
        out_path = OUTPUT_DIR / fname
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved: {out_path}")

    log.info("All done.")


if __name__ == "__main__":
    main()
