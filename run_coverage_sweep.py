#!/usr/bin/env python
"""
run_coverage_sweep.py
----------------------
Run NRE inference (both full-MLP and d1-only models) across every truth in a
coverage-check parameter file, then tabulate the resulting posteriors
(median, 68% CI) against the injected truths. Reuses whatever
nre_Madd..._sa..._sb....npz observation files already exist in --obs-dir
(built via generate_catalog_database.py + build_nre_database.py).

This is the sweep used to diagnose (and later re-check) the sigma_UV
training bias described in nre_sigma_bias_report.pdf -- see that report for
context. Re-run this after retraining to compare against the pre-fix numbers.

Usage
-----
    # Run inference for every truth, then print the summary table
    python run_coverage_sweep.py --param-file coverage_params.dat \
        --obs-dir /groups/astro/ivannik/projects/Neighbors/nre_database_new \
        --model-dir-full /groups/astro/ivannik/projects/Neighbors/nre_model_prioronly_capped_only_ang \
        --model-dir-d1   /groups/astro/ivannik/projects/Neighbors/nre_model_d1_prioronly_capped_only_ang \
        --output-dir /groups/astro/ivannik/projects/Neighbors/coverage_check_v2 \
        --n-obs 10

    # Just re-print the summary table from an already-completed sweep
    python run_coverage_sweep.py --output-dir .../coverage_check_v2 --summarize-only
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

from build_nre_database import make_output_name  # same encoding used to build obs files

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PARAM_NAMES = ["Muv_add", "sigmaUV_a", "sigmaUV_b"]


def truth_label(muv_add: float, sa: float, sb: float) -> str:
    """Short, filesystem-safe label for a truth's output subdirectory."""
    def enc(v):
        return f"{v:+.2f}".replace(".", "p").replace("+", "p").replace("-", "m")
    return f"Madd{enc(muv_add)}_sa{enc(sa)}_sb{enc(sb)}"


def run_inference(script: str, obs_file: Path, model_dir: Path, output_dir: Path, n_obs: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, script,
        "--obs-file", str(obs_file),
        "--model-dir", str(model_dir),
        "--output-dir", str(output_dir),
        "--n-obs", str(n_obs),
    ]
    log.info(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"  FAILED ({script}):\n{result.stderr[-2000:]}")
        return False
    return True


def summarize(output_dir: Path) -> dict | None:
    """Load whichever posterior_samples_*.npy landed in output_dir and
    return {param_name: (lo16, median, hi84)}, or None if nothing found."""
    candidates = sorted(output_dir.glob("posterior_samples_*.npy"))
    if not candidates:
        return None
    samples = np.load(candidates[-1])  # if >1, last one (alphabetically) wins
    stats = {}
    for i, name in enumerate(PARAM_NAMES):
        lo, med, hi = np.percentile(samples[:, i], [16, 50, 84])
        stats[name] = (lo, med, hi)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--param-file", type=Path,
                   help="Coverage-check .dat file: Muv_add sigmaUV_a sigmaUV_b log_like(unused)")
    p.add_argument("--obs-dir", type=Path,
                   help="Directory with pre-built nre_Madd..._sa..._sb....npz observation files.")
    p.add_argument("--model-dir-full", type=Path, default=None,
                   help="full-MLP model dir (for infer_nre.py). Omit to skip full-MLP.")
    p.add_argument("--model-dir-d1", type=Path, default=None,
                   help="d1-only model dir (for infer_nre_d1.py). Omit to skip d1-only.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Base directory for per-truth subdirectories and the summary table.")
    p.add_argument("--n-obs", type=int, default=10)
    p.add_argument("--summarize-only", action="store_true",
                   help="Skip running inference; just tabulate whatever's already in --output-dir "
                        "(subdirectories named <label>_full / <label>_d1).")
    args = p.parse_args()

    if not args.summarize_only:
        if args.param_file is None or args.obs_dir is None:
            p.error("--param-file and --obs-dir are required unless --summarize-only is set.")
        params = np.loadtxt(args.param_file)
        if params.ndim == 1:
            params = params[np.newaxis, :]

        for muv_add, sa, sb, *_ in params:
            obs_file = args.obs_dir / make_output_name(muv_add, sa, sb)
            label = truth_label(muv_add, sa, sb)
            log.info(f"Truth ({muv_add:+.2f}, {sa:+.2f}, {sb:+.2f}) -> {label}")

            if not obs_file.exists():
                log.warning(f"  Observation file not found, skipping this truth: {obs_file}")
                continue

            if args.model_dir_full is not None:
                run_inference("infer_nre.py", obs_file, args.model_dir_full,
                               args.output_dir / f"{label}_full", args.n_obs)
            if args.model_dir_d1 is not None:
                run_inference("infer_nre_d1.py", obs_file, args.model_dir_d1,
                               args.output_dir / f"{label}_d1", args.n_obs)

    # ------------------------------------------------------------------
    # Tabulate whatever's in output_dir
    # ------------------------------------------------------------------
    rows = []
    for sub in sorted(args.output_dir.glob("*_full")) + sorted(args.output_dir.glob("*_d1")):
        stats = summarize(sub)
        if stats is None:
            continue
        model = "full-MLP" if sub.name.endswith("_full") else "d1-only"
        label = sub.name[:-len("_full")] if sub.name.endswith("_full") else sub.name[:-len("_d1")]
        rows.append((label, model, stats))

    if not rows:
        log.warning("No posterior_samples_*.npy found anywhere in --output-dir.")
        return

    header = f"{'truth':<24s} {'model':<10s} " + \
             "".join(f"{name:>22s}" for name in PARAM_NAMES)
    print(header)
    print("-" * len(header))
    for label, model, stats in rows:
        line = f"{label:<24s} {model:<10s} "
        for name in PARAM_NAMES:
            lo, med, hi = stats[name]
            line += f"{med:+7.2f} [{lo:+5.2f},{hi:+5.2f}] "
        print(line)


if __name__ == "__main__":
    main()