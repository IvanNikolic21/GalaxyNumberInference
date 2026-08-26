#!/usr/bin/env python
"""
generate_coeval_box.py
------------------------
Generate one independent 21cmFAST coeval box at z=10.5, for a given random
seed. Adapted from Ivan's original single-seed script: restricted to z=10.5
only (the only redshift this project currently needs), and dropped the
manual per-redshift .save() calls -- p21c.OutputCache already persists
everything needed, so those were redundant.

Run once per seed (see run_coeval_boxes_slurm.sh for launching several in
parallel as a job array).

Usage
-----
    python generate_coeval_box.py --seed 1955
"""

import argparse

import py21cmfast as p21c
from py21cmfast import config

config["HALO_CATALOG_MEM_FACTOR"] = 5.0   # start with 3-5
config["SAMPLER_BUFFER_FACTOR"] = 3.0     # optional but helpful


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True,
                   help="21cmFAST random seed -- use a different one per independent box.")
    p.add_argument("--n-threads", type=int, default=8)
    p.add_argument("--cache-dir", type=str, default="/lustre/astro/ivannik/21cmFAST_cache/")
    return p.parse_args()


def main():
    args = parse_args()

    inputs = p21c.InputParameters(
        # cosmo_params = p21c.CosmoParams(SIGMA_8=0.6),
        astro_options=p21c.AstroOptions(USE_TS_FLUCT=True),
        matter_options=p21c.MatterOptions(SOURCE_MODEL='CHMF-SAMPLER'),
        simulation_options=p21c.SimulationOptions(
            SAMPLER_MIN_MASS=5e8,
            N_THREADS=args.n_threads,
            BOX_LEN=512,
            HII_DIM=512,
        ),
        random_seed=args.seed,
    )

    cache = p21c.OutputCache(args.cache_dir)

    # Only z=10.5 -- the sole redshift this project currently uses. Output is
    # cached automatically; no need to hold onto or manually save the result.
    p21c.run_coeval(inputs=inputs, out_redshifts=[10.5], cache=cache, regenerate=False)


if __name__ == "__main__":
    main()