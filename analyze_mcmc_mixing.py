#!/usr/bin/env python
"""
analyze_mcmc_mixing.py
-----------------------
Diagnose whether an apparently multimodal NRE posterior is real (all walkers
agree on the same, genuinely multimodal target distribution) or an MCMC
mixing failure (different walkers get stuck in different regions and never
cross over -- which looks identical to real multimodality in a flattened
corner plot, but isn't).

Requires a run of infer_nre.py with --save-chain, which writes
chain_raw_N{n}.npy (shape: n_steps-n_burn, n_walkers, N_PARAMS) and
rhat_N{n}.npz alongside the usual posterior_samples/corner outputs.

Usage
-----
    python analyze_mcmc_mixing.py --chain-dir /path/to/output_dir --n-obs 50
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PARAM_NAMES = ["Muv_add", "sigmaUV_a", "sigmaUV_b"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chain-dir", type=Path, required=True,
                   help="Directory containing chain_raw_N{n}.npy and rhat_N{n}.npz "
                        "(the --output-dir passed to infer_nre.py --save-chain).")
    p.add_argument("--n-obs", type=int, required=True,
                   help="The N used when the chain was generated (sets the filename).")
    p.add_argument("--param-index", type=int, default=0,
                   help="Which parameter to check for mode occupancy (default 0 = Muv_add, "
                        "the one that showed apparent bimodality).")
    p.add_argument("--mode-boundaries", type=float, nargs=2, default=None,
                   metavar=("LOW_HIGH_CUT", "HIGH_LOW_CUT"),
                   help="Two thresholds splitting the parameter range into 'low mode' "
                        "(< first value), 'trough' (between), and 'high mode' (> second value). "
                        "Default: data-driven, using the 20th/80th percentile of the flattened "
                        "samples as a rough split.")
    args = p.parse_args()

    chain_path = args.chain_dir / f"chain_raw_N{args.n_obs}.npy"
    rhat_path  = args.chain_dir / f"rhat_N{args.n_obs}.npz"
    if not chain_path.exists():
        raise FileNotFoundError(
            f"{chain_path} not found -- rerun infer_nre.py with --save-chain to produce it."
        )

    chain = np.load(chain_path)  # (n_steps, n_walkers, N_PARAMS)
    rhat  = dict(np.load(rhat_path)) if rhat_path.exists() else {}
    n_steps, n_walkers, n_params = chain.shape
    log.info(f"Loaded chain: {n_steps} post-burn-in steps x {n_walkers} walkers x {n_params} params")

    log.info("")
    log.info("=== R-hat (Gelman-Rubin) per parameter ===")
    log.info("R-hat ~ 1.0 -> walkers agree (good mixing, or a genuinely shared multimodal "
             "target). R-hat >> 1.1 -> walkers disagree -- a mixing failure, not necessarily "
             "real structure.")
    for i, name in enumerate(PARAM_NAMES[:n_params]):
        key = f"param{i}"
        if key in rhat:
            r = float(rhat[key])
            flag = "" if r < 1.1 else "  <-- POOR MIXING"
            log.info(f"  {name}: R-hat = {r:.4f}{flag}")

    # Per-walker mode occupancy for the flagged parameter
    pidx = args.param_index
    pname = PARAM_NAMES[pidx] if pidx < len(PARAM_NAMES) else f"param{pidx}"
    flat = chain[:, :, pidx].ravel()

    if args.mode_boundaries is not None:
        lo_cut, hi_cut = args.mode_boundaries
    else:
        lo_cut, hi_cut = np.percentile(flat, [20, 80])
    log.info("")
    log.info(f"=== Per-walker mode occupancy for {pname} ===")
    log.info(f"Low mode: < {lo_cut:.3f}   High mode: > {hi_cut:.3f}   "
             f"(pass --mode-boundaries to override)")

    n_visit_both, n_low_only, n_high_only, n_neither = 0, 0, 0, 0
    for w in range(n_walkers):
        wchain = chain[:, w, pidx]
        visits_low  = np.any(wchain < lo_cut)
        visits_high = np.any(wchain > hi_cut)
        if visits_low and visits_high:
            n_visit_both += 1
        elif visits_low:
            n_low_only += 1
        elif visits_high:
            n_high_only += 1
        else:
            n_neither += 1

    log.info(f"  Walkers visiting BOTH regions (real multimodality/good mixing): {n_visit_both}/{n_walkers}")
    log.info(f"  Walkers confined to LOW region only:  {n_low_only}/{n_walkers}")
    log.info(f"  Walkers confined to HIGH region only: {n_high_only}/{n_walkers}")
    log.info(f"  Walkers visiting neither (stuck in the trough): {n_neither}/{n_walkers}")
    log.info("")
    if n_visit_both / n_walkers > 0.5:
        log.info("  ==> Most walkers cross between both regions: this is consistent with a "
                 "genuinely shared (multimodal, or boundary-pinned) target distribution, not a "
                 "mixing failure. Proceed to the network-independent degeneracy check.")
    else:
        log.info("  ==> Most walkers stay confined to one region: consistent with a MIXING "
                 "FAILURE, not necessarily real structure. Try more steps/burn-in, more walkers, "
                 "or a different sampler before concluding the posterior is really multimodal.")


if __name__ == "__main__":
    main()
