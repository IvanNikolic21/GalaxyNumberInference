#!/usr/bin/env python
"""
train_halo_mass.py
--------------------
Separate network: infer p(log10(Mh/Msun) | environment, theta) for the
single bright galaxy that environment belongs to, conditioned on the
astrophysical model parameters theta = (Muv_add, sigmaUV_a, sigmaUV_b) as a
KNOWN input (not something being inferred here -- that's the separate NRE
task in train_nre.py/train_nre_d1.py).

Unlike the NRE scripts, this is plain conditional regression, not
ratio-estimation: no real/fake contrastive pairing, no classifier --
ground-truth log(Mh) for each bright galaxy is directly available in the
database (bright_logmh, added by build_nre_database.py +
patch_bright_logmh.py for the existing multibox databases). Output is
(mu, log_sigma) of a Gaussian, trained via negative log-likelihood, giving
a per-galaxy point estimate + uncertainty directly.

Reuses env_to_array / normalize_params / ResidualBlock from train_nre.py
rather than reimplementing them, so the environment encoding can't
silently drift out of sync between the two scripts.

Model and training choices, in one place for reference
--------------------------------------------------------
- Heteroscedastic Gaussian regression (predict mu AND a per-example sigma,
  not just a point estimate): Nix & Weigend (1994), "Estimating the mean and
  variance of the target probability distribution," IEEE ICNN. Also Kendall &
  Gal (2017), "What Uncertainties Do We Need in Bayesian Deep Learning for
  Computer Vision?", https://arxiv.org/abs/1703.04977 (their "aleatoric
  uncertainty" loss is the same Gaussian NLL used in gaussian_nll() below).
- Residual MLP blocks (Linear -> LayerNorm -> GELU -> Dropout, twice, plus a
  skip connection): residual connections are from He et al. (2016), "Deep
  Residual Learning for Image Recognition," https://arxiv.org/abs/1512.03385
  (originally for CNNs; the same skip-connection idea is used here for a
  plain MLP, purely to make a deep stack of layers easier to optimize).
- LayerNorm: Ba, Kiros & Hinton (2016), https://arxiv.org/abs/1607.06450.
- GELU activation: Hendrycks & Gimpel (2016), https://arxiv.org/abs/1606.08415.
- Dropout: Srivastava et al. (2014), JMLR 15:1929-1958,
  https://jmlr.org/papers/v15/srivastava14a.html.
- Adam optimizer: Kingma & Ba (2015), https://arxiv.org/abs/1412.6980.
- Cosine-annealing learning-rate schedule: Loshchilov & Hutter (2017),
  "SGDR: Stochastic Gradient Descent with Warm Restarts,"
  https://arxiv.org/abs/1608.03983 (we only ever use the first "cycle," i.e.
  one cosine decay from --lr down to ~0 over --epochs, no restarts).
- Gradient norm clipping: Pascanu, Mikolov & Bengio (2013), "On the
  difficulty of training recurrent neural networks,"
  https://arxiv.org/abs/1211.5063. PyTorch implementation/API docs:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- Clipping extreme importance/example weights to bound their influence on a
  gradient estimate is the same idea behind "truncated importance sampling";
  see e.g. Ionides (2008), "Truncated Importance Sampling," J. Computational
  and Graphical Statistics 17(2), https://doi.org/10.1198/106186008X320456.
- Label Distribution Smoothing (the --reweight-by-mass mechanism itself, as
  of the v4 update below): Yang, Zha, Chen, Wang & Katabi (2021), "Delving
  into Deep Imbalanced Regression," https://arxiv.org/abs/2102.09554 (ICML).
  Their diagnosis: for a CONTINUOUS target, raw inverse-bin-count weights
  are statistically unstable, because a sparsely-populated bin's raw count
  is a noisy estimate of its true density -- nearby bins' true densities are
  correlated (similar log-Mh values should be similarly rare/common), but
  raw per-bin counts don't know that, so one bin can get a wildly different
  weight from its neighbor purely from sampling noise. This is the mechanism
  that produced the >100x weight ratios --max-weight-ratio had to clip away
  in the v2/v3 runs. LDS's fix: convolve the bin-count histogram with a
  symmetric kernel BEFORE computing inverse weights, so a bin's effective
  weight reflects the smoothed density in its neighborhood rather than its
  own noisy count -- see _smooth_bin_counts() and its use below.
- beta-NLL, the fix for the v2/v4 sigma-collapse (loss frozen with log_sigma
  pegged near its ceiling by epoch ~10, under TWO very different reweighting
  schemes -- ruling out reweighting as the mechanism): Seitzer, Tavakoli,
  Kirchhof & Archambeau (2022), "On the Pitfalls of Heteroscedastic
  Uncertainty Estimation with Probabilistic Neural Networks," ICLR,
  https://arxiv.org/abs/2203.09168. Diagnosis: plain NLL's gradient on mu is
  proportional to 1/sigma^2, so sigma drifting up even slightly starves mu's
  gradient, mu stops improving, and inflating sigma further becomes the only
  way left to reduce the loss -- a runaway feedback loop. Fix: scale each
  example's NLL by sigma.detach()**(2*beta) (--beta-nll, default 0.5) --
  see gaussian_nll() below for the full mechanism.
- Deep Sets (--architecture deepset): Zaheer, Kottur, Ravanbakhsh, Poczos,
  Salakhutdinov & Smola (2017), "Deep Sets," https://arxiv.org/abs/1703.06114
  (NeurIPS). HaloMassNetwork flattens the distance-sorted, zero-padded
  neighbor list into ONE vector and feeds it to a plain MLP -- the network
  has to rediscover permutation-invariance from scratch (two environments
  that differ only in which of two nearly-equidistant neighbors sorted
  first look like completely different inputs to an MLP's first layer),
  and the v1-v7 runs' persistent shrinkage-to-the-mean bias never moved
  under any loss-function fix (LDS, beta-NLL, early stopping) -- all of
  which left this encoding untouched. HaloMassSetNetwork instead applies a
  small shared MLP (phi) to EACH neighbor row independently (so neighbor
  order literally cannot matter), masks out the padding rows, and SUM-pools
  the per-neighbor embeddings into one fixed-size vector before the same
  ResidualBlock stack (rho) as before -- see HaloMassSetNetwork below. This
  implements the paper's basic phi/sum-pool/rho architecture, not the
  attention-based Set Transformer (Lee et al. 2019, arXiv:1810.00825, also
  on the reading list) some neighbor-set problems need -- worth trying next
  if plain sum-pooling isn't enough, since attention can weight closer/
  brighter neighbors differently instead of treating the pooling sum as a
  single undifferentiated whole.
- UVLF-plausibility reweighting (--uvlf-reweight): v8 (Deep Sets) STILL
  showed the same shrinkage-to-the-mean bias (RMS identical to v7, 0.361
  dex both), which led to checking whether the uniform theta PRIOR itself
  was the problem rather than anything in the model. diagnose_uvlf_
  stratified_neighbor_count.py found the prior is overwhelmingly UVLF-
  inconsistent (sampled log-likelihoods from -1.5 to -56,069,751.6 across
  2000 theta), and plot_eval_vs_truth_uvlf_consistent.py confirmed a
  trained model STILL carries a residual bias even when evaluated only on
  the UVLF-consistent subset -- i.e. the bias partly reflects what the
  model was TRAINED on, not just what it's evaluated on. This reweights the
  training loss by each example's parent theta's UVLF plausibility,
  reusing the exact log-likelihood infer_nre.py's --use-uvlf path and
  check_uvlf_consistency.py already compute, converted to an equivalent
  Gaussian sigma and passed through a smooth Gaussian taper
  (--uvlf-sigma-scale) rather than a raw softmax -- see the long comment
  above this block in HaloMassDataset for why the raw-softmax version
  would numerically underflow nearly the entire dataset to zero weight
  given this prior's enormous log-likelihood range.

Usage
-----
    python train_halo_mass.py \\
        --database-dir /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed1955 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed2027 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed3142 \\
                        /groups/astro/ivannik/projects/Neighbors/nre_database_prior_capped_seed4242 \\
        --only-angular --epochs 100 --max-per-catalog 0 --weight-by-catalog-count \\
        --reweight-by-mass --diag-every 10 \\
        --output-dir /groups/astro/ivannik/projects/Neighbors/halo_mass_model_multibox4

--reweight-by-mass inverse-density-weights the loss by true log(Mh) bin (via
Label Distribution Smoothing, --lds-kernel-sigma controlling how much the
per-bin density estimate is smoothed before inverting it -- see Yang et al.
2021 in the docstring above), to counter the shrinkage-to-the-mean seen in
both tails of eval_vs_truth.pdf.
--diag-every logs per-mass-bin val RMS/bias/68%-coverage periodically, so tail
convergence can be checked directly instead of inferring it from the (middle-
band-dominated) mean val loss alone.
--max-weight-ratio and --grad-clip guard against the failure mode discovered
when --reweight-by-mass (then using RAW, unsmoothed inverse-bin-count
weights) was first combined with --weight-by-catalog-count: see the long
comment above the weight-clipping block in HaloMassDataset, below, for the
full story. LDS should make that clipping less necessary, but it's left on
by default -- check the "N/len clipped" log line to see how much it's
actually binding under LDS.
"""
# argparse: standard library command-line argument parser, used in parse_args().
import argparse
# logging: standard library structured logging (timestamps + levels), used
# instead of bare print() for the "real" progress messages (print() is still
# used for a few per-file skip warnings inside the data-loading loop, purely
# because those can fire thousands of times and log.info's extra formatting
# overhead/verbosity isn't worth it there).
import logging
# math: only used for math.log(2 * math.pi), the constant term in the
# Gaussian negative-log-likelihood (see gaussian_nll() below).
import math
# zipfile: a .npz file is a zip archive internally; zipfile.BadZipFile is
# caught explicitly when a database file is truncated/corrupted (e.g. an
# interrupted write), so one bad file doesn't crash the whole data-loading pass.
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
# Dataset: the abstract base class we subclass as HaloMassDataset, below --
# just requires __len__ and __getitem__. DataLoader: wraps a Dataset to give
# batching, shuffling, and (optionally, not used here) parallel loading.
# https://docs.pytorch.org/docs/stable/data.html
from torch.utils.data import Dataset, DataLoader

# Reused, not reimplemented, so this script's environment encoding can't
# silently drift out of sync with the NRE scripts' encoding:
#   env_to_array      -- turns a variable-length list of (dx, dy, dz, MUV)
#                         neighbor rows into a fixed-size, sorted-by-distance,
#                         zero-padded feature vector (see its docstring in
#                         train_nre.py for the exact layout).
#   normalize_params  -- rescales theta = (Muv_add, sigmaUV_a, sigmaUV_b) from
#                         its raw prior range to [-1, 1] componentwise, the
#                         standard "feature scaling" step for NN inputs.
#   ResidualBlock     -- one Linear->LayerNorm->GELU->Dropout->Linear->LayerNorm
#                         block with a GELU-gated additive skip connection
#                         (He et al. 2016 residual-connection idea, see the
#                         module docstring above), used as the network's
#                         repeated hidden-layer unit in HaloMassNetwork below.
#   MAX_NEIGHBORS     -- fixed number of nearest neighbors kept per
#                         environment (10); environments with fewer are
#                         zero-padded, environments with more are truncated
#                         to the 10 closest (env_to_array sorts by distance
#                         first).
#   N_FEATURES_FULL   -- number of features stored per neighbor in the
#                         non-angular, non-summary encoding: 4, i.e.
#                         (dx, dy, dz, MUV). Used below only to reshape the
#                         flat augmentation-target array back into its
#                         (MAX_NEIGHBORS, N_FEATURES_FULL) layout.
#   N_FEATURES_SUMM, N_FEATURES_ANG -- same idea for --summary-mode (2:
#                         distance, MUV) and --only-angular (3: dx, dy, MUV)
#                         -- only needed by _n_features() below, for
#                         HaloMassSetNetwork to know how to reshape the flat
#                         input back into (MAX_NEIGHBORS, n_feat) rows.
#   N_PARAMS          -- length of theta = (Muv_add, sigmaUV_a, sigmaUV_b): 3.
#                         Only needed by HaloMassSetNetwork, to know where
#                         theta starts in the flat input vector.
from train_nre import (
    env_to_array, normalize_params, ResidualBlock,
    MAX_NEIGHBORS, N_FEATURES_FULL, N_FEATURES_SUMM, N_FEATURES_ANG, N_PARAMS,
)

# Configure the module-level logger used by every log.info(...) call below:
# one line per message, each stamped with a HH:MM:SS timestamp and the level
# name (e.g. "INFO"), written to stderr (logging's default stream) -- this is
# why SLURM job logs for this script end up in the .err file, not .out (only
# the bare print() calls inside HaloMassDataset and the companion
# evaluate_halo_mass_model.py script, which don't go through this logger,
# land in .out).
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _smooth_bin_counts(counts: np.ndarray, kernel_sigma: float) -> np.ndarray:
    """Label Distribution Smoothing's kernel step (Yang et al. 2021, see the
    module docstring above): convolve a histogram of per-bin example counts
    with a symmetric Gaussian kernel, so a bin's SMOOTHED count reflects the
    density in its neighborhood rather than its own (possibly noisy, for a
    sparsely-populated bin) raw count.

    kernel_sigma is in units of BINS (not dex) -- e.g. kernel_sigma=2 with
    --reweight-bins 30 spanning the dataset's full log(Mh) range means the
    kernel's standard deviation is about 1/15th of that range. kernel_sigma=0
    disables smoothing (returns counts unchanged), which is just the old raw
    inverse-bin-count behavior -- useful for an apples-to-apples comparison.
    """
    if kernel_sigma is None or kernel_sigma <= 0:
        return counts.astype(np.float64)
    # Kernel half-width of 3*sigma captures >99% of a Gaussian's mass;
    # np.arange(-radius, radius+1) gives an odd-length, zero-centered kernel.
    radius = max(1, int(round(3 * kernel_sigma)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / kernel_sigma) ** 2)
    kernel = kernel / kernel.sum()
    # Explicit zero-pad + mode="valid" (NOT "same"): "same" returns an array
    # of length max(len(counts), len(kernel)), so if the kernel ever ends up
    # LONGER than counts (a large --lds-kernel-sigma with a small
    # --reweight-bins), "same" would silently return a longer, misaligned
    # array instead of erroring -- bin_idx below would then index into the
    # wrong positions with no crash to flag it. Zero-padding by `radius` on
    # each side first and using "valid" always returns exactly len(counts)
    # elements, correctly centered, regardless of the kernel/bin-count size
    # ratio. The zero-padding means the two end bins get less smoothing
    # support than interior bins (no real data out there to smooth them
    # with) -- a standard, accepted edge behavior for this kind of kernel
    # smoothing, not specific to LDS.
    padded = np.pad(counts.astype(np.float64), radius, mode="constant")
    return np.convolve(padded, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Dataset -- no real/fake pairing, just (environment, theta) -> true log(Mh)
# ---------------------------------------------------------------------------

class HaloMassDataset(Dataset):
    def __init__(
        self,
        database_dirs: list,
        param_min: np.ndarray,
        param_max: np.ndarray,
        augment: bool = True,
        max_per_catalog: int = 200,
        summary_mode: bool = False,
        only_angular: bool = False,
        weight_by_catalog_count: bool = False,
        reweight_by_mass: bool = False,
        reweight_alpha: float = 0.5,
        reweight_bins: int = 30,
        lds_kernel_sigma: float = 2.0,
        max_weight_ratio: float = 10.0,
        uvlf_reweight: bool = False,
        uvlf_sigma_scale: float = 5.0,
    ):
        # Theta normalization bounds (computed once, in main(), across ALL
        # loaded database files, and passed in here) -- stored so
        # __getitem__ can call normalize_params() per example at fetch time
        # rather than normalizing everything up front.
        self.param_min    = param_min
        self.param_max    = param_max
        # Whether __getitem__ applies the random spatial-shift augmentation
        # (see __getitem__, below) -- note this is a single flag shared by
        # BOTH the train and val splits, since torch.utils.data.random_split
        # (used in main()) wraps this SAME dataset instance with two index
        # subsets rather than creating two separate dataset objects; i.e.
        # augmentation is (perhaps not ideally) applied during validation too.
        self.augment      = augment
        # Passed straight through to env_to_array() at fetch time -- these
        # two flags select which of the three possible per-neighbor feature
        # layouts (and therefore the network's input width) is used; see
        # env_to_array's docstring in train_nre.py for the exact definitions.
        self.summary_mode = summary_mode
        self.only_angular = only_angular

        # Parallel lists (not yet batched into a single array), one entry
        # per (environment, log Mh) training example:
        self.envs      = []   # raw (N_neighbors_in_this_env, 4) coordinate/MUV arrays
        self.params    = []   # that example's theta = (Muv_add, sigmaUV_a, sigmaUV_b)
        self.logmh     = []   # ground-truth log10(Mh/Msun) of the bright galaxy
        self.weights   = []   # per-example loss weight (see below)

        cat_idx = 0  # counts catalogs (.npz files) actually processed; not read anywhere else
        for db_dir in database_dirs:
            # Every build_nre_database.py / patch_bright_logmh.py output file
            # for one (Muv_add, sigmaUV_a, sigmaUV_b) grid point is named
            # nre_<encoded params>.npz; sorted() just gives deterministic
            # (reproducible) file-processing order across runs/machines.
            files = sorted(Path(db_dir).glob("nre_*.npz"))
            log.info(f"Loading {len(files)} files from {db_dir} ...")

            for path in files:
                try:
                    # np.load on a .npz is lazy (an NpzFile object, not yet a
                    # dict of arrays) -- the bracketed accesses below are what
                    # actually read each named array out of the zip archive.
                    data    = np.load(path)
                    coords  = data['coords']    # (total_neighbor_rows, 4) across ALL environments in this file
                    offsets = data['offsets']   # (n_envs+1,) CSR-style index: environment i is coords[offsets[i]:offsets[i+1]]
                    params  = data['params']    # this file's fixed (Muv_add, sigmaUV_a, sigmaUV_b), shape (3,)
                    if 'bright_logmh' not in data.files:
                        # Older build_nre_database.py outputs, or seeds this
                        # session's patch_bright_logmh.py backfill hasn't
                        # reached yet, won't have this field -- the halo-mass
                        # ground truth this whole script depends on. Skip
                        # rather than crash, so a partially-patched database
                        # dir still trains on whatever IS patched.
                        print(f"Skipping file with no bright_logmh (not yet patched): {path}")
                        continue
                    bright_logmh = data['bright_logmh']  # (n_envs,) halo mass of the bright galaxy owning each environment
                except (EOFError, ValueError, OSError, KeyError, zipfile.BadZipFile) as e:
                    # Catches a truncated/corrupted .npz (e.g. a build job
                    # killed mid-write) without aborting the whole load.
                    print(f"Skipping corrupted file: {path} ({e})")
                    continue

                n_envs = len(offsets) - 1  # CSR convention: n indices need n+1 offset boundaries
                if len(bright_logmh) != n_envs:
                    # Defensive check: if patch_bright_logmh.py's reproduced
                    # selection ever drifted out of sync with this file's
                    # actual offsets (it shouldn't -- see that script's own
                    # length cross-check -- but this is the second line of
                    # defense at training time), skip the file instead of
                    # silently pairing the wrong halo mass to an environment.
                    print(f"Skipping file with mismatched bright_logmh length: {path} "
                          f"({len(bright_logmh)} vs {n_envs} offsets)")
                    continue

                # --- subsample this catalog's environments, if requested ---
                indices = np.arange(n_envs)
                if max_per_catalog is not None and max_per_catalog > 0 and len(indices) > max_per_catalog:
                    # Uniform random subsample without replacement, capping
                    # how many of this one catalog's (often very numerous)
                    # bright-galaxy environments get used -- without this,
                    # catalogs with thousands of bright galaxies could
                    # swamp catalogs with only a handful, the opposite
                    # imbalance --weight-by-catalog-count (below) targets.
                    # NOTE: uses the GLOBAL np.random state (seeded once, in
                    # main(), via np.random.seed(args.seed)), not a per-file
                    # np.random.default_rng -- so which environments get
                    # subsampled depends on the ORDER files are processed in,
                    # unlike patch_bright_logmh.py's reproducible per-row
                    # np.random.default_rng(seed + i) pattern.
                    indices = np.random.choice(indices, max_per_catalog, replace=False)

                # --- per-example weight, part 1: inverse catalog size ---
                n_used = len(indices)
                # If --weight-by-catalog-count, each of this catalog's used
                # examples gets weight 1/n_used, so EVERY catalog contributes
                # the same TOTAL weight (n_used * 1/n_used = 1) to the loss
                # regardless of how many bright galaxies it happened to have
                # -- otherwise catalogs with many bright galaxies would
                # dominate the gradient just by volume. If the flag is off,
                # every example gets weight 1.0 (plain unweighted average).
                weight = (1.0 / n_used) if (weight_by_catalog_count and n_used > 0) else 1.0

                for i in indices:
                    # CSR slice: this environment's neighbor rows.
                    env = coords[offsets[i]:offsets[i + 1]]
                    if len(env) == 0:
                        # A bright galaxy with zero neighbors within the
                        # search radius -- env_to_array would still handle
                        # this (via zero-padding), but there's no spatial
                        # information at all to learn from, so it's dropped.
                        continue
                    # .copy() so this example doesn't keep the WHOLE file's
                    # coords array alive in memory via a view/reference once
                    # `data` (the NpzFile) goes out of scope at the end of
                    # the `for path in files` loop body.
                    self.envs.append(env.copy())
                    self.params.append(params)
                    self.logmh.append(float(bright_logmh[i]))
                    self.weights.append(weight)

                cat_idx += 1

        log.info(f"Total (environment, log Mh) pairs: {len(self.envs)}")

        if reweight_by_mass and len(self.logmh) > 0:
            # Label-Distribution-Smoothing (LDS) inverse-density reweighting
            # -- see Yang et al. (2021) in the module docstring above. The
            # predicted-vs-true plot showed shrinkage-to-the-mean worst in
            # the sparsest true-logMh regions (both tails), because the
            # dense middle band dominates the loss; the fix is to give each
            # example a weight ~ 1/density(bin)^alpha. v3 computed
            # "density" as the bin's OWN raw count, which is a noisy
            # estimate for a sparsely-populated bin and produced individual
            # weight ratios >100x, destabilizing training (the sigma-
            # collapse bug --max-weight-ratio/--grad-clip were added to
            # patch over). v4 instead smooths the count histogram across
            # neighboring bins first (_smooth_bin_counts), which is the
            # actual LDS mechanism from the paper -- a less noisy, more
            # stable density estimate, so the resulting weights shouldn't
            # need nearly as aggressive clipping downstream.
            logmh_arr = np.array(self.logmh)
            # reweight_bins equal-width bins spanning the full observed
            # log(Mh) range; +1e-6 on the upper edge so the single highest-
            # mass example (which would otherwise sit exactly ON the last
            # edge) is included in the last bin rather than falling outside
            # all bins (np.digitize's right-open-interval convention).
            edges = np.linspace(logmh_arr.min(), logmh_arr.max() + 1e-6, reweight_bins + 1)
            # np.digitize returns 1-indexed bin numbers (0 means "below the
            # first edge"); the -1 converts to 0-indexed, and the outer clip
            # is a defensive guard against any example landing in bin -1 or
            # reweight_bins due to floating-point edge effects.
            bin_idx = np.clip(np.digitize(logmh_arr, edges) - 1, 0, reweight_bins - 1)
            # Raw per-bin example counts, THEN kernel-smoothed across
            # neighboring bins -- the LDS step. With lds_kernel_sigma=0 this
            # is a no-op (smoothed_counts == raw counts), i.e. the old v3
            # behavior, for direct comparison.
            counts = np.bincount(bin_idx, minlength=reweight_bins)
            smoothed_counts = _smooth_bin_counts(counts, lds_kernel_sigma)
            # Per-example weight ~ 1 / (that example's SMOOTHED bin density)^alpha.
            # alpha=1 would fully flatten the mass distribution's
            # contribution to the loss (every bin contributes equally);
            # alpha=0 disables this term entirely (weight=1 for everyone);
            # the default 0.5 is a softened compromise between the two.
            # smoothed_counts[bin_idx] maps each example back to ITS bin's
            # smoothed density via fancy indexing; np.maximum(..., eps)
            # guards against a near-zero smoothed value at the extreme
            # edges of the range (possible, unlike with raw counts, since
            # smoothing can push a low-count edge bin's value arbitrarily
            # close to zero rather than floor at the bin's own count of >=1).
            mass_weight = 1.0 / np.maximum(smoothed_counts[bin_idx], 1e-6) ** reweight_alpha
            # Renormalize so the mean weight is exactly 1 -- keeps the
            # overall magnitude of gaussian_nll's weighted average (see
            # gaussian_nll() below) from drifting as reweight_alpha/
            # reweight_bins/lds_kernel_sigma change, which would otherwise
            # also silently change the effective learning rate.
            mass_weight = mass_weight / mass_weight.mean()
            self.weights = (np.array(self.weights) * mass_weight).tolist()
            log.info(f"Applied LDS mass reweighting (bins={reweight_bins}, "
                      f"alpha={reweight_alpha}, kernel_sigma={lds_kernel_sigma} bins); "
                      f"raw counts range [{counts.min()}, {counts.max()}], "
                      f"smoothed range [{smoothed_counts.min():.1f}, {smoothed_counts.max():.1f}]")

        if uvlf_reweight and len(self.params) > 0:
            # UVLF-plausibility reweighting. Motivated by
            # diagnose_uvlf_stratified_neighbor_count.py: the UNIFORM theta
            # prior this database was built from is overwhelmingly
            # inconsistent with the observed UVLF (Donnan+24) -- sampled
            # log-likelihoods ranged from -1.5 to -56,069,751.6 across just
            # 2000 theta, i.e. nearly the whole prior is astrophysically
            # impossible. A model trained on it unweighted spends almost
            # all its gradient budget fitting examples from theta that no
            # real observation could ever favor, diluting whatever signal
            # exists in the tiny physically-relevant region --
            # plot_eval_vs_truth_uvlf_consistent.py showed a model trained
            # this way still carries a residual shrinkage bias even when
            # EVALUATED only on UVLF-consistent theta, consistent with this
            # diagnosis. This reweights the TRAINING loss itself by each
            # example's parent theta's UVLF plausibility, continuously (no
            # hard cutoff, matching how Ivan asked for the diagnostic
            # stratification to work) -- same underlying log-likelihood
            # infer_nre.py's --use-uvlf path and check_uvlf_consistency.py
            # already use, not a new consistency criterion invented here.
            #
            # Imported locally (not at module level) so running this script
            # WITHOUT --uvlf-reweight never requires hmf/scipy installed.
            from uvlf import Mason15, Observations
            from scipy.stats import chi2, norm

            log.info("Computing per-theta UVLF plausibility for reweighting "
                      "(builds Mason15(z=10.0) once, then one calculate_UVLF call per "
                      "DISTINCT theta, cached) ...")
            # Mason15's Muv_Mh_dict is only ever built for z=10.0 (see
            # uvlf.py) -- this database is at z=10.5 (build_nre_database.py's
            # REDSHIFT), but check_uvlf_consistency.py already established
            # the precedent of using Mason15(z=10.0) for this exact z=10.5
            # problem, which is followed here rather than inventing a
            # different answer to a question the rest of the pipeline has
            # already settled the same way.
            mason15  = Mason15(z=10.0)
            obs      = Observations(ang=False, uvlf=True)
            muv_obs, phi_obs, (sig_p, sig_m) = obs.get_obs_uvlf_z10_Donnan24()
            muv_grid = np.linspace(-25, -13, 100)
            dof      = len(muv_obs)  # UVLF data-vector length; theta is EVALUATED against
                                       # it, not fit to it, so no fitted-parameter dof correction.

            sigma_cache = {}

            def _uvlf_sigma(muv_add, sa, sb):
                # Same chi2 -> p -> sigma conversion as
                # plot_eval_vs_truth_uvlf_consistent.py, verified exactly
                # against the dof=1 reference case there (chi2=25 -> 5.0
                # sigma requires the /2, since chi2.sf is a one-sided tail
                # probability but chi2 itself is the two-sided-squared sum
                # of Gaussian deviates).
                key = (round(float(muv_add), 8), round(float(sa), 8), round(float(sb), 8))
                if key in sigma_cache:
                    return sigma_cache[key]
                phi_pred_grid = mason15.calculate_UVLF(muv_add, sa, sb, Muv_grid=muv_grid)
                phi_pred  = np.interp(muv_obs, muv_grid, phi_pred_grid)
                sigma_obs = np.where(phi_pred >= phi_obs, sig_p, sig_m)
                ll = -0.5 * np.sum(((phi_obs - phi_pred) / sigma_obs) ** 2)
                p  = chi2.sf(-2.0 * ll, df=dof) / 2.0
                p  = np.clip(p, 0.0, 1.0 - 1e-16)
                sigma_equiv = max(0.0, float(norm.isf(p)))
                sigma_cache[key] = sigma_equiv
                return sigma_equiv

            # NOT exp(ll - ll_max): with this prior's enormous log-L range
            # (down to -56 MILLION), that raw softmax would underflow to
            # EXACTLY 0.0 for all but a literal handful of examples in
            # double precision (exp(x) underflows below x~-745) -- rather
            # than continuously downweighting implausible theta, it would
            # silently hard-delete ~all of the training set, the opposite
            # of what a "continuous, no hard cutoff" reweighting is
            # supposed to do. Working in SIGMA units instead compresses the
            # dynamic range first (sigma grows only like sqrt(-ll)), THEN
            # applying a Gaussian taper exp(-0.5*(sigma/scale)^2) gives a
            # smooth, gradual falloff across the "broad, a few sigma"
            # plausibility range Ivan described, while still cleanly
            # vanishing for the truly absurd tail. Conceptually the same
            # family of idea as tempering an extreme likelihood before
            # using it as a weight, e.g. "power posteriors" (Friel &
            # Pettitt 2008, J. R. Stat. Soc. B, 70(3):589-607) -- not a
            # literal implementation of that method, just the same
            # motivation (an untempered extreme likelihood is numerically
            # unusable as a weight).
            sigmas = np.array([_uvlf_sigma(p_[0], p_[1], p_[2]) for p_ in self.params])
            uvlf_weight = np.exp(-0.5 * (sigmas / uvlf_sigma_scale) ** 2)
            uvlf_weight = uvlf_weight / uvlf_weight.mean()
            self.weights = (np.array(self.weights) * uvlf_weight).tolist()
            # Effective sample size (standard importance-weighting
            # diagnostic: ESS = (sum w)^2 / sum(w^2), equal to len(weights)
            # if all weights are equal and shrinking toward 0 as the weight
            # distribution gets more lopsided) -- given how overwhelmingly
            # UVLF-inconsistent this prior is, most of the dataset's TOTAL
            # weight is likely to end up concentrated on a small fraction
            # of examples; this makes that concentration visible instead of
            # a hidden side effect of the reweighting.
            ess = float(uvlf_weight.sum() ** 2 / (uvlf_weight ** 2).sum())
            log.info(f"Applied UVLF-plausibility reweighting (sigma_scale={uvlf_sigma_scale}, "
                      f"{len(sigma_cache)} distinct theta); sigma range "
                      f"[{sigmas.min():.1f}, {sigmas.max():.1f}], "
                      f"weight range [{uvlf_weight.min():.2e}, {uvlf_weight.max():.2e}], "
                      f"effective sample size {ess:.0f}/{len(uvlf_weight)} "
                      f"({ess/len(uvlf_weight):.1%})")

        if max_weight_ratio is not None and max_weight_ratio > 0 and len(self.weights) > 0:
            # The v2 overnight run showed train/val loss freeze bit-for-bit by
            # epoch ~10 with log_sigma pegged at its ceiling for ~every
            # example (mean predicted sigma = 20.09 dex, 100% "coverage") --
            # the same cheat-by-inflating-sigma failure as the earlier hard-
            # clamp bug, just re-triggered against the smooth bound. Root
            # cause: --reweight-by-mass combined (multiplicatively) with
            # --weight-by-catalog-count gave individual examples weight
            # ratios of ~100x+ (mass bin counts alone ranged [0, 20404]), so
            # a handful of extreme-weight examples dominated every batch's
            # NLL gradient, and the cheapest way to shrink their loss
            # contribution is to blow up sigma for everyone. Clipping the
            # final combined weight to a bounded ratio around the median
            # caps how much any single example can dominate a batch, without
            # removing the reweighting itself. (Same idea as "truncated
            # importance sampling" -- Ionides 2008, cited in the module
            # docstring above -- applied to an example weight instead of an
            # importance-sampling ratio.)
            w = np.array(self.weights)
            # Median computed over STRICTLY POSITIVE weights only, and the
            # lower bound is fixed at 0 rather than median/ratio -- found
            # necessary once --uvlf-reweight entered the picture: that
            # taper can legitimately drive MOST weights to exactly 0.0 (by
            # design -- that's the whole point, downweighting implausible
            # theta toward nothing), which would make the OLD median-over-
            # everything exactly 0 too, collapsing [median/ratio,
            # median*ratio] to [0, 0] and clipping the ENTIRE weight array
            # to zero -- a silent, catastrophic weight.sum()==0 crash in
            # gaussian_nll on the very first batch. Verified this failure
            # mode numerically against a realistic lopsided sigma
            # distribution before this fix existed. The lower clip was
            # never actually load-bearing for the ORIGINAL v2 bug either
            # (that was specifically about a few examples' weight being too
            # LARGE, not too small) -- dropping it is a correction, not a
            # loosening of the protection that bug needed.
            positive = w[w > 0]
            if len(positive) == 0:
                log.warning("All combined example weights are exactly 0 -- "
                            "--max-weight-ratio clipping skipped (nothing to clip around).")
            else:
                median_w = np.median(positive)
                hi = median_w * max_weight_ratio
                n_clipped = int(np.sum(w > hi))
                w = np.minimum(w, hi)
                self.weights = w.tolist()
                log.info(f"Clipped combined example weights to at most {hi:.4g} "
                          f"(ratio={max_weight_ratio}x median-of-positive-weights); "
                          f"{n_clipped}/{len(w)} examples clipped")

    def __len__(self):
        # Required by the torch.utils.data.Dataset interface: DataLoader
        # and random_split both call this to know how many examples exist.
        return len(self.envs)

    def __getitem__(self, idx):
        # Required by the Dataset interface: called once per example per
        # epoch (via DataLoader, which batches up however many __getitem__
        # calls are needed to fill one batch of size args.batch_size).
        env    = self.envs[idx]
        params = self.params[idx]

        # Sort-by-distance + zero-pad-to-MAX_NEIGHBORS + flatten, exactly as
        # train_nre.py's NRE models consume their environments; n is the
        # TRUE (pre-padding) neighbor count, dists are the (already sorted)
        # per-neighbor distances -- both only used below for augmentation,
        # not returned from this function.
        flat, n_norm, n, dists = env_to_array(env, summary_mode=self.summary_mode,
                                               only_angular=self.only_angular)

        if self.augment and n > 0 and not self.summary_mode and not self.only_angular:
            # Random rigid translation augmentation: shift every real
            # (non-padding) neighbor's (dx, dy, dz) by the SAME random
            # offset, uniform in [-5, +5] (comoving Mpc, matching the
            # coordinate units coords/env are stored in). This is valid
            # because the network's input is purely RELATIVE neighbor
            # positions around the bright galaxy -- translating the whole
            # environment rigidly doesn't change which galaxy is "bright" or
            # its true halo mass, so it's a label-preserving augmentation
            # that teaches the network not to overfit to absolute position
            # within whatever coordinate frame the simulation box happens to
            # use. Only applied in the full (dx,dy,dz,MUV) encoding -- in
            # summary_mode (distance-only) or only_angular (2D, no dz) modes
            # this particular augmentation isn't well-defined / would need a
            # different implementation, so it's skipped there.
            shift = np.random.uniform(-5.0, 5.0, size=3).astype(np.float32)
            # flat is a 1D (MAX_NEIGHBORS*N_FEATURES_FULL,) array; reshape
            # back to the 2D (MAX_NEIGHBORS, N_FEATURES_FULL) layout so the
            # shift can be added to just the first 3 (spatial) columns of
            # just the first n (real, non-padding) rows, leaving the MUV
            # column and all padding rows untouched.
            flat_2d = flat.reshape(MAX_NEIGHBORS, N_FEATURES_FULL)
            flat_2d[:n, :3] += shift
            flat = flat_2d.flatten()

        # Final network input: flattened neighbor features, the true
        # neighbor-count feature (n_norm, already /200-normalized inside
        # env_to_array), and theta rescaled to [-1, 1] -- concatenated into
        # one 1D vector of length input_dim (e.g. 34 for only_angular mode:
        # 10*3 + 1 + 3).
        x = torch.from_numpy(np.concatenate([flat, n_norm]))
        theta = torch.from_numpy(normalize_params(params, self.param_min, self.param_max))
        x_in = torch.cat([x, theta])

        # Regression target (ground-truth log Mh) and this example's loss
        # weight, both as 0-dim float32 tensors -- DataLoader's default
        # collate_fn stacks a batch of these into 1D (batch_size,) tensors.
        target = torch.tensor(self.logmh[idx], dtype=torch.float32)
        weight = torch.tensor(self.weights[idx], dtype=torch.float32)
        return x_in, target, weight


# ---------------------------------------------------------------------------
# Model -- same ResidualBlock backbone as NRENetwork, 2-output (mu, log_sigma) head
# ---------------------------------------------------------------------------

class HaloMassNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [256, 256, 256, 256], dropout: float = 0.1):
        super().__init__()
        # First layer: project the raw input_dim-length vector up to the
        # network's working width (hidden_dims[0]), then normalize
        # (LayerNorm) and nonlinearly activate (GELU) -- standard "stem"
        # before a stack of residual blocks, so every ResidualBlock below
        # can assume a fixed-width input/output.
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
        )
        blocks = []
        for i in range(len(hidden_dims)):
            in_d  = hidden_dims[i]
            # The width this block should output: the NEXT hidden_dims
            # entry, or (for the last block) its own width again, so the
            # final block's output matches hidden_dims[-1] -- the width
            # self.output (below) expects.
            out_d = hidden_dims[i + 1] if i + 1 < len(hidden_dims) else hidden_dims[i]
            if in_d == out_d:
                # Widths match: use a true residual block (with the
                # He-et-al.-style additive skip connection, see ResidualBlock
                # in train_nre.py) -- the default --hidden-dims (four or six
                # equal-width layers) hits this branch every time.
                blocks.append(ResidualBlock(in_d, dropout))
            else:
                # Widths differ: a skip connection isn't shape-compatible
                # without an extra projection, so just use a plain
                # Linear->LayerNorm->GELU->Dropout layer with no skip. Not
                # currently exercised by this script's own CLI default or
                # the overnight scripts (which always pass equal-width
                # --hidden-dims), but kept for compatibility with
                # train_nre.py's NRENetwork, which supports varying widths.
                blocks.append(nn.Sequential(
                    nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.GELU(), nn.Dropout(dropout),
                ))
        self.blocks = nn.ModuleList(blocks)
        # Final linear head: 2 outputs per example, (mu, pre-sigmoid log_sigma)
        # -- "pre-sigmoid" because forward() below still has to squash this
        # second output through a bounded sigmoid transform before it's a
        # usable log_sigma; see forward().
        self.output = nn.Linear(hidden_dims[-1], 2)  # (mu, pre-sigmoid log_sigma)
        with torch.no_grad():
            # Start log_sigma's raw output at 0 -> sigmoid(0)=0.5 -> log_sigma=0 (sigma=1)
            # initially, a sane starting uncertainty rather than drifting toward the
            # edge of the allowed range right out of the gate. (nn.Linear's default
            # init already gives weights/biases small random values; this just
            # overrides the bias for the log_sigma output specifically, leaving
            # the mu output's bias at its default random init.)
            self.output.bias[1] = 0.0

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        out = self.output(h)
        mu = out[:, 0]
        # Smooth (always-differentiable) bound instead of a hard clamp: a hard
        # clamp has exactly zero gradient once log_sigma is pushed past the
        # boundary, so if the network ever "cheats" the NLL by inflating sigma
        # to make the (target-mu)^2/sigma^2 term negligible, it gets stuck
        # there forever with no gradient left to learn mu or recover. sigmoid
        # keeps a nonzero gradient everywhere, so that trap can't form. Range
        # (-3, 3) -> sigma in [0.05, 20] dex, already generous for log(Mh)
        # spanning ~8-12; no reason to allow sigma=e^5~148 like before.
        # (In practice the v2 overnight run showed this bound can still be
        # SATURATED NEAR -- never literally reaching, but close enough that
        # its gradient is negligible -- if training is unstable for other
        # reasons; see the weight-clipping comment in HaloMassDataset above
        # and --grad-clip below, which address the actual instability rather
        # than widening this bound further.)
        log_sigma = -3.0 + 6.0 * torch.sigmoid(out[:, 1])
        return mu, log_sigma


def _n_features(only_angular: bool, summary_mode: bool) -> int:
    """Per-neighbor feature count for the CURRENT encoding flags -- mirrors
    env_to_array's own branching in train_nre.py exactly (summary_mode takes
    priority over only_angular there, so it does here too), so
    HaloMassSetNetwork always reshapes the flat input back into the same
    (MAX_NEIGHBORS, n_feat) layout env_to_array actually produced it in.
    """
    if summary_mode:
        return N_FEATURES_SUMM
    if only_angular:
        return N_FEATURES_ANG
    return N_FEATURES_FULL


# ---------------------------------------------------------------------------
# Model (Deep Sets variant) -- Zaheer et al. (2017), module docstring above.
# Same flat (neighbors, n_norm, theta) input as HaloMassNetwork (so the
# Dataset/DataLoader/CLI data-loading code above needs ZERO changes), but
# reshapes the neighbor block back into (MAX_NEIGHBORS, n_feat) rows and
# processes it as a genuine SET instead of one flattened MLP input.
# ---------------------------------------------------------------------------

class HaloMassSetNetwork(nn.Module):
    def __init__(self, max_neighbors: int, n_feat: int, phi_dims: list = [64, 64],
                 rho_hidden_dims: list = [256, 256, 256, 256], dropout: float = 0.1,
                 pooling: str = "sum"):
        super().__init__()
        self.max_neighbors = max_neighbors
        self.n_feat        = n_feat
        self.pooling       = pooling

        # phi: a small MLP applied INDEPENDENTLY and IDENTICALLY to every
        # neighbor row (nn.Linear/nn.LayerNorm broadcast over any number of
        # leading batch dims, so applying this Sequential to a
        # (batch, max_neighbors, n_feat) tensor runs the SAME weights on
        # every one of the max_neighbors rows -- this is what makes the
        # encoding permutation-invariant: swapping two neighbors' rows just
        # swaps which row gets which (identical) embedding, which sum-
        # pooling below doesn't even notice).
        phi_layers = []
        in_d = n_feat
        for d in phi_dims:
            phi_layers += [nn.Linear(in_d, d), nn.LayerNorm(d), nn.GELU()]
            in_d = d
        self.phi = nn.Sequential(*phi_layers)
        embed_dim = phi_dims[-1]

        # rho: identical ResidualBlock backbone to HaloMassNetwork, just
        # operating on [pooled_embedding, n_norm, theta] (width
        # embed_dim + 1 + N_PARAMS) instead of the old flattened
        # (MAX_NEIGHBORS*n_feat + 1 + N_PARAMS)-wide vector.
        rho_input_dim = embed_dim + 1 + N_PARAMS
        self.input_proj = nn.Sequential(
            nn.Linear(rho_input_dim, rho_hidden_dims[0]),
            nn.LayerNorm(rho_hidden_dims[0]),
            nn.GELU(),
        )
        blocks = []
        for i in range(len(rho_hidden_dims)):
            in_d2  = rho_hidden_dims[i]
            out_d2 = rho_hidden_dims[i + 1] if i + 1 < len(rho_hidden_dims) else rho_hidden_dims[i]
            if in_d2 == out_d2:
                blocks.append(ResidualBlock(in_d2, dropout))
            else:
                blocks.append(nn.Sequential(
                    nn.Linear(in_d2, out_d2), nn.LayerNorm(out_d2), nn.GELU(), nn.Dropout(dropout),
                ))
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Linear(rho_hidden_dims[-1], 2)  # (mu, pre-sigmoid log_sigma), same head as HaloMassNetwork
        with torch.no_grad():
            self.output.bias[1] = 0.0  # same sane sigma=1 start as HaloMassNetwork, see its comment

    def forward(self, x):
        n_flat = self.max_neighbors * self.n_feat
        # Undo __getitem__'s flattening: x is [flat_neighbors, n_norm, theta]
        # concatenated into one vector (see HaloMassDataset.__getitem__) --
        # slice it back into its three original pieces by position.
        neighbor_flat = x[:, :n_flat]
        n_norm        = x[:, n_flat:n_flat + 1]
        theta         = x[:, n_flat + 1:]
        neighbors = neighbor_flat.view(-1, self.max_neighbors, self.n_feat)

        # A padding row is one env_to_array left all-zero (every real row
        # has at least a nonzero MUV -- these are magnitudes, never
        # literally 0 in this dataset -- so "all features exactly zero" is
        # an unambiguous padding marker, not a real neighbor coincidentally
        # landing on zero). This needs no information beyond x itself --
        # no Dataset/collate change, no separate mask tensor to plumb through.
        mask = (neighbors.abs().sum(dim=-1, keepdim=True) > 0).float()  # (batch, max_neighbors, 1)

        embeddings = self.phi(neighbors) * mask  # zero out whatever phi(0) produces for padding rows
        if self.pooling == "mean":
            # Mean over REAL neighbors only (mask.sum == true neighbor count,
            # clamped to >=1 defensively -- HaloMassDataset already drops
            # any environment with zero real neighbors, so this never
            # actually divides by zero in practice).
            pooled = embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            # Sum pooling (Deep Sets' original formulation, module
            # docstring above) -- unlike mean, this lets the pooled vector's
            # MAGNITUDE itself encode how many neighbors there are, which is
            # otherwise only available through the separate n_norm feature.
            pooled = embeddings.sum(dim=1)

        combined = torch.cat([pooled, n_norm, theta], dim=-1)
        h = self.input_proj(combined)
        for block in self.blocks:
            h = block(h)
        out = self.output(h)
        mu = out[:, 0]
        log_sigma = -3.0 + 6.0 * torch.sigmoid(out[:, 1])  # same smooth bound as HaloMassNetwork, see its comment
        return mu, log_sigma


def gaussian_nll(mu, log_sigma, target, weight, beta=0.0):
    # Negative log-likelihood of `target` under a Normal(mu, sigma^2), per
    # example, then combined into one WEIGHTED MEAN loss across the batch.
    # The Normal log-density is
    #     log N(target; mu, sigma^2)
    #       = -log(sigma) - 0.5*log(2*pi) - 0.5*((target-mu)/sigma)^2
    # so the per-example NEGATIVE log-likelihood (what we want to MINIMIZE)
    # is the sign-flip of that: +log(sigma) + 0.5*log(2*pi) + 0.5*(...)^2.
    # Since the network outputs log_sigma directly (not sigma), log(sigma)
    # IS log_sigma, so no extra log() call is needed for that term -- only
    # the quadratic term needs sigma = exp(log_sigma) explicitly.
    sigma = torch.exp(log_sigma)
    nll = log_sigma + 0.5 * math.log(2 * math.pi) + 0.5 * ((target - mu) / sigma) ** 2

    if beta > 0:
        # beta-NLL (Seitzer, Tavakoli, Kirchhof & Archambeau 2022, "On the
        # Pitfalls of Heteroscedastic Uncertainty Estimation with
        # Probabilistic Neural Networks," https://arxiv.org/abs/2203.09168;
        # see the module docstring above). Plain NLL's gradient w.r.t. mu is
        # proportional to 1/sigma^2 -- so if sigma drifts up even slightly
        # (noise, an unlucky batch), mu's gradient shrinks, mu stops
        # improving, and the only way left to reduce the loss is inflating
        # sigma further, which shrinks mu's gradient more: a runaway
        # positive-feedback loop straight into "sigma eats the loss, mu
        # never learns" -- exactly the v2/v4 collapse (loss frozen at
        # log_sigma pegged near its ceiling by epoch ~10), which happened
        # under TWO very different reweighting schemes, ruling out
        # reweighting itself as the mechanism.
        #
        # The fix: multiply each example's NLL by sigma.detach()**(2*beta)
        # -- a per-example scalar that tracks the network's CURRENT sigma
        # for that example, but contributes NO gradient of its own (.detach()
        # stops autograd from tracing back through it). Differentiating the
        # now-weighted loss w.r.t. mu gives a term proportional to
        # sigma^(2*beta) * (target-mu)/sigma^2 = (target-mu) * sigma^(2*beta-2):
        # beta=0 recovers plain NLL (the 1/sigma^2 feedback problem,
        # unchanged); beta=1 makes mu's gradient fully INDEPENDENT of sigma
        # (pure MSE-like pull toward target, sigma can't starve it at all);
        # beta=0.5 (this script's default) is the paper's general-purpose
        # middle ground -- gradient ~ 1/sigma instead of 1/sigma^2, enough
        # to break the runaway feedback while still training a genuinely
        # heteroscedastic sigma. sigma's OWN gradient (from differentiating
        # w.r.t. log_sigma) still flows normally through the unweighted nll
        # expression above, just scaled by this same frozen per-example
        # factor -- beta-NLL does not stop sigma from learning, only
        # decouples how hard IT'S PULLED from mu's learning signal.
        nll = nll * (sigma.detach() ** (2 * beta))

    # Weighted average (not weighted sum): dividing by weight.sum() instead
    # of by batch size keeps the loss on the same natural scale as an
    # unweighted mean NLL regardless of how large/small the weights are
    # (e.g. after the mass-reweighting/clipping above), which is what makes
    # a single shared --lr sensible whether or not reweighting is enabled.
    # (weight here is the mass/catalog-count example weight from
    # HaloMassDataset, independent of and multiplicatively combined with the
    # beta-NLL factor above -- the two address different problems and don't
    # interact.)
    return (nll * weight).sum() / weight.sum()


# ---------------------------------------------------------------------------
# Train / val epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, grad_clip=5.0, beta=0.0):
    model.train()  # enables Dropout + (if present) BatchNorm's training-mode behavior; LayerNorm is unaffected
    total = 0.0
    for x, target, weight in loader:
        # DataLoader yields CPU tensors regardless of `device`; move each
        # batch onto the GPU (or leave on CPU) right before use.
        x, target, weight = x.to(device), target.to(device), weight.to(device)
        # Zero out gradients from the PREVIOUS batch -- PyTorch accumulates
        # gradients into .grad by default (useful for gradient accumulation
        # across multiple batches, not used here), so this must be called
        # every batch or gradients would silently keep adding up.
        optimizer.zero_grad()
        mu, log_sigma = model(x)
        loss = gaussian_nll(mu, log_sigma, target, weight, beta=beta)
        # Backpropagation: populates every parameter's .grad with
        # d(loss)/d(parameter), via autograd walking the computation graph
        # built while computing `loss` above.
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            # Rescales ALL parameters' gradients together (treated as one
            # long vector) so their combined L2 norm never exceeds
            # grad_clip, preserving direction -- caps how large a single
            # optimizer step can be regardless of why a particular batch's
            # loss (and therefore gradient) happened to be unusually large.
            # See the module docstring above for the Pascanu et al. (2013)
            # reference and the PyTorch docs link.
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # Apply the (possibly clipped) gradients: one Adam update step.
        optimizer.step()
        total += loss.item()  # .item() pulls the scalar loss value off the GPU/graph, for plain-Python accumulation
    return total / len(loader)  # mean loss across batches (NOT across individual examples, if the last batch is a partial one)


def val_epoch(model, loader, device):
    model.eval()  # disables Dropout (and switches any BatchNorm to eval-mode running statistics; irrelevant here, no BatchNorm is used)
    total = 0.0
    with torch.no_grad():
        # Disables autograd graph construction entirely for this block --
        # correct and faster/lower-memory for evaluation, since no
        # .backward() will ever be called on these losses.
        for x, target, weight in loader:
            x, target, weight = x.to(device), target.to(device), weight.to(device)
            mu, log_sigma = model(x)
            # Deliberately NOT passing beta here (defaults to 0.0, plain
            # NLL): beta-NLL reshapes the TRAINING objective to fix an
            # optimization pathology, but val_loss should stay the honest,
            # beta-independent likelihood -- otherwise "best_val" in main()
            # would mean something different for every --beta-nll value,
            # and best-checkpoint selection across runs wouldn't be
            # comparable.
            total += gaussian_nll(mu, log_sigma, target, weight).item()
    return total / len(loader)


def log_per_bin_metrics(model, loader, device, n_bins=6):
    # Diagnostic for whether the tails (sparsest true-logMh regions, where the
    # eval plot showed shrinkage-to-the-mean) are actually improving, rather
    # than being masked by the dense middle band dominating the mean val loss.
    model.eval()
    targets, mus, sigmas = [], [], []
    with torch.no_grad():
        for x, target, weight in loader:
            # Note: `target` is deliberately NOT moved to `device` here --
            # it's only ever compared against mu/sigmas after both are
            # pulled back to CPU numpy arrays below, so there's no need.
            x = x.to(device)
            mu, log_sigma = model(x)
            targets.append(target.numpy())
            mus.append(mu.cpu().numpy())
            sigmas.append(torch.exp(log_sigma).cpu().numpy())
    # Concatenate the per-batch lists into single flat arrays covering the
    # WHOLE loader (typically the validation set) at once.
    targets = np.concatenate(targets)
    mus     = np.concatenate(mus)
    sigmas  = np.concatenate(sigmas)

    # Same equal-width-binning pattern as the mass-reweighting block in
    # HaloMassDataset above, but over the VALIDATION set's true values, and
    # purely for logging/diagnostics -- it does not feed back into training.
    edges = np.linspace(targets.min(), targets.max() + 1e-6, n_bins + 1)
    bin_idx = np.clip(np.digitize(targets, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        # Root-mean-square error of the point estimate mu within this bin.
        rms = np.sqrt(np.mean((targets[mask] - mus[mask]) ** 2))
        # Signed mean residual: positive means the model systematically
        # OVER-predicts log Mh in this bin, negative means it UNDER-predicts
        # -- this is what directly diagnoses shrinkage-to-the-mean (expect
        # positive bias in the low-mass tail, negative in the high-mass
        # tail, if that failure mode is present).
        bias = np.mean(mus[mask] - targets[mask])
        # Fraction of this bin's examples where the true value falls within
        # the model's own predicted 1-sigma band -- should be close to 68%
        # (the area under a Gaussian within +/-1 standard deviation) if the
        # predicted uncertainties are well CALIBRATED. A value near 100%
        # means sigma is being over-predicted (too conservative/uninformative);
        # a value well below 68% means sigma is under-predicted (overconfident).
        cov68 = np.mean(np.abs(targets[mask] - mus[mask]) < sigmas[mask])
        log.info(f"    bin [{edges[b]:.2f},{edges[b+1]:.2f})  n={mask.sum():5d}  "
                  f"rms={rms:.3f}  bias={bias:+.3f}  cov68={cov68:.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # One or more existing build_nre_database.py/patch_bright_logmh.py output
    # directories (nargs='+' -> at least one path, collected into a list);
    # this is how the 4-box multibox training set is assembled -- pass all
    # 4 seeds' directories at once.
    p.add_argument("--database-dir", type=Path, nargs='+', required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    # Fraction of ALL loaded (environment, log Mh) pairs held out for
    # validation (see torch.utils.data.random_split in main()); the
    # remainder is the training split.
    p.add_argument("--val-frac", type=float, default=0.2)
    # One width per residual stage; e.g. the default [256,256,256,256] gives
    # a 4-block network at constant width 256. Passing e.g.
    # "512 512 512 512 512 512" builds a wider (512) AND deeper (6-block) network.
    p.add_argument("--hidden-dims", type=int, nargs='+', default=[256, 256, 256, 256],
                   help="Residual-stack widths for rho (the post-pooling backbone in "
                        "--architecture deepset, or the whole network in --architecture mlp).")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--architecture", choices=["mlp", "deepset"], default="mlp",
                   help="'mlp' (default, unchanged from v1-v7): flatten the distance-sorted, "
                        "zero-padded neighbor list into one vector, feed it straight to the "
                        "ResidualBlock stack. 'deepset': Zaheer et al. 2017 (module docstring) -- "
                        "a shared per-neighbor MLP (--phi-dims) + masked --pooling into one "
                        "fixed-size embedding, THEN the same ResidualBlock stack. Permutation-"
                        "invariant by construction; targets the shrinkage-to-the-mean bias that "
                        "no loss-function fix (LDS, beta-NLL, early stopping) has moved.")
    p.add_argument("--phi-dims", type=int, nargs='+', default=[64, 64],
                   help="--architecture deepset only: widths of the shared per-neighbor "
                        "embedding MLP (phi). Last value is the pooled embedding's dimension.")
    p.add_argument("--pooling", choices=["sum", "mean"], default="sum",
                   help="--architecture deepset only: how per-neighbor embeddings are combined. "
                        "'sum' (Deep Sets' original formulation) lets the pooled vector's "
                        "magnitude itself encode neighbor count; 'mean' normalizes that out.")
    # Per-catalog environment subsampling cap -- see the --max-per-catalog
    # use inside HaloMassDataset.__init__ above. 0 (or any non-positive
    # value) disables the cap, using every environment in every catalog.
    p.add_argument("--max-per-catalog", type=int, default=200)
    p.add_argument("--weight-by-catalog-count", action="store_true")
    p.add_argument("--only-angular", action="store_true")
    p.add_argument("--summary-mode", action="store_true")
    p.add_argument("--reweight-by-mass", action="store_true",
                   help="Label-Distribution-Smoothing (LDS) inverse-density reweight the loss "
                        "by true log(Mh) bin, to counter shrinkage-to-the-mean in the sparse "
                        "tails seen in eval_vs_truth.pdf. See Yang et al. 2021 (module docstring).")
    p.add_argument("--reweight-alpha", type=float, default=0.5,
                   help="Exponent on 1/smoothed-density(bin); 1.0 = fully flatten, 0.0 = no reweighting.")
    p.add_argument("--reweight-bins", type=int, default=30)
    p.add_argument("--lds-kernel-sigma", type=float, default=2.0,
                   help="Gaussian kernel std, in BINS, used to smooth the per-bin example-count "
                        "histogram before computing inverse-density weights (the actual LDS step "
                        "from Yang et al. 2021). 0 disables smoothing (raw per-bin counts, the "
                        "v3 behavior that needed aggressive --max-weight-ratio clipping).")
    p.add_argument("--uvlf-reweight", action="store_true",
                   help="Reweight the training loss by each example's parent theta's UVLF "
                        "plausibility (continuous Gaussian taper in equivalent-sigma units, "
                        "--uvlf-sigma-scale -- see the long comment above this block in "
                        "HaloMassDataset). Requires hmf and scipy (only imported if this flag "
                        "is set). Motivated by plot_eval_vs_truth_uvlf_consistent.py showing a "
                        "residual bias even when evaluating only on UVLF-consistent theta.")
    p.add_argument("--uvlf-sigma-scale", type=float, default=5.0,
                   help="Gaussian taper width, in equivalent UVLF sigma units, for "
                        "--uvlf-reweight (e.g. theta at this many sigma from the observed UVLF "
                        "get weight ~exp(-0.5)=0.61; much more at 2-3x this). 5.0 matches the "
                        "'broad, a few sigma' plausibility range already used elsewhere in this "
                        "project (plot_eval_vs_truth_uvlf_consistent.py's --max-sigma default).")
    p.add_argument("--max-weight-ratio", type=float, default=10.0,
                   help="Clip the combined (mass x catalog-count x UVLF) per-example weight to "
                        "[median/ratio, median*ratio], to stop a few extreme-weight examples "
                        "from dominating a batch's gradient and collapsing sigma to its "
                        "ceiling. 0 or negative disables clipping.")
    p.add_argument("--grad-clip", type=float, default=5.0,
                   help="Max gradient norm (0 disables). Cheap insurance against the same "
                        "instability clipped example weights target.")
    p.add_argument("--beta-nll", type=float, default=0.5,
                   help="beta-NLL exponent (Seitzer et al. 2022, module docstring): multiplies "
                        "each example's NLL by sigma.detach()**(2*beta) during TRAINING ONLY "
                        "(val_loss always uses plain NLL, beta=0). 0.0 = plain NLL (the v2/v4 "
                        "collapse-prone loss); 1.0 = mu's gradient fully decoupled from sigma.")
    p.add_argument("--diag-every", type=int, default=10,
                   help="Log per-mass-bin val RMS/bias/coverage every N epochs (0 to disable).")
    p.add_argument("--diag-bins", type=int, default=6)
    p.add_argument("--early-stop-patience", type=int, default=30,
                   help="Stop training if val loss hasn't improved for this many consecutive "
                        "epochs (0 disables early stopping, training always runs the full "
                        "--epochs). The v5 run's val loss bottomed out around epoch ~100 then "
                        "rose to 30x that by epoch 300 (pure overfitting) -- this stops that waste "
                        "automatically. halo_mass_best.pt is always the checkpoint to use either way.")
    p.add_argument("--early-stop-min-delta", type=float, default=0.0,
                   help="Minimum val-loss decrease to count as an improvement for "
                        "--early-stop-patience (0.0 = any decrease at all counts).")
    p.add_argument("--early-stop-warmup", type=int, default=70,
                   help="Epochs before the early-stopping patience counter starts (best_val/"
                        "checkpointing still run every epoch regardless). Needed because escaping "
                        "the early high-sigma, barely-trained regime causes a TEMPORARY val-loss "
                        "spike (plain NLL scores a barely-trained high-sigma model deceptively well) "
                        "before real convergence -- v5/v6 both show this spike persisting past "
                        "epoch 40-50; without a warmup, patience can exhaust DURING the spike and "
                        "save the worst checkpoint in the run (see the long comment in main()).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    # parents=True: create any missing intermediate directories too.
    # exist_ok=True: don't error if --output-dir already exists (e.g. a
    # previous, now-superseded run) -- its contents just get overwritten below.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Seed BOTH PyTorch's and NumPy's global random state. torch.manual_seed
    # fixes model weight initialization and anything else that draws from
    # torch's RNG (e.g. DataLoader's shuffle=True ordering below, dropout
    # masks, random_split's split assignment). np.random.seed fixes NumPy's
    # global RNG, used by the --max-per-catalog subsampling (np.random.choice)
    # and the augmentation shift (np.random.uniform) inside HaloMassDataset
    # above. This makes a given --seed reproducible on the SAME hardware/
    # library-version combination, but is not a strict cross-platform
    # reproducibility guarantee (e.g. GPU nondeterminism in some ops is not
    # separately disabled here).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Prefer GPU if one is visible to this process (e.g. via CUDA_VISIBLE_DEVICES
    # or a SLURM --gres=gpu:1 allocation); fall back to CPU otherwise.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Normalization bounds: computed the same way train_nre.py does, across
    # all database files' params (theta grid), NOT across log(Mh) -- the
    # target (log Mh) is left in its natural units, only theta is normalized
    # to [-1, 1] for network input, matching the NRE scripts' convention.
    log.info("Computing parameter normalization ...")
    all_params = []
    for db_dir in args.database_dir:
        for path in sorted(Path(db_dir).glob("nre_*.npz")):
            try:
                # Only reads the small 'params' array out of each file here
                # (a cheap pass just to find the overall min/max across the
                # whole theta grid) -- the full, expensive load (coords,
                # offsets, bright_logmh) happens again, separately, inside
                # HaloMassDataset.__init__ below. This means every database
                # file gets opened twice over the course of one training run.
                all_params.append(np.load(path)['params'])
            except Exception:
                # Deliberately broad: any file unreadable here will also be
                # unreadable (and skipped, with a logged reason) in
                # HaloMassDataset.__init__ below, so there's no need to
                # duplicate that error handling/reporting here.
                continue
    all_params = np.array(all_params)
    param_min = all_params.min(axis=0)  # elementwise min across the 3 theta components
    param_max = all_params.max(axis=0)
    log.info(f"  param_min: {param_min}")
    log.info(f"  param_max: {param_max}")

    dataset = HaloMassDataset(
        database_dirs=args.database_dir,
        param_min=param_min, param_max=param_max,
        max_per_catalog=args.max_per_catalog,
        summary_mode=args.summary_mode, only_angular=args.only_angular,
        weight_by_catalog_count=args.weight_by_catalog_count,
        reweight_by_mass=args.reweight_by_mass,
        reweight_alpha=args.reweight_alpha,
        reweight_bins=args.reweight_bins,
        lds_kernel_sigma=args.lds_kernel_sigma,
        uvlf_reweight=args.uvlf_reweight,
        uvlf_sigma_scale=args.uvlf_sigma_scale,
        max_weight_ratio=args.max_weight_ratio,
    )

    # Split the single HaloMassDataset instance into disjoint train/val
    # INDEX subsets (random_split returns lightweight Subset wrappers, not
    # copies of the underlying data) -- generator=... with a fixed seed
    # makes the split itself reproducible independent of any other random
    # draws that happened before/after it.
    n_val = int(len(dataset) * args.val_frac)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    log.info(f"Train: {n_train}  Val: {n_val}")

    # shuffle=True for training (a fresh random batch order every epoch,
    # standard SGD practice); shuffle=False for validation (order doesn't
    # matter for computing a mean loss, and keeping it fixed makes manual
    # debugging/inspection runs reproducible example-for-example).
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # Peek at one example purely to read off the network's required input
    # width (depends on --only-angular/--summary-mode, via env_to_array's
    # differing output sizes) -- cheaper than recomputing it from the CLI
    # flags by hand, and guaranteed to match whatever __getitem__ actually produces.
    sample_x, _, _ = dataset[0]
    input_dim = sample_x.shape[0]
    log.info(f"Input dim: {input_dim}")

    if args.architecture == "deepset":
        n_feat = _n_features(args.only_angular, args.summary_mode)
        model = HaloMassSetNetwork(MAX_NEIGHBORS, n_feat, args.phi_dims, args.hidden_dims,
                                    args.dropout, pooling=args.pooling).to(device)
        log.info(f"Architecture: deepset  (n_feat={n_feat}, phi_dims={args.phi_dims}, "
                  f"pooling={args.pooling})")
    else:
        model = HaloMassNetwork(input_dim, args.hidden_dims, args.dropout).to(device)
        log.info("Architecture: mlp")
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Single cosine decay from args.lr down toward 0 over the full
    # args.epochs run (T_max=args.epochs means one full half-cosine-period
    # cycle spans the whole training run; see the Loshchilov & Hutter (2017)
    # reference in the module docstring above -- no warm restarts are used,
    # despite "SGDR" being named for them).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float('inf')
    # Epochs since the last val improvement -- the early-stopping counter.
    # The v5 run showed WHY early stopping matters: best val (0.3346) was
    # reached around epoch ~100, then val loss climbed to 11.06 by epoch 300
    # while train loss kept dropping -- 200 epochs of pure overfitting.
    #
    # But the FIRST early-stopping attempt (v6) immediately exposed a sharper
    # bug: it stopped at epoch 41 with "best val" still 3.9191 -- the exact
    # plain-NLL collapse floor from the pre-beta-NLL runs (log_sigma pegged
    # near its sigmoid ceiling). beta-NLL WAS working (train loss was
    # dropping normally, nothing like v2/v4's frozen 3.9190 the whole run),
    # but escaping that early high-sigma, barely-trained regime causes a
    # TEMPORARY SPIKE in val_loss (deliberately plain NLL, beta=0, for
    # comparability across --beta-nll settings -- see val_epoch above) as mu
    # starts moving and its errors are, for a while, scored against a sigma
    # that hasn't re-calibrated yet. v5 shows the same spike at epoch 50
    # (val=25.88) before recovering to val=0.53 by epoch 100. A plain
    # patience counter has no way to tell "still stuck" apart from "mid-
    # recovery from a known early transient" -- 30 epochs of a worse-looking
    # val loss reads identically to both, so it fired right as real learning
    # was getting started and saved the single WORST checkpoint in the run.
    #
    # Fix: args.early_stop_warmup epochs where the patience COUNTER doesn't
    # run at all (best_val/checkpointing still happen normally every epoch,
    # so there's always a saved model even if the job dies mid-warmup) --
    # only once epoch > warmup does "no improvement" start being held
    # against the run, giving it a full fresh --early-stop-patience window
    # AFTER the known escape-transient, not overlapping with it.
    epochs_since_improvement = 0
    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device,
                                  grad_clip=args.grad_clip, beta=args.beta_nll)
        val_loss   = val_epoch(model, val_loader, device)
        # Advance the LR schedule by one step (= one epoch here, since it's
        # called once per epoch rather than once per batch).
        scheduler.step()
        log.info(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")
        # args.early_stop_min_delta guards against resetting the patience
        # counter on a change too small to be a real improvement (float
        # noise between near-identical epochs) -- 0.0 (the default) means
        # any decrease at all counts.
        improved = val_loss < best_val - args.early_stop_min_delta
        if improved:
            # Checkpoint ONLY the single best-val-loss model seen so far
            # (overwriting halo_mass_best.pt each time a new best is found)
            # -- tracked EVERY epoch, warmup or not (see comment above), so
            # there's always a checkpoint on disk; it's the patience COUNTER
            # below, not this save, that's gated by warmup.
            best_val = val_loss
            torch.save(model.state_dict(), args.output_dir / "halo_mass_best.pt")
            log.info(f"  -> New best model saved (val={val_loss:.4f})")
        if args.diag_every > 0 and epoch % args.diag_every == 0:
            log_per_bin_metrics(model, val_loader, device, n_bins=args.diag_bins)
        if epoch > args.early_stop_warmup:
            epochs_since_improvement = 0 if improved else epochs_since_improvement + 1
            if args.early_stop_patience > 0 and epochs_since_improvement >= args.early_stop_patience:
                log.info(f"Early stopping at epoch {epoch}: no val improvement for "
                          f"{epochs_since_improvement} epochs since the {args.early_stop_warmup}-epoch "
                          f"warmup ended (--early-stop-patience={args.early_stop_patience}). Best val "
                          f"{best_val:.4f} is in halo_mass_best.pt; note the LR schedule was set up for "
                          f"the full {args.epochs} epochs and never finished its cosine decay -- this "
                          f"only affects halo_mass_final.pt below, not halo_mass_best.pt, which was "
                          f"already saved under whatever LR was active when it was reached.")
                stopped_early = True
                break

    # Also save the FINAL epoch's weights (separate from the best-val
    # checkpoint above, and -- with early stopping on -- typically somewhat
    # WORSE than it, by construction: stopping is triggered by a run of
    # epochs with no improvement, so the final epoch is inside that run, not
    # at its start), plus everything needed to reconstruct the model and
    # reproduce its input preprocessing later without re-deriving anything
    # from the training database (used by evaluate_halo_mass_model.py):
    #   model_config.npz   -- architecture hyperparameters (HaloMassNetwork's
    #                         constructor arguments) and which env_to_array
    #                         feature layout was used.
    #   normalization.npz  -- the theta [-1,1] rescaling bounds, so inference-
    #                         time inputs get normalized identically to
    #                         training-time inputs.
    torch.save(model.state_dict(), args.output_dir / "halo_mass_final.pt")
    # architecture/phi_dims/pooling are saved unconditionally (even for the
    # default --architecture mlp, where they're unused) so
    # evaluate_halo_mass_model.py has one uniform loading path instead of
    # needing to guess an older checkpoint's architecture from which fields
    # happen to be present.
    np.savez(args.output_dir / "model_config.npz",
             hidden_dims=np.array(args.hidden_dims),
             dropout=args.dropout,
             input_dim=input_dim,
             only_angular=int(args.only_angular),
             summary_mode=int(args.summary_mode),
             architecture=args.architecture,
             phi_dims=np.array(args.phi_dims),
             pooling=args.pooling,
             max_neighbors=MAX_NEIGHBORS)
    np.savez(args.output_dir / "normalization.npz", param_min=param_min, param_max=param_max)
    log.info(f"Training {'stopped early' if stopped_early else 'complete'}. Best val loss: {best_val:.4f}")
    log.info(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
