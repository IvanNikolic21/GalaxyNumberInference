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

--reweight-by-mass inverse-density-weights the loss by true log(Mh) bin, to
counter the shrinkage-to-the-mean seen in both tails of eval_vs_truth.pdf.
--diag-every logs per-mass-bin val RMS/bias/68%-coverage periodically, so tail
convergence can be checked directly instead of inferring it from the (middle-
band-dominated) mean val loss alone.
--max-weight-ratio and --grad-clip guard against the failure mode discovered
when --reweight-by-mass was first combined with --weight-by-catalog-count:
see the long comment above the weight-clipping block in HaloMassDataset,
below, for the full story.
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
from train_nre import (
    env_to_array, normalize_params, ResidualBlock,
    MAX_NEIGHBORS, N_FEATURES_FULL,
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
        max_weight_ratio: float = 10.0,
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
            # Inverse-density reweighting: the predicted-vs-true plot showed
            # shrinkage-to-the-mean worst in the sparsest true-logMh regions
            # (both tails), because the dense middle band dominates the loss.
            # Bin true logMh and give each example a weight ~ 1/count(bin)^alpha,
            # softened by alpha<1 so empty-ish edge bins don't blow up to
            # enormous weights from one or two examples. Combined
            # (multiplicatively) with the existing per-catalog weight, then
            # renormalized to mean 1 so the overall loss scale is unchanged.
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
            # Number of examples landing in each of the reweight_bins bins.
            counts = np.bincount(bin_idx, minlength=reweight_bins)
            # Per-example weight ~ 1 / (that example's bin count)^alpha.
            # alpha=1 would fully flatten the mass distribution's
            # contribution to the loss (every bin contributes equally);
            # alpha=0 disables this term entirely (weight=1 for everyone);
            # the default 0.5 is a softened compromise between the two.
            # counts[bin_idx] maps each example back to ITS bin's count via
            # fancy indexing; np.maximum(..., 1) guards the (impossible
            # here, since every example's own bin necessarily has count>=1)
            # division-by-zero case.
            mass_weight = 1.0 / np.maximum(counts[bin_idx], 1) ** reweight_alpha
            # Renormalize so the mean weight is exactly 1 -- keeps the
            # overall magnitude of gaussian_nll's weighted average (see
            # gaussian_nll() below) from drifting as reweight_alpha/
            # reweight_bins change, which would otherwise also silently
            # change the effective learning rate.
            mass_weight = mass_weight / mass_weight.mean()
            self.weights = (np.array(self.weights) * mass_weight).tolist()
            log.info(f"Applied inverse-density mass reweighting "
                      f"(bins={reweight_bins}, alpha={reweight_alpha}); "
                      f"bin counts range [{counts.min()}, {counts.max()}]")

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
            median_w = np.median(w)
            lo, hi = median_w / max_weight_ratio, median_w * max_weight_ratio
            n_clipped = int(np.sum((w < lo) | (w > hi)))
            w = np.clip(w, lo, hi)
            self.weights = w.tolist()
            log.info(f"Clipped combined example weights to [{lo:.4g}, {hi:.4g}] "
                      f"(ratio={max_weight_ratio}x median); {n_clipped}/{len(w)} examples clipped")

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


def gaussian_nll(mu, log_sigma, target, weight):
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
    # Weighted average (not weighted sum): dividing by weight.sum() instead
    # of by batch size keeps the loss on the same natural scale as an
    # unweighted mean NLL regardless of how large/small the weights are
    # (e.g. after the mass-reweighting/clipping above), which is what makes
    # a single shared --lr sensible whether or not reweighting is enabled.
    return (nll * weight).sum() / weight.sum()


# ---------------------------------------------------------------------------
# Train / val epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, grad_clip=5.0):
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
        loss = gaussian_nll(mu, log_sigma, target, weight)
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
    p.add_argument("--hidden-dims", type=int, nargs='+', default=[256, 256, 256, 256])
    p.add_argument("--dropout", type=float, default=0.1)
    # Per-catalog environment subsampling cap -- see the --max-per-catalog
    # use inside HaloMassDataset.__init__ above. 0 (or any non-positive
    # value) disables the cap, using every environment in every catalog.
    p.add_argument("--max-per-catalog", type=int, default=200)
    p.add_argument("--weight-by-catalog-count", action="store_true")
    p.add_argument("--only-angular", action="store_true")
    p.add_argument("--summary-mode", action="store_true")
    p.add_argument("--reweight-by-mass", action="store_true",
                   help="Inverse-density reweight the loss by true log(Mh) bin, to counter "
                        "shrinkage-to-the-mean in the sparse tails seen in eval_vs_truth.pdf.")
    p.add_argument("--reweight-alpha", type=float, default=0.5,
                   help="Exponent on 1/count(bin); 1.0 = fully flatten, 0.0 = no reweighting.")
    p.add_argument("--reweight-bins", type=int, default=30)
    p.add_argument("--max-weight-ratio", type=float, default=10.0,
                   help="Clip the combined (mass x catalog-count) per-example weight to "
                        "[median/ratio, median*ratio], to stop a few extreme-weight examples "
                        "from dominating a batch's gradient and collapsing sigma to its "
                        "ceiling. 0 or negative disables clipping.")
    p.add_argument("--grad-clip", type=float, default=5.0,
                   help="Max gradient norm (0 disables). Cheap insurance against the same "
                        "instability clipped example weights target.")
    p.add_argument("--diag-every", type=int, default=10,
                   help="Log per-mass-bin val RMS/bias/coverage every N epochs (0 to disable).")
    p.add_argument("--diag-bins", type=int, default=6)
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

    model = HaloMassNetwork(input_dim, args.hidden_dims, args.dropout).to(device)
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
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, grad_clip=args.grad_clip)
        val_loss   = val_epoch(model, val_loader, device)
        # Advance the LR schedule by one step (= one epoch here, since it's
        # called once per epoch rather than once per batch).
        scheduler.step()
        log.info(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val:
            # Checkpoint ONLY the single best-val-loss model seen so far
            # (overwriting halo_mass_best.pt each time a new best is found)
            # -- standard early-stopping-adjacent practice: the final epoch's
            # weights (saved separately below, after the loop) aren't
            # necessarily the best if validation loss started increasing
            # (overfitting) partway through the run.
            best_val = val_loss
            torch.save(model.state_dict(), args.output_dir / "halo_mass_best.pt")
            log.info(f"  -> New best model saved (val={val_loss:.4f})")
        if args.diag_every > 0 and epoch % args.diag_every == 0:
            log_per_bin_metrics(model, val_loader, device, n_bins=args.diag_bins)

    # Also save the FINAL epoch's weights (separate from the best-val
    # checkpoint above), plus everything needed to reconstruct the model
    # and reproduce its input preprocessing later without re-deriving
    # anything from the training database (used by evaluate_halo_mass_model.py):
    #   model_config.npz   -- architecture hyperparameters (HaloMassNetwork's
    #                         constructor arguments) and which env_to_array
    #                         feature layout was used.
    #   normalization.npz  -- the theta [-1,1] rescaling bounds, so inference-
    #                         time inputs get normalized identically to
    #                         training-time inputs.
    torch.save(model.state_dict(), args.output_dir / "halo_mass_final.pt")
    np.savez(args.output_dir / "model_config.npz",
             hidden_dims=np.array(args.hidden_dims),
             dropout=args.dropout,
             input_dim=input_dim,
             only_angular=int(args.only_angular),
             summary_mode=int(args.summary_mode))
    np.savez(args.output_dir / "normalization.npz", param_min=param_min, param_max=param_max)
    log.info(f"Training complete. Best val loss: {best_val:.4f}")
    log.info(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
