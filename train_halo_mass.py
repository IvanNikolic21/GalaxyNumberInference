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
"""
import argparse
import logging
import math
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_nre import (
    env_to_array, normalize_params, ResidualBlock,
    MAX_NEIGHBORS, N_FEATURES_FULL,
)

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
        self.param_min    = param_min
        self.param_max    = param_max
        self.augment      = augment
        self.summary_mode = summary_mode
        self.only_angular = only_angular

        self.envs      = []
        self.params    = []
        self.logmh     = []
        self.weights   = []

        cat_idx = 0
        for db_dir in database_dirs:
            files = sorted(Path(db_dir).glob("nre_*.npz"))
            log.info(f"Loading {len(files)} files from {db_dir} ...")

            for path in files:
                try:
                    data    = np.load(path)
                    coords  = data['coords']
                    offsets = data['offsets']
                    params  = data['params']
                    if 'bright_logmh' not in data.files:
                        print(f"Skipping file with no bright_logmh (not yet patched): {path}")
                        continue
                    bright_logmh = data['bright_logmh']
                except (EOFError, ValueError, OSError, KeyError, zipfile.BadZipFile) as e:
                    print(f"Skipping corrupted file: {path} ({e})")
                    continue

                n_envs = len(offsets) - 1
                if len(bright_logmh) != n_envs:
                    print(f"Skipping file with mismatched bright_logmh length: {path} "
                          f"({len(bright_logmh)} vs {n_envs} offsets)")
                    continue

                indices = np.arange(n_envs)
                if max_per_catalog is not None and max_per_catalog > 0 and len(indices) > max_per_catalog:
                    indices = np.random.choice(indices, max_per_catalog, replace=False)

                n_used = len(indices)
                weight = (1.0 / n_used) if (weight_by_catalog_count and n_used > 0) else 1.0

                for i in indices:
                    env = coords[offsets[i]:offsets[i + 1]]
                    if len(env) == 0:
                        continue
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
            edges = np.linspace(logmh_arr.min(), logmh_arr.max() + 1e-6, reweight_bins + 1)
            bin_idx = np.clip(np.digitize(logmh_arr, edges) - 1, 0, reweight_bins - 1)
            counts = np.bincount(bin_idx, minlength=reweight_bins)
            mass_weight = 1.0 / np.maximum(counts[bin_idx], 1) ** reweight_alpha
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
            # removing the reweighting itself.
            w = np.array(self.weights)
            median_w = np.median(w)
            lo, hi = median_w / max_weight_ratio, median_w * max_weight_ratio
            n_clipped = int(np.sum((w < lo) | (w > hi)))
            w = np.clip(w, lo, hi)
            self.weights = w.tolist()
            log.info(f"Clipped combined example weights to [{lo:.4g}, {hi:.4g}] "
                      f"(ratio={max_weight_ratio}x median); {n_clipped}/{len(w)} examples clipped")

    def __len__(self):
        return len(self.envs)

    def __getitem__(self, idx):
        env    = self.envs[idx]
        params = self.params[idx]

        flat, n_norm, n, dists = env_to_array(env, summary_mode=self.summary_mode,
                                               only_angular=self.only_angular)

        if self.augment and n > 0 and not self.summary_mode and not self.only_angular:
            shift = np.random.uniform(-5.0, 5.0, size=3).astype(np.float32)
            flat_2d = flat.reshape(MAX_NEIGHBORS, N_FEATURES_FULL)
            flat_2d[:n, :3] += shift
            flat = flat_2d.flatten()

        x = torch.from_numpy(np.concatenate([flat, n_norm]))
        theta = torch.from_numpy(normalize_params(params, self.param_min, self.param_max))
        x_in = torch.cat([x, theta])

        target = torch.tensor(self.logmh[idx], dtype=torch.float32)
        weight = torch.tensor(self.weights[idx], dtype=torch.float32)
        return x_in, target, weight


# ---------------------------------------------------------------------------
# Model -- same ResidualBlock backbone as NRENetwork, 2-output (mu, log_sigma) head
# ---------------------------------------------------------------------------

class HaloMassNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [256, 256, 256, 256], dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
        )
        blocks = []
        for i in range(len(hidden_dims)):
            in_d  = hidden_dims[i]
            out_d = hidden_dims[i + 1] if i + 1 < len(hidden_dims) else hidden_dims[i]
            if in_d == out_d:
                blocks.append(ResidualBlock(in_d, dropout))
            else:
                blocks.append(nn.Sequential(
                    nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.GELU(), nn.Dropout(dropout),
                ))
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Linear(hidden_dims[-1], 2)  # (mu, pre-sigmoid log_sigma)
        with torch.no_grad():
            # Start log_sigma's raw output at 0 -> sigmoid(0)=0.5 -> log_sigma=0 (sigma=1)
            # initially, a sane starting uncertainty rather than drifting toward the
            # edge of the allowed range right out of the gate.
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
        log_sigma = -3.0 + 6.0 * torch.sigmoid(out[:, 1])
        return mu, log_sigma


def gaussian_nll(mu, log_sigma, target, weight):
    sigma = torch.exp(log_sigma)
    nll = log_sigma + 0.5 * math.log(2 * math.pi) + 0.5 * ((target - mu) / sigma) ** 2
    return (nll * weight).sum() / weight.sum()


# ---------------------------------------------------------------------------
# Train / val epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, grad_clip=5.0):
    model.train()
    total = 0.0
    for x, target, weight in loader:
        x, target, weight = x.to(device), target.to(device), weight.to(device)
        optimizer.zero_grad()
        mu, log_sigma = model(x)
        loss = gaussian_nll(mu, log_sigma, target, weight)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def val_epoch(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
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
            x = x.to(device)
            mu, log_sigma = model(x)
            targets.append(target.numpy())
            mus.append(mu.cpu().numpy())
            sigmas.append(torch.exp(log_sigma).cpu().numpy())
    targets = np.concatenate(targets)
    mus     = np.concatenate(mus)
    sigmas  = np.concatenate(sigmas)

    edges = np.linspace(targets.min(), targets.max() + 1e-6, n_bins + 1)
    bin_idx = np.clip(np.digitize(targets, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rms = np.sqrt(np.mean((targets[mask] - mus[mask]) ** 2))
        bias = np.mean(mus[mask] - targets[mask])
        cov68 = np.mean(np.abs(targets[mask] - mus[mask]) < sigmas[mask])
        log.info(f"    bin [{edges[b]:.2f},{edges[b+1]:.2f})  n={mask.sum():5d}  "
                  f"rms={rms:.3f}  bias={bias:+.3f}  cov68={cov68:.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database-dir", type=Path, nargs='+', required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--hidden-dims", type=int, nargs='+', default=[256, 256, 256, 256])
    p.add_argument("--dropout", type=float, default=0.1)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
                all_params.append(np.load(path)['params'])
            except Exception:
                continue
    all_params = np.array(all_params)
    param_min = all_params.min(axis=0)
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

    n_val = int(len(dataset) * args.val_frac)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    log.info(f"Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    sample_x, _, _ = dataset[0]
    input_dim = sample_x.shape[0]
    log.info(f"Input dim: {input_dim}")

    model = HaloMassNetwork(input_dim, args.hidden_dims, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, grad_clip=args.grad_clip)
        val_loss   = val_epoch(model, val_loader, device)
        scheduler.step()
        log.info(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.output_dir / "halo_mass_best.pt")
            log.info(f"  -> New best model saved (val={val_loss:.4f})")
        if args.diag_every > 0 and epoch % args.diag_every == 0:
            log_per_bin_metrics(model, val_loader, device, n_bins=args.diag_bins)

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
