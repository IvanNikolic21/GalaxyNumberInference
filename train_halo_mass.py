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
        --output-dir /groups/astro/ivannik/projects/Neighbors/halo_mass_model_multibox4
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
        self.output = nn.Linear(hidden_dims[-1], 2)  # (mu, log_sigma)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        out = self.output(h)
        mu, log_sigma = out[:, 0], out[:, 1].clamp(-5.0, 5.0)
        return mu, log_sigma


def gaussian_nll(mu, log_sigma, target, weight):
    sigma = torch.exp(log_sigma)
    nll = log_sigma + 0.5 * math.log(2 * math.pi) + 0.5 * ((target - mu) / sigma) ** 2
    return (nll * weight).sum() / weight.sum()


# ---------------------------------------------------------------------------
# Train / val epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for x, target, weight in loader:
        x, target, weight = x.to(device), target.to(device), weight.to(device)
        optimizer.zero_grad()
        mu, log_sigma = model(x)
        loss = gaussian_nll(mu, log_sigma, target, weight)
        loss.backward()
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
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss   = val_epoch(model, val_loader, device)
        scheduler.step()
        log.info(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.output_dir / "halo_mass_best.pt")
            log.info(f"  -> New best model saved (val={val_loss:.4f})")

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
