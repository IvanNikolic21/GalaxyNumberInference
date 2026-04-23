#!/usr/bin/env python
"""
train_nre_d1.py
---------------
Minimal NRE using only the nearest neighbor summary:
    x = (d1, MUV_nearest, n_neighbors_total)  — 3 features + 3 params = 6 inputs

d1 is the distance to the nearest faint neighbor (same as our d1 statistic).
This is the simplest possible summary that still carries physical information.

Usage
-----
    python train_nre_d1.py --database-dir /path/to/nre_database \
                           --prior-database-dir /path/to/nre_database_prior
"""

import argparse
import logging
from pathlib import Path
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

N_PARAMS  = 3
INPUT_DIM = 3 + N_PARAMS  # d1, MUV_nearest, n_neighbors + 3 params = 6


def normalize_params(params, param_min, param_max):
    return (2 * (params - param_min) / (param_max - param_min) - 1).astype(np.float32)


def env_to_d1_summary(env):
    """Extract (d1, MUV_nearest, n_neighbors) from environment.

    Parameters
    ----------
    env : np.ndarray, shape (N, 4) — (dx, dy, dz, MUV)

    Returns
    -------
    np.ndarray, shape (3,) — (d1, MUV_nearest, n_neighbors_normalized)
    """
    dists   = np.sqrt((env[:, 0]**2 + env[:, 1]**2 + env[:, 2]**2))
    nearest = np.argmin(dists)
    d1      = dists[nearest]
    muv_nn  = env[nearest, 3]
    n_norm  = len(env) / 200.0
    return np.array([d1, muv_nn, n_norm], dtype=np.float32)


class D1NREDataset(Dataset):
    """Dataset using only d1, MUV_nearest, n_neighbors as summary."""

    def __init__(
        self,
        database_dirs: list,
        param_min: np.ndarray,
        param_max: np.ndarray,
        max_per_catalog: int = 200,
    ):
        self.param_min = param_min
        self.param_max = param_max

        self.summaries   = []
        self.params      = []
        self.catalog_ids = []
        self.is_prior    = []  # track which db each env came from

        cat_idx = 0
        for db_idx, db_dir in enumerate(database_dirs):
            files = sorted(Path(db_dir).glob("nre_*.npz"))
            log.info(f"Loading {len(files)} files from {db_dir} ...")

            for path in files:
                try:
                    data = np.load(path)
                except (EOFError, ValueError, OSError) as e:
                    print(f"Skipping corrupted file: {path} ({e})")
                    continue
                coords  = data['coords']
                offsets = data['offsets']
                params  = data['params']

                indices = np.arange(len(offsets) - 1)
                if max_per_catalog is not None and len(indices) > max_per_catalog:
                    indices = np.random.choice(indices, max_per_catalog, replace=False)

                for i in indices:
                    env = coords[offsets[i]:offsets[i+1]]
                    if len(env) == 0:
                        continue
                    self.summaries.append(env_to_d1_summary(env))
                    self.params.append(params)
                    self.catalog_ids.append(cat_idx)
                    self.is_prior.append(db_idx > 0)

                cat_idx += 1

        log.info(f"Total environments: {len(self.summaries)}")
        n_prior = sum(self.is_prior)
        n_post  = len(self.summaries) - n_prior
        log.info(f"  Posterior: {n_post}  Prior: {n_prior}")

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, idx):
        summary = self.summaries[idx]
        params  = self.params[idx]
        current_catalog = self.catalog_ids[idx]

        x = torch.from_numpy(summary)

        theta_real = torch.from_numpy(
            normalize_params(params, self.param_min, self.param_max)
        )

        # Fake theta: 50% from database, 50% from prior
        if np.random.rand() < 0.5:
            fake_idx = np.random.randint(len(self.summaries))
            while self.catalog_ids[fake_idx] == current_catalog:
                fake_idx = np.random.randint(len(self.summaries))
            fake_params = self.params[fake_idx]
        else:
            fake_params = np.array([
                np.random.uniform(self.param_min[0], self.param_max[0]),
                np.random.uniform(self.param_min[1], self.param_max[1]),
                np.random.uniform(self.param_min[2], self.param_max[2]),
            ])

        theta_fake = torch.from_numpy(
            normalize_params(fake_params, self.param_min, self.param_max)
        )

        x_real = torch.cat([x, theta_real])
        x_fake = torch.cat([x, theta_fake])

        return x_real, torch.tensor(1.0), x_fake, torch.tensor(0.0)


class D1NRENetwork(nn.Module):
    """Small MLP for the minimal d1 summary — 6 inputs only."""

    def __init__(self, input_dim=INPUT_DIM, hidden_dims=[64, 64, 64], dropout=0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x_real, label_r, x_fake, label_f in loader:
        x_real, x_fake   = x_real.to(device), x_fake.to(device)
        label_r, label_f = label_r.to(device), label_f.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_real), label_r) + criterion(model(x_fake), label_f)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_real, label_r, x_fake, label_f in loader:
            x_real, x_fake   = x_real.to(device), x_fake.to(device)
            label_r, label_f = label_r.to(device), label_f.to(device)
            loss = criterion(model(x_real), label_r) + criterion(model(x_fake), label_f)
            total_loss += loss.item()
    return total_loss / len(loader)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train minimal d1 NRE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--database-dir",       type=Path, default=None,
                   help="Primary (posterior) database directory. "
                        "Required unless --prior-only is set.")
    p.add_argument("--prior-database-dir", type=Path, default=None,
                   help="Flat prior database directory. "
                        "Required when --prior-only is set.")
    p.add_argument("--prior-only", action="store_true",
                   help="Train using only the flat prior database "
                        "(ignores --database-dir). "
                        "Requires --prior-database-dir.")
    p.add_argument("--output-dir",  type=Path,
                   default=Path("/groups/astro/ivannik/projects/Neighbors/nre_model_d1"))
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--batch-size",      type=int,   default=512)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--val-frac",        type=float, default=0.2)
    p.add_argument("--hidden-dims",     type=int,   nargs='+', default=[64, 64, 64])
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--max-per-catalog", type=int,   default=200)
    p.add_argument("--oversample-prior",action="store_true",
                   help="Oversample prior database to balance with posterior.")
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    log.info(f"Device: {device}")

    if args.prior_only:
        if args.prior_database_dir is None:
            raise ValueError("--prior-only requires --prior-database-dir to be set.")
        db_dirs = [args.prior_database_dir]
        log.info("Prior-only mode: training on flat prior database only.")
    else:
        if args.database_dir is None:
            raise ValueError("--database-dir is required unless --prior-only is set.")
        db_dirs = [args.database_dir]
        if args.prior_database_dir is not None:
            db_dirs.append(args.prior_database_dir)

    # Param normalization
    log.info("Computing parameter normalization ...")
    all_params = []
    for db_dir in db_dirs:
        for path in sorted(Path(db_dir).glob("nre_*.npz")):
            try:
                all_params.append(np.load(path)['params'])
            except (EOFError, ValueError, OSError, KeyError, zipfile.BadZipFile) as e:
                print(f"Skipping corrupted file: {path} ({e})")
                continue
    all_params = np.stack(all_params)
    param_min  = all_params.min(axis=0)
    param_max  = all_params.max(axis=0)
    log.info(f"  param_min: {param_min}")
    log.info(f"  param_max: {param_max}")
    np.savez(args.output_dir / "normalization.npz",
             param_min=param_min, param_max=param_max)

    dataset = D1NREDataset(
        database_dirs   = db_dirs,
        param_min       = param_min,
        param_max       = param_max,
        max_per_catalog = args.max_per_catalog,
    )

    n_val   = int(len(dataset) * args.val_frac)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Optionally oversample prior to balance with posterior
    if args.oversample_prior and args.prior_database_dir is not None and not args.prior_only:
        log.info("Applying oversampling to balance prior vs posterior ...")
        train_indices = train_ds.indices
        is_prior = np.array(dataset.is_prior)
        n_prior_train = is_prior[train_indices].sum()
        n_post_train  = (~is_prior[train_indices]).sum()
        # Weight: posterior gets 1.0, prior gets n_post/n_prior
        ratio = n_post_train / max(n_prior_train, 1)
        weights = np.where(is_prior[train_indices], ratio, 1.0)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=len(train_indices),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=0)
        log.info(f"  Prior weight: {ratio:.2f}x  (n_post={n_post_train} n_prior={n_prior_train})")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    log.info(f"Train: {n_train}  Val: {n_val}  Input dim: {INPUT_DIM}")

    model     = D1NRENetwork(INPUT_DIM, args.hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return float(epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.BCEWithLogitsLoss()

    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        log.info(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output_dir / "nre_best.pt")
            log.info(f"  -> New best model saved (val={val_loss:.4f})")

    torch.save(model.state_dict(), args.output_dir / "nre_final.pt")
    np.savez(args.output_dir / "model_config.npz",
             hidden_dims=np.array(args.hidden_dims),
             dropout=args.dropout,
             input_dim=INPUT_DIM,
             summary_mode=2)  # 2 = d1 mode, distinct from 0=full, 1=summary

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    log.info(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()