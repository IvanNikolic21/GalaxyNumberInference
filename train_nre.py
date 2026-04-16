#!/usr/bin/env python
"""
train_nre.py
------------
Train a Neural Ratio Estimator (NRE) on the galaxy neighbor database.

Architecture: residual MLP.
- 10 closest neighbors sorted by distance, padded to MAX_NEIGHBORS=10
- Each neighbor has 4 features: (dx, dy, dz, MUV) relative to bright central
- n_neighbors (normalized) appended as extra feature
- Parameters theta = (Muv_add, sigmaUV_a, sigmaUV_b) normalized to [-1, 1]
- Residual MLP outputs log-ratio

Training: binary cross-entropy
- Real pairs (env, theta): label = 1
- Fake pairs (env, shuffled theta): label = 0
- Fake theta: 50% from posterior database, 50% from uniform prior

Usage
-----
    python train_nre.py --database-dir /path/to/nre_database
    python train_nre.py --database-dir /path/to/nre_database \
                        --prior-database-dir /path/to/nre_database_prior \
                        --epochs 100
"""

import argparse
import logging
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

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
# Constants
# ---------------------------------------------------------------------------
MAX_NEIGHBORS = 10
N_FEATURES    = 4    # dx, dy, dz, MUV
N_PARAMS      = 3    # Muv_add, sigmaUV_a, sigmaUV_b
INPUT_DIM     = MAX_NEIGHBORS * N_FEATURES + 1 + N_PARAMS  # 44

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_params(params, param_min, param_max):
    return (2 * (params - param_min) / (param_max - param_min) - 1).astype(np.float32)


def env_to_array(env):
    """Sort by distance, keep 10 closest, pad, append n_neighbors."""
    dists = np.sqrt((env[:, 0]**2 + env[:, 1]**2 + env[:, 2]**2))
    env   = env[np.argsort(dists)]
    n_total = len(env)
    padded  = np.zeros((MAX_NEIGHBORS, N_FEATURES), dtype=np.float32)
    n       = min(n_total, MAX_NEIGHBORS)
    padded[:n] = env[:n]
    n_norm  = np.array([n_total / 200.0], dtype=np.float32)
    return padded.flatten(), n_norm, n


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NREDataset(Dataset):
    """Loads .npz files from one or two database directories.

    Parameters
    ----------
    database_dirs : list of Path
        Primary (posterior) and optional secondary (prior) database directories.
    param_min, param_max : np.ndarray
        Normalization bounds computed across all files.
    augment : bool
        Apply random translation augmentation to xyz coords.
    max_per_catalog : int or None
        Max environments to load per catalog file.
    """

    def __init__(
        self,
        database_dirs: list,
        param_min: np.ndarray,
        param_max: np.ndarray,
        augment: bool = True,
        max_per_catalog: int = 200,
    ):
        self.param_min = param_min
        self.param_max = param_max
        self.augment   = augment

        self.envs        = []
        self.params      = []
        self.catalog_ids = []

        cat_idx = 0
        for db_dir in database_dirs:
            files = sorted(Path(db_dir).glob("nre_*.npz"))
            log.info(f"Loading {len(files)} files from {db_dir} ...")

            for path in files:
                data    = np.load(path)
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
                    self.envs.append(env)
                    self.params.append(params)
                    self.catalog_ids.append(cat_idx)

                cat_idx += 1

        log.info(f"Total environments: {len(self.envs)}")

    def __len__(self):
        return len(self.envs)

    def __getitem__(self, idx):
        env    = self.envs[idx]
        params = self.params[idx]
        current_catalog = self.catalog_ids[idx]

        # Process environment
        flat, n_norm, n = env_to_array(env)

        # Augmentation: random translation of xyz
        if self.augment and n > 0:
            shift = np.random.uniform(-5.0, 5.0, size=3).astype(np.float32)
            flat_2d = flat.reshape(MAX_NEIGHBORS, N_FEATURES)
            flat_2d[:n, :3] += shift
            flat = flat_2d.flatten()

        x = torch.from_numpy(np.concatenate([flat, n_norm]))

        # Real theta
        theta_real = torch.from_numpy(
            normalize_params(params, self.param_min, self.param_max)
        )

        # Fake theta: 50% posterior (hard), 50% prior (easy)
        if np.random.rand() < 0.5:
            fake_idx = np.random.randint(len(self.envs))
            while self.catalog_ids[fake_idx] == current_catalog:
                fake_idx = np.random.randint(len(self.envs))
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


# ---------------------------------------------------------------------------
# Model — Residual MLP
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Single residual block: Linear -> LayerNorm -> GELU -> Dropout + skip."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class NRENetwork(nn.Module):
    """Residual MLP NRE.

    Architecture:
        input -> project to hidden_dim -> N residual blocks -> output scalar

    Residual connections help gradients flow and allow deeper networks
    without vanishing gradients — important for learning fine distinctions
    in a concentrated posterior.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: list = [256, 256, 256, 256],
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
        )

        # Residual blocks — all same width for skip connections
        # If hidden_dims has varying widths, project between them
        blocks = []
        for i in range(len(hidden_dims)):
            in_d  = hidden_dims[i]
            out_d = hidden_dims[i+1] if i+1 < len(hidden_dims) else hidden_dims[i]
            if in_d == out_d:
                blocks.append(ResidualBlock(in_d, dropout))
            else:
                # Projection layer between different widths
                blocks.append(nn.Sequential(
                    nn.Linear(in_d, out_d),
                    nn.LayerNorm(out_d),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ))
        self.blocks = nn.ModuleList(blocks)

        self.output = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x_real, label_r, x_fake, label_f in loader:
        x_real  = x_real.to(device)
        x_fake  = x_fake.to(device)
        label_r = label_r.to(device)
        label_f = label_f.to(device)
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
            x_real  = x_real.to(device)
            x_fake  = x_fake.to(device)
            label_r = label_r.to(device)
            label_f = label_f.to(device)
            loss = criterion(model(x_real), label_r) + criterion(model(x_fake), label_f)
            total_loss += loss.item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train NRE on galaxy neighbor database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--database-dir", type=Path, required=True,
                   help="Primary (posterior) database directory.")
    p.add_argument("--prior-database-dir", type=Path, default=None,
                   help="Optional secondary (prior) database directory.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("/groups/astro/ivannik/projects/Neighbors/nre_model"),
                   help="Where to save model and normalization stats.")
    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--batch-size",       type=int,   default=512)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--val-frac",         type=float, default=0.2)
    p.add_argument("--hidden-dims",      type=int,   nargs='+', default=[256, 256, 256, 256])
    p.add_argument("--dropout",          type=float, default=0.1)
    p.add_argument("--max-per-catalog",  type=int,   default=200)
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    log.info(f"Device: {device}")

    # Collect all database dirs
    db_dirs = [args.database_dir]
    if args.prior_database_dir is not None:
        db_dirs.append(args.prior_database_dir)
    log.info(f"Database dirs: {db_dirs}")

    # Compute param normalization across ALL files from ALL dirs
    log.info("Computing parameter normalization ...")
    all_params = []
    for db_dir in db_dirs:
        for path in sorted(Path(db_dir).glob("nre_*.npz")):
            try:
                all_params.append(np.load(path)['params'])
            except (zipfile.BadZipFile, OSError, ValueError, KeyError) as e:
                print(f"Skipping {path}: {e}")
    all_params = np.stack(all_params)
    param_min  = all_params.min(axis=0)
    param_max  = all_params.max(axis=0)
    log.info(f"  param_min: {param_min}")
    log.info(f"  param_max: {param_max}")

    np.savez(args.output_dir / "normalization.npz",
             param_min=param_min, param_max=param_max)

    # Build dataset
    dataset = NREDataset(
        database_dirs    = db_dirs,
        param_min        = param_min,
        param_max        = param_max,
        augment          = True,
        max_per_catalog  = args.max_per_catalog,
    )

    n_val   = int(len(dataset) * args.val_frac)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    log.info(f"Train: {n_train}  Val: {n_val}")

    # Model
    model     = NRENetwork(INPUT_DIM, args.hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
    )
    criterion = nn.BCEWithLogitsLoss()

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = val_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        log.info(f"Epoch {epoch:3d}/{args.epochs}  "
                 f"train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output_dir / "nre_best.pt")
            log.info(f"  -> New best model saved (val={val_loss:.4f})")

    torch.save(model.state_dict(), args.output_dir / "nre_final.pt")
    np.savez(args.output_dir / "model_config.npz",
             hidden_dims=np.array(args.hidden_dims),
             dropout=args.dropout,
             input_dim=INPUT_DIM)

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    log.info(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()