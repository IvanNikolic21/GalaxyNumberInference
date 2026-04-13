#!/usr/bin/env python
"""
train_nre.py
------------
Train a Neural Ratio Estimator (NRE) on the galaxy neighbor database.

Architecture: padded MLP with mask.
- Each environment is padded to MAX_NEIGHBORS=200 neighbors
- Each neighbor has 4 features: (dx, dy, dz, MUV) relative to bright central
- Parameters theta = (Muv_add, sigmaUV_a, sigmaUV_b) are normalized to [-1, 1]
- MLP takes flattened masked environment + theta, outputs log-ratio

Training: binary cross-entropy
- Real pairs (env, theta): label = 1
- Fake pairs (env, shuffled theta): label = 0

Output: saved model + normalization stats for inference

Usage
-----
    python train_nre.py --database-dir /path/to/nre_database
    python train_nre.py --database-dir /path/to/nre_database --epochs 50
"""

import argparse
import logging
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
MAX_NEIGHBORS = 10   # only keep 10 closest
N_FEATURES    = 4
N_PARAMS      = 3
INPUT_DIM     = MAX_NEIGHBORS * N_FEATURES + 1 + N_PARAMS  # +1 for n_neighbors
# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def normalize_params(params, param_min, param_max):
    return 2 * (params - param_min) / (param_max - param_min) - 1

class NREDataset(Dataset):
    """Loads all .npz files from database_dir.

    Each item is a single environment from a single parameter set.
    Environments are padded to MAX_NEIGHBORS with zeros.
    """

    def __init__(self, database_dir: Path, param_min: np.ndarray, param_max: np.ndarray, augment: bool = True, max_per_catalog: int = 200):
        self.param_min = param_min
        self.param_max = param_max
        self.catalog_ids = []
        self.augment = augment
        # Load all environments and their parameters
        self.envs   = []   # list of (N_i, 4) arrays
        self.params = []   # list of (3,) arrays

        files = sorted(database_dir.glob("nre_*.npz"))
        log.info(f"Loading {len(files)} catalog files ...")

        for cat_idx, path in enumerate(files):
            data    = np.load(path)
            coords  = data['coords']   # (total_neighbors, 4)
            offsets = data['offsets']  # (n_bright + 1,)
            params  = data['params']   # (3,)

            indices = np.arange(len(offsets) - 1)
            if max_per_catalog is not None and len(indices) > max_per_catalog:
                indices = np.random.choice(indices, max_per_catalog, replace=False)
            for i in indices:
                env = coords[offsets[i]:offsets[i + 1]]
                if len(env) == 0:
                    continue
                self.envs.append(env)
                self.params.append(params)
                self.catalog_ids.append(cat_idx)
                if augment:
                    # Random translation augmentation — shift all neighbor coords by same random offset
                    # Physical scale: search box half-side is ~12 Mpc, so shift within ±5 Mpc
                    shift = np.random.uniform(-5.0, 5.0, size=(1, 3)).astype(np.float32)
                    env = env.copy()
                    env[:, :3] += shift  # shift only xyz, not MUV (column 3)

        log.info(f"Total environments: {len(self.envs)}")

    def __len__(self):
        return len(self.envs)

    def __getitem__(self, idx):
        env = self.envs[idx]
        params = self.params[idx]

        # Pad and mask
        # Sort by distance (closest first)
        dists = np.sqrt((env[:, 0] ** 2 + env[:, 1] ** 2 + env[:, 2] ** 2))
        order = np.argsort(dists)
        env = env[order]

        # Store total count before truncating
        n_total = len(env)

        # Keep only 10 closest, pad if fewer
        padded = np.zeros((MAX_NEIGHBORS, N_FEATURES), dtype=np.float32)
        n = min(n_total, MAX_NEIGHBORS)
        padded[:n] = env[:n]

        # Augmentation on xyz only
        if self.augment:
            shift = np.random.uniform(-5.0, 5.0, size=(1, 3)).astype(np.float32)
            padded[:n, :3] += shift

        # Flatten + append normalized n_neighbors
        n_norm = np.array([n_total / 200.0], dtype=np.float32)
        x = torch.from_numpy(np.concatenate([padded.flatten(), n_norm]))

        # Normalize real params
        theta_real = torch.from_numpy(
            normalize_params(params, self.param_min, self.param_max).astype(np.float32)
        )

        # Fake params — random from all params in dataset
        # Draw fake theta from a different parameter set — but not too far
        # Find the catalog index of current env, pick a different one
        current_catalog = self.catalog_ids[idx]
        fake_idx = idx
        while self.catalog_ids[fake_idx] == current_catalog:
            fake_idx = np.random.randint(len(self.envs))
        theta_fake = torch.from_numpy(
            normalize_params(self.params[fake_idx], self.param_min, self.param_max).astype(np.float32)
        )

        x_real = torch.cat([x, theta_real])
        x_fake = torch.cat([x, theta_fake])

        return x_real, torch.tensor(1.0), x_fake, torch.tensor(0.0)


# class NREPairedDataset(Dataset):
#     """Wraps NREDataset to produce real + fake pairs for NRE training.
#
#     For each real (env, theta) pair, a fake pair is created by
#     shuffling theta across the batch.
#     """
#
#     def __init__(self, base: NREDataset):
#         self.base = base
#
#     def __len__(self):
#         return len(self.base)
#
#     def __getitem__(self, idx):
#         x, theta = self.base[idx]
#
#         # Fake: draw a random different theta
#         fake_idx = np.random.randint(0, len(self.base))
#         _, theta_fake = self.base[fake_idx]
#
#         # Real pair: label 1, fake pair: label 0
#         x_real  = torch.cat([x, theta])
#         x_fake  = torch.cat([x, theta_fake])
#         label_r = torch.tensor(1.0)
#         label_f = torch.tensor(0.0)
#
#         return x_real, label_r, x_fake, label_f


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NRENetwork(nn.Module):
    """MLP that takes flattened (masked environment + theta) and outputs log-ratio."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x_real, label_r, x_fake, label_f in loader:
        x_real   = x_real.to(device)
        x_fake   = x_fake.to(device)
        label_r  = label_r.to(device)
        label_f  = label_f.to(device)

        optimizer.zero_grad()
        logits_r = model(x_real)
        logits_f = model(x_fake)
        loss = criterion(logits_r, label_r) + criterion(logits_f, label_f)
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
            logits_r = model(x_real)
            logits_f = model(x_fake)
            loss = criterion(logits_r, label_r) + criterion(logits_f, label_f)
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
                   help="Directory containing nre_*.npz files.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("/groups/astro/ivannik/projects/Neighbors/nre_model"),
                   help="Where to save model and normalization stats.")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--val-frac",   type=float, default=0.2)
    p.add_argument("--hidden-dims", type=int, nargs='+', default=[128, 64, 32])
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--max-per-catalog", type=int, default=200)
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

    # Compute param normalization from all files
    log.info("Computing parameter normalization ...")
    all_params = []
    for path in sorted(args.database_dir.glob("nre_*.npz")):
        all_params.append(np.load(path)['params'])
    all_params = np.stack(all_params)
    param_min  = all_params.min(axis=0)
    param_max  = all_params.max(axis=0)
    log.info(f"  param_min: {param_min}")
    log.info(f"  param_max: {param_max}")

    # Save normalization for inference
    np.savez(args.output_dir / "normalization.npz",
             param_min=param_min, param_max=param_max)

    # Build dataset
    dataset = NREDataset(args.database_dir, param_min, param_max,
                         augment=True, max_per_catalog=args.max_per_catalog)

    n_val = int(len(dataset) * args.val_frac)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))
    # n_val   = int(len(paired) * args.val_frac)
    # n_train = len(paired) - n_val
    # train_ds, val_ds = random_split(paired, [n_train, n_val],
    #                                 generator=torch.Generator().manual_seed(args.seed))

    # train_ds = NREPairedDataset(NREDataset(args.database_dir, param_min, param_max, augment=True))
    # val_ds = NREPairedDataset(NREDataset(args.database_dir, param_min, param_max, augment=False))

    # Then split indices manually

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    log.info(f"Train: {n_train}  Val: {n_val}")

    # Model
    model     = NRENetwork(INPUT_DIM, args.hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5,# verbose=True
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
            torch.save(model.state_dict(),
                       args.output_dir / "nre_best.pt")
            log.info(f"  -> New best model saved (val={val_loss:.4f})")

    # Save final model + config
    torch.save(model.state_dict(), args.output_dir / "nre_final.pt")
    np.savez(args.output_dir / "model_config.npz",
             hidden_dims=args.hidden_dims,
             dropout=args.dropout,
             input_dim=INPUT_DIM)

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    log.info(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()