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
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

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
MAX_NEIGHBORS      = 10
N_FEATURES_FULL    = 4    # dx, dy, dz, MUV
N_FEATURES_SUMM    = 2    # distance, MUV
N_FEATURES_ANG     = 3    # dx, dy, MUV (no dz)
N_PARAMS           = 3    # Muv_add, sigmaUV_a, sigmaUV_b
INPUT_DIM_FULL     = MAX_NEIGHBORS * N_FEATURES_FULL + 1 + N_PARAMS  # 44
INPUT_DIM_SUMMARY  = MAX_NEIGHBORS * N_FEATURES_SUMM + 1 + N_PARAMS  # 24
INPUT_DIM_ANGULAR  = MAX_NEIGHBORS * N_FEATURES_ANG  + 1 + N_PARAMS  # 34

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_params(params, param_min, param_max):
    return (2 * (params - param_min) / (param_max - param_min) - 1).astype(np.float32)


def env_to_array(env, summary_mode=False, only_angular=False):
    """Sort by distance, keep 10 closest, pad, append n_neighbors.

    Parameters
    ----------
    env : np.ndarray, shape (N, 4) — (dx, dy, dz, MUV)
    summary_mode : bool
        If True, use only (distance, MUV) per neighbor — 2 features instead of 4.
        Input dim becomes MAX_NEIGHBORS * 2 + 1 + N_PARAMS = 24.
        If False, use full (dx, dy, dz, MUV) — 4 features (default).
    only_angular : bool
        If True, use projected 2D distance sqrt(dx²+dy²) instead of full 3D
        distance sqrt(dx²+dy²+dz²). Matches real observations where only
        angular separation is available.
    """
    if only_angular:
        dists = np.sqrt(env[:, 0]**2 + env[:, 1]**2)
    else:
        dists = np.sqrt(env[:, 0]**2 + env[:, 1]**2 + env[:, 2]**2)
    order = np.argsort(dists)
    env   = env[order]
    dists = dists[order]
    n_total = len(env)
    n       = min(n_total, MAX_NEIGHBORS)

    if summary_mode:
        # (distance, MUV) only — 2 features, translation invariant by construction
        n_feat  = N_FEATURES_SUMM
        padded  = np.zeros((MAX_NEIGHBORS, n_feat), dtype=np.float32)
        padded[:n, 0] = dists[:n]          # distance (2D if only_angular, else 3D)
        padded[:n, 1] = env[:n, 3]         # MUV
    elif only_angular:
        # (dx, dy, MUV) — 3 features, dz dropped
        n_feat  = N_FEATURES_ANG
        padded  = np.zeros((MAX_NEIGHBORS, n_feat), dtype=np.float32)
        padded[:n, 0] = env[:n, 0]         # dx
        padded[:n, 1] = env[:n, 1]         # dy
        padded[:n, 2] = env[:n, 3]         # MUV (skip dz at index 2)
    else:
        # Full (dx, dy, dz, MUV) — 4 features
        n_feat  = N_FEATURES_FULL
        padded  = np.zeros((MAX_NEIGHBORS, n_feat), dtype=np.float32)
        padded[:n] = env[:n]

    n_norm = np.array([n_total / 200.0], dtype=np.float32)
    return padded.flatten(), n_norm, n, dists


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
        summary_mode: bool = False,
        only_angular: bool = False,
        weight_by_catalog_count: bool = False,
    ):
        self.param_min    = param_min
        self.param_max    = param_max
        self.augment      = augment
        self.summary_mode = summary_mode
        self.only_angular = only_angular

        self.envs        = []
        self.params      = []
        self.catalog_ids = []
        self.is_prior    = []  # True for environments from prior database
        self.weights     = []  # per-example loss weight (see weight_by_catalog_count)

        cat_idx = 0
        for db_idx, db_dir in enumerate(database_dirs):
            files = sorted(Path(db_dir).glob("nre_*.npz"))
            log.info(f"Loading {len(files)} files from {db_dir} ...")

            for path in files:
                try:
                    data    = np.load(path)
                    coords  = data['coords']
                    offsets = data['offsets']
                    params  = data['params']
                except (EOFError, ValueError, OSError, KeyError, zipfile.BadZipFile) as e:
                    print(f"Skipping corrupted file: {path} ({e})")
                    continue

                indices = np.arange(len(offsets) - 1)
                if max_per_catalog is not None and max_per_catalog > 0 and len(indices) > max_per_catalog:
                    indices = np.random.choice(indices, max_per_catalog, replace=False)

                # Inverse-frequency weighting: give every catalog (parameter point)
                # equal total contribution to the loss, regardless of how many
                # bright-galaxy environments it happened to produce. Weight is set
                # from n_used (the count actually loaded for this catalog, after
                # any --max-per-catalog subsampling above) so each catalog's
                # per-epoch environments sum to weight 1 in aggregate -- catalogs
                # astrophysically rich in bright galaxies (e.g. high sigma_UV,b)
                # no longer dominate the gradient purely by volume. Approximate:
                # a few environments per catalog may turn out empty below and get
                # skipped without re-normalizing n_used, which is fine for this
                # purpose (balances catalogs, not exact per-example weight).
                n_used = len(indices)
                weight = (1.0 / n_used) if (weight_by_catalog_count and n_used > 0) else 1.0

                for i in indices:
                    env = coords[offsets[i]:offsets[i+1]]
                    if len(env) == 0:
                        continue
                    # .copy() breaks the view reference to coords so the full
                    # coords array (potentially hundreds of MB per file) can be
                    # garbage-collected once we move to the next file.
                    self.envs.append(env.copy())
                    self.params.append(params)
                    self.catalog_ids.append(cat_idx)
                    self.is_prior.append(db_idx > 0)
                    self.weights.append(weight)

                cat_idx += 1

        log.info(f"Total environments: {len(self.envs)}")

    def __len__(self):
        return len(self.envs)

    def __getitem__(self, idx):
        env    = self.envs[idx]
        params = self.params[idx]
        current_catalog = self.catalog_ids[idx]

        # Process environment
        flat, n_norm, n, dists = env_to_array(env, summary_mode=self.summary_mode,
                                               only_angular=self.only_angular)

        # Augmentation: random translation of xyz (only in full 3D mode -- summary
        # uses distances, and only_angular drops dz and uses N_FEATURES_ANG=3, not
        # N_FEATURES_FULL=4, so the reshape below is only valid when NEITHER is set).
        if self.augment and n > 0 and not self.summary_mode and not self.only_angular:
            shift = np.random.uniform(-5.0, 5.0, size=3).astype(np.float32)
            flat_2d = flat.reshape(MAX_NEIGHBORS, N_FEATURES_FULL)
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

        weight = torch.tensor(self.weights[idx], dtype=torch.float32)
        return x_real, torch.tensor(1.0), x_fake, torch.tensor(0.0), weight


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
        input_dim: int = INPUT_DIM_FULL,
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

def _weighted_bce(criterion, logits_r, label_r, logits_f, label_f, weight):
    """criterion must use reduction='none'. Combines real+fake terms per
    example, then takes the weight-normalized mean over the batch. With all
    weights == 1 this is numerically identical to
    criterion(...).mean() + criterion(...).mean() under mean-reduction."""
    loss_r = criterion(logits_r, label_r)
    loss_f = criterion(logits_f, label_f)
    return ((loss_r + loss_f) * weight).sum() / weight.sum()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x_real, label_r, x_fake, label_f, weight in loader:
        x_real  = x_real.to(device)
        x_fake  = x_fake.to(device)
        label_r = label_r.to(device)
        label_f = label_f.to(device)
        weight  = weight.to(device)
        optimizer.zero_grad()
        loss = _weighted_bce(criterion, model(x_real), label_r, model(x_fake), label_f, weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_real, label_r, x_fake, label_f, weight in loader:
            x_real  = x_real.to(device)
            x_fake  = x_fake.to(device)
            label_r = label_r.to(device)
            label_f = label_f.to(device)
            weight  = weight.to(device)
            loss = _weighted_bce(criterion, model(x_real), label_r, model(x_fake), label_f, weight)
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
    p.add_argument("--database-dir", type=Path, default=None,
                   help="Primary (posterior) database directory. "
                        "Required unless --prior-only is set.")
    p.add_argument("--prior-database-dir", type=Path, default=None,
                   help="Flat prior database directory. "
                        "Required when --prior-only is set.")
    p.add_argument("--prior-only", action="store_true",
                   help="Train using only the flat prior database "
                        "(ignores --database-dir). "
                        "Requires --prior-database-dir.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("/groups/astro/ivannik/projects/Neighbors/nre_model"),
                   help="Where to save model and normalization stats.")
    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--batch-size",       type=int,   default=512)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--val-frac",         type=float, default=0.2)
    p.add_argument("--hidden-dims",      type=int,   nargs='+', default=[256, 256, 256, 256])
    p.add_argument("--dropout",          type=float, default=0.1)
    p.add_argument("--max-per-catalog",  type=int,   default=200,
                   help="Cap on environments loaded per catalog file, randomly "
                        "subsampled when exceeded. Set to 0 to disable (load every "
                        "environment) -- combine with --weight-by-catalog-count to "
                        "keep all data while still balancing catalogs' loss contribution.")
    p.add_argument("--weight-by-catalog-count", action="store_true",
                   help="Weight each environment's loss contribution by 1/(environments "
                        "used for its catalog), so every catalog contributes equally to "
                        "the loss regardless of how many bright-galaxy environments it "
                        "produced. Alternative to (or combinable with) --max-per-catalog; "
                        "with --max-per-catalog 0 this reweights the full, undiscarded "
                        "dataset instead of hard-capping it.")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--oversample-prior", action="store_true",
                   help="Oversample prior database environments to balance 1:1 with posterior.")
    p.add_argument("--summary-mode",    action="store_true",
                   help="Use (distance, MUV) per neighbor instead of (dx,dy,dz,MUV). "                        "Input dim: 24 instead of 44. Translation invariant by construction.")
    p.add_argument("--only-angular",    action="store_true",
                   help="Use projected 2D distance sqrt(dx²+dy²) instead of full 3D. "
                        "Matches real observations with angular separations only.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Suffix output dir for angular-only models to avoid overwriting standard
    # models -- must happen BEFORE mkdir/normalization.npz below, or those land
    # in the unsuffixed directory while checkpoints try to write to the (never
    # created) suffixed one.
    if args.only_angular:
        args.output_dir = args.output_dir.parent / (args.output_dir.name + "_only_ang")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    log.info(f"Device: {device}")

    # Collect all database dirs
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
    log.info(f"Database dirs: {db_dirs}")

    # Compute param normalization across ALL files from ALL dirs
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

    # Build dataset
    if args.summary_mode:
        input_dim = INPUT_DIM_SUMMARY
        mode_str  = "summary (distance+MUV)"
    elif args.only_angular:
        input_dim = INPUT_DIM_ANGULAR
        mode_str  = "angular (dx,dy,MUV)"
    else:
        input_dim = INPUT_DIM_FULL
        mode_str  = "full (dx,dy,dz,MUV)"
    log.info(f"Mode: {mode_str}  input_dim={input_dim}")

    dataset = NREDataset(
        database_dirs           = db_dirs,
        param_min               = param_min,
        param_max               = param_max,
        augment                 = True,
        max_per_catalog         = args.max_per_catalog,
        summary_mode            = args.summary_mode,
        only_angular            = args.only_angular,
        weight_by_catalog_count = args.weight_by_catalog_count,
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
        ratio   = n_post_train / max(n_prior_train, 1)
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
                                  shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    log.info(f"Train: {n_train}  Val: {n_val}")

    # Model
    model     = NRENetwork(input_dim, args.hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Warmup 5 epochs then cosine annealing
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return float(epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # per-example, for _weighted_bce

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

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
             input_dim=input_dim,
             summary_mode=int(args.summary_mode),
             only_angular=int(args.only_angular))

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    log.info(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()