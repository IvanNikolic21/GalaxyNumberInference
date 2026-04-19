#!/usr/bin/env python
"""
infer_nre.py
------------
Run NRE inference on a set of observed environments.

Produces a corner plot of the posterior p(theta | x_1, ..., x_N) by
combining N independent environment observations:

    log p(theta | x_1..N) = sum_i log_ratio(x_i, theta) + const

Uses MCMC (emcee) for posterior sampling — much finer resolution than grid.
Falls back to grid if emcee is not installed.

Usage
-----
    # Use a catalog from the database as mock observation
    python infer_nre.py --obs-file /path/to/nre_database/nre_*.npz --n-obs 50

    # Use a real observation file (same .npz format)
    python infer_nre.py --obs-file /path/to/real_observation.npz --n-obs 1
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import corner

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
# Constants — must match training
# ---------------------------------------------------------------------------
MAX_NEIGHBORS     = 10
N_FEATURES_FULL   = 4
N_FEATURES_SUMM   = 2
N_PARAMS          = 3
INPUT_DIM_FULL    = MAX_NEIGHBORS * N_FEATURES_FULL + 1 + N_PARAMS  # 44
INPUT_DIM_SUMMARY = MAX_NEIGHBORS * N_FEATURES_SUMM + 1 + N_PARAMS  # 24

# ---------------------------------------------------------------------------
# Model definition — must match train_nre.py
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.net(x))


class NRENetwork(nn.Module):
    def __init__(self, input_dim=INPUT_DIM_SUMMARY, hidden_dims=[256,256,256,256], dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
        )
        blocks = []
        for i in range(len(hidden_dims)):
            in_d  = hidden_dims[i]
            out_d = hidden_dims[i+1] if i+1 < len(hidden_dims) else hidden_dims[i]
            if in_d == out_d:
                blocks.append(ResidualBlock(in_d, dropout))
            else:
                blocks.append(nn.Sequential(
                    nn.Linear(in_d, out_d), nn.LayerNorm(out_d),
                    nn.GELU(), nn.Dropout(dropout),
                ))
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(-1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_params(params, param_min, param_max):
    return (2 * (params - param_min) / (param_max - param_min) - 1).astype(np.float32)


def env_to_tensor(env, summary_mode=False):
    dists = np.sqrt((env[:, 0]**2 + env[:, 1]**2 + env[:, 2]**2))
    order = np.argsort(dists)
    env   = env[order]
    dists = dists[order]
    n_total = len(env)
    n       = min(n_total, MAX_NEIGHBORS)

    if summary_mode:
        padded = np.zeros((MAX_NEIGHBORS, N_FEATURES_SUMM), dtype=np.float32)
        padded[:n, 0] = dists[:n]
        padded[:n, 1] = env[:n, 3]
    else:
        padded = np.zeros((MAX_NEIGHBORS, N_FEATURES_FULL), dtype=np.float32)
        padded[:n] = env[:n]

    n_norm = np.array([n_total / 200.0], dtype=np.float32)
    return torch.from_numpy(np.concatenate([padded.flatten(), n_norm]))


def log_posterior(thetas, env_tensors, model, param_min, param_max):
    """Evaluate sum of log-ratios over all environments for a batch of thetas.

    Parameters
    ----------
    thetas : np.ndarray, shape (B, 3) — unnormalized parameter values
    env_tensors : list of torch.Tensor — pre-computed environment tensors
    model : NRENetwork
    param_min, param_max : np.ndarray

    Returns
    -------
    log_post : np.ndarray, shape (B,)
    """
    thetas_norm = torch.from_numpy(
        np.stack([normalize_params(t, param_min, param_max) for t in thetas])
    )  # (B, 3)

    log_post = torch.zeros(len(thetas))
    with torch.no_grad():
        for x in env_tensors:
            x_rep = x.unsqueeze(0).expand(len(thetas), -1)  # (B, 41)
            inp   = torch.cat([x_rep, thetas_norm], dim=1)   # (B, 44)
            log_post += model(inp)

    return log_post.numpy()

# ---------------------------------------------------------------------------
# Posterior sampling
# ---------------------------------------------------------------------------

def sample_posterior_mcmc(
    env_tensors, model, param_min, param_max,
    n_walkers=32, n_steps=2000, n_burn=500,
):
    """Sample posterior using emcee MCMC."""
    try:
        import emcee
    except ImportError:
        raise ImportError("pip install emcee")

    def log_prob(theta):
        # Prior: uniform within param_min, param_max
        if np.any(theta < param_min) or np.any(theta > param_max):
            return -np.inf
        return float(log_posterior(theta[np.newaxis], env_tensors, model, param_min, param_max)[0])

    # Initialize walkers around the prior center
    center = (param_min + param_max) / 2
    scale  = (param_max - param_min) / 10
    p0     = center + scale * np.random.randn(n_walkers, N_PARAMS)
    p0     = np.clip(p0, param_min, param_max)

    sampler = emcee.EnsembleSampler(n_walkers, N_PARAMS, log_prob)
    log.info(f"Running MCMC: {n_walkers} walkers, {n_steps} steps, {n_burn} burn-in ...")
    sampler.run_mcmc(p0, n_steps, progress=True)

    samples = sampler.get_chain(discard=n_burn, flat=True)
    log.info(f"  Acceptance fraction: {sampler.acceptance_fraction.mean():.3f}")
    log.info(f"  Samples: {len(samples)}")
    return samples


def sample_posterior_grid(
    env_tensors, model, param_min, param_max, n_grid=50, n_samples=10000,
):
    """Sample posterior via grid evaluation + weighted resampling."""
    g0 = np.linspace(param_min[0], param_max[0], n_grid)
    g1 = np.linspace(param_min[1], param_max[1], n_grid)
    g2 = np.linspace(param_min[2], param_max[2], n_grid)
    G0, G1, G2 = np.meshgrid(g0, g1, g2, indexing='ij')
    grid_params = np.stack([G0.ravel(), G1.ravel(), G2.ravel()], axis=1)

    log.info(f"Evaluating on {len(grid_params)} grid points ...")
    log_ratios = np.zeros(len(grid_params))
    batch_size = 1024
    for start in range(0, len(grid_params), batch_size):
        batch = grid_params[start:start+batch_size]
        log_ratios[start:start+batch_size] = log_posterior(
            batch, env_tensors, model, param_min, param_max
        )

    log_post = log_ratios - log_ratios.max()
    weights  = np.exp(log_post)
    weights /= weights.sum()
    idx      = np.random.choice(len(weights), size=n_samples, p=weights)
    return grid_params[idx]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="NRE inference — produce posterior corner plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--obs-file",   type=Path, required=True,
                   help="Path to .npz observation file (nre_*.npz format).")
    p.add_argument("--model-dir",  type=Path,
                   default=Path("/groups/astro/ivannik/projects/Neighbors/nre_model"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("/groups/astro/ivannik/projects/Neighbors/nre_model"))
    p.add_argument("--n-obs",      type=int,  default=50,
                   help="Number of environments to use as observation.")
    p.add_argument("--n-grid",     type=int,  default=50,
                   help="Grid resolution per parameter (used if --use-grid).")
    p.add_argument("--use-grid",   action="store_true",
                   help="Use grid sampling instead of MCMC.")
    p.add_argument("--n-walkers",  type=int,  default=32,
                   help="MCMC walkers (emcee).")
    p.add_argument("--n-steps",    type=int,  default=2000,
                   help="MCMC steps per walker.")
    p.add_argument("--n-burn",     type=int,  default=500,
                   help="MCMC burn-in steps.")
    p.add_argument("--truths",     type=float, nargs=3, default=None,
                   help="True parameter values for marking on plot (optional).")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    config    = np.load(args.model_dir / "model_config.npz")
    norm      = np.load(args.model_dir / "normalization.npz")
    param_min = norm['param_min']
    param_max = norm['param_max']

    hidden_dims  = list(config['hidden_dims'])
    dropout      = float(config['dropout'])
    input_dim    = int(config['input_dim'])
    summary_mode = bool(int(config['summary_mode'])) if 'summary_mode' in config else False
    log.info(f"Mode: {'summary' if summary_mode else 'full'}  input_dim={input_dim}")

    model = NRENetwork(input_dim, hidden_dims, dropout)
    model.load_state_dict(torch.load(args.model_dir / "nre_best.pt", map_location="cpu"))
    model.eval()
    log.info("Model loaded.")

    # Load observation
    obs_data   = np.load(args.obs_file)
    obs_coords = obs_data['coords']
    obs_offs   = obs_data['offsets']

    # True params — from file if available, else from --truths
    if 'params' in obs_data:
        true_params = obs_data['params']
        log.info(f"True params: Muv_add={true_params[0]:.3f}  "
                 f"sigmaUV_a={true_params[1]:.3f}  sigmaUV_b={true_params[2]:.3f}")
    else:
        true_params = np.array(args.truths) if args.truths is not None else None

    # Build environment tensors
    env_tensors = []
    for i in range(len(obs_offs) - 1):
        env = obs_coords[obs_offs[i]:obs_offs[i+1]]
        if len(env) > 0:
            env_tensors.append(env_to_tensor(env, summary_mode=summary_mode))
        if len(env_tensors) == args.n_obs:
            break
    log.info(f"Using {len(env_tensors)} environments.")

    # Sample posterior
    if args.use_grid:
        samples = sample_posterior_grid(
            env_tensors, model, param_min, param_max,
            n_grid=args.n_grid,
        )
    else:
        samples = sample_posterior_mcmc(
            env_tensors, model, param_min, param_max,
            n_walkers=args.n_walkers,
            n_steps=args.n_steps,
            n_burn=args.n_burn,
        )

    # Corner plot
    param_labels = [r"$M_{\rm UV,add}$", r"$\sigma_{\rm UV,a}$", r"$\sigma_{\rm UV,b}$"]
    truths = list(true_params) if true_params is not None else None

    fig = corner.corner(
        samples,
        labels=param_labels,
        truths=truths,
        truth_color='red',
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 13},
        quantiles=[0.16, 0.5, 0.84],
        bins=40,
        smooth=1.0,
        range = [(-1.5,2.0), (-1.0, 1.5), (0.0, 3.0)],
        levels = [0.68,0.95],
        color='black' ,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=True,
    )
    fig.suptitle(
        f"NRE posterior — {len(env_tensors)} environment(s)  "
        f"({'MCMC' if not args.use_grid else 'grid'})",
        fontsize=14, y=1.02,
    )

    out = args.output_dir / f"corner_N{len(env_tensors)}_{true_params[0]}.pdf"
    fig.savefig(out, bbox_inches="tight")
    log.info(f"Saved: {out}")

    # Also save samples for external use
    np.save(args.output_dir / f"posterior_samples_N{len(env_tensors)}.npy", samples)
    log.info(f"Samples saved.")


if __name__ == "__main__":
    main()