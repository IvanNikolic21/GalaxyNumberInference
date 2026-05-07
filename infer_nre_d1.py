#!/usr/bin/env python
"""
infer_nre_d1.py
---------------
Inference for the minimal d1 NRE (train_nre_d1.py).

Summary per environment: (d1, MUV_nearest, n_neighbors_normalized)
Combined over N environments: log p(theta|x_1..N) = sum_i log_ratio(x_i, theta)

Usage
-----
    # Mock observation from database
    python infer_nre_d1.py --obs-file /path/to/nre_database/nre_*.npz --n-obs 50

    # Single environment
    python infer_nre_d1.py --obs-file /path/to/nre_*.npz --n-obs 1 --use-grid

    # Real observation (same .npz format, no 'params' key needed)
    python infer_nre_d1.py --obs-file /path/to/real_obs.npz \
                           --truths -0.4 0.1 0.7
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import corner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

N_PARAMS  = 3
INPUT_DIM = 3 + N_PARAMS  # d1, MUV_nearest, n_neighbors + theta

# Reference model parameters: (Muv_add, sigmaUV_a, sigmaUV_b)
FIDUCIAL_PARAMS   = np.array([-0.8,   0.0,  0.3])
STOCHASTIC_PARAMS = np.array([ 0.3,  -0.34, 0.6])


# ---------------------------------------------------------------------------
# Model — must match train_nre_d1.py
# ---------------------------------------------------------------------------

class D1NRENetwork(nn.Module):
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_params(params, param_min, param_max):
    return (2 * (params - param_min) / (param_max - param_min) - 1).astype(np.float32)


def env_to_d1_summary(env, only_angular=False):
    """Extract (d1, MUV_nearest, n_neighbors_normalized) from environment."""
    if only_angular:
        dists = np.sqrt(env[:, 0]**2 + env[:, 1]**2)
    else:
        dists = np.sqrt(env[:, 0]**2 + env[:, 1]**2 + env[:, 2]**2)
    nearest = np.argmin(dists)
    d1      = dists[nearest]
    muv_nn  = env[nearest, 3]
    n_norm  = len(env) / 200.0
    return np.array([d1, muv_nn, n_norm], dtype=np.float32)


def log_posterior(thetas, summaries, model, param_min, param_max):
    """Evaluate sum of log-ratios over all environments for a batch of thetas.

    Parameters
    ----------
    thetas : np.ndarray, shape (B, 3)
    summaries : list of np.ndarray, shape (3,) each
    model : D1NRENetwork

    Returns
    -------
    log_post : np.ndarray, shape (B,)
    """
    thetas_norm = torch.from_numpy(
        np.stack([normalize_params(t, param_min, param_max) for t in thetas])
    )  # (B, 3)

    log_post = torch.zeros(len(thetas))
    with torch.no_grad():
        for s in summaries:
            s_rep = torch.from_numpy(s).unsqueeze(0).expand(len(thetas), -1)  # (B, 3)
            inp   = torch.cat([s_rep, thetas_norm], dim=1)                     # (B, 6)
            log_post += model(inp)

    return log_post.numpy()


# ---------------------------------------------------------------------------
# Posterior sampling
# ---------------------------------------------------------------------------

def sample_posterior_mcmc(
    summaries, model, param_min, param_max,
    n_walkers=32, n_steps=2000, n_burn=500,
):
    try:
        import emcee
    except ImportError:
        raise ImportError("pip install emcee")

    def log_prob(theta):
        if np.any(theta < param_min) or np.any(theta > param_max):
            return -np.inf
        return float(log_posterior(theta[np.newaxis], summaries, model, param_min, param_max)[0])

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
    summaries, model, param_min, param_max, n_grid=50, n_samples=10000,
):
    g0 = np.linspace(param_min[0], param_max[0], n_grid)
    g1 = np.linspace(param_min[1], param_max[1], n_grid)
    g2 = np.linspace(param_min[2], param_max[2], n_grid)
    G0, G1, G2   = np.meshgrid(g0, g1, g2, indexing='ij')
    grid_params  = np.stack([G0.ravel(), G1.ravel(), G2.ravel()], axis=1)

    log.info(f"Evaluating on {len(grid_params)} grid points ...")
    log_ratios = np.zeros(len(grid_params))
    batch_size = 1024
    for start in range(0, len(grid_params), batch_size):
        batch = grid_params[start:start+batch_size]
        log_ratios[start:start+batch_size] = log_posterior(
            batch, summaries, model, param_min, param_max
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
        description="d1 NRE inference — posterior corner plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--obs-file",   type=Path, required=True,
                   help="Observation .npz file (nre_*.npz format).")
    p.add_argument("--model-dir",  type=Path,
                   default=Path("/groups/astro/ivannik/projects/Neighbors/nre_model_d1"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("/groups/astro/ivannik/projects/Neighbors/nre_model_d1"))
    p.add_argument("--n-obs",      type=int,  default=50,
                   help="Number of environments to use.")
    p.add_argument("--n-grid",     type=int,  default=50,
                   help="Grid resolution per parameter (--use-grid only).")
    p.add_argument("--use-grid",   action="store_true",
                   help="Use grid sampling instead of MCMC.")
    p.add_argument("--n-walkers",  type=int,  default=32)
    p.add_argument("--n-steps",    type=int,  default=2000)
    p.add_argument("--n-burn",     type=int,  default=500)
    p.add_argument("--truths",       type=float, nargs=3, default=None,
                   help="True params (Muv_add sigmaUV_a sigmaUV_b) if not in file.")
    p.add_argument("--only-angular", action="store_true",
                   help="Use projected 2D distance sqrt(dx²+dy²). "
                        "Loaded from model_config.npz if not set.")
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
    only_angular = bool(int(config['only_angular'])) if 'only_angular' in config else False
    if args.only_angular:
        only_angular = True  # CLI flag overrides config
    log.info(f"only_angular={only_angular}")

    model = D1NRENetwork(input_dim, hidden_dims, dropout)
    model.load_state_dict(torch.load(args.model_dir / "nre_best.pt", map_location="cpu"))
    model.eval()
    log.info(f"Model loaded. input_dim={input_dim} hidden_dims={hidden_dims}")

    # Load observation
    obs_data   = np.load(args.obs_file)
    obs_coords = obs_data['coords']
    obs_offs   = obs_data['offsets']

    if 'params' in obs_data:
        true_params = obs_data['params']
        log.info(f"True params: Muv_add={true_params[0]:.3f}  "
                 f"sigmaUV_a={true_params[1]:.3f}  sigmaUV_b={true_params[2]:.3f}")
    else:
        true_params = np.array(args.truths) if args.truths is not None else None

    # Build d1 summaries
    summaries = []
    for i in range(len(obs_offs) - 1):
        env = obs_coords[obs_offs[i]:obs_offs[i+1]]
        if len(env) > 0:
            summaries.append(env_to_d1_summary(env, only_angular=only_angular))
        if len(summaries) == args.n_obs:
            break
    log.info(f"Using {len(summaries)} environments.")
    log.info(f"  d1 values: mean={np.mean([s[0] for s in summaries]):.3f}  "
             f"std={np.std([s[0] for s in summaries]):.3f}")

    # Sample posterior
    if args.use_grid:
        samples = sample_posterior_grid(
            summaries, model, param_min, param_max, n_grid=args.n_grid,
        )
    else:
        samples = sample_posterior_mcmc(
            summaries, model, param_min, param_max,
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
    corner.overplot_lines(fig, FIDUCIAL_PARAMS,   color='C1')
    corner.overplot_lines(fig, STOCHASTIC_PARAMS, color='C0')

    legend_handles = [
        mlines.Line2D([], [], color='C1', lw=1.5, label='increased luminosity'),
        mlines.Line2D([], [], color='C0', lw=1.5, label='increased stochasticity'),
    ]
    fig.legend(handles=legend_handles, loc='upper right', fontsize=11,
               bbox_to_anchor=(1.0, 1.0))

    fig.suptitle(
        f"d1 NRE posterior — {len(summaries)} environment(s)  "
        f"({'MCMC' if not args.use_grid else 'grid'})",
        fontsize=14, y=1.02,
    )

    out = args.output_dir / f"corner_d1_N{len(summaries)}.pdf"
    fig.savefig(out, bbox_inches="tight")
    log.info(f"Saved: {out}")

    np.save(args.output_dir / f"posterior_samples_d1_N{len(summaries)}.npy", samples)
    log.info("Samples saved.")


if __name__ == "__main__":
    main()