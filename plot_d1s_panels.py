#!/usr/bin/env python
"""
plot_d1s_panels.py
------------------
Three sets of 3-panel d1s comparison plots:

  Fig 1: Fixed Muv,0=-21.5, z=10.5 — varying Muv,lim = -18, -18.5, -19
  Fig 2: Fixed Muv,lim=-18.5, z=10.5 — varying Muv,0 = -21, -21.5, -22
  Fig 3: Fixed Muv,0=-21.5, Muv,lim=-18.5 — varying z = 8, 12, 14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

from galaxy_neighbors import AnalysisConfig
from galaxy_d1s import load_d1s, D1sConfig

# ---------------------------------------------------------------------------
# Config — must match cache
# ---------------------------------------------------------------------------
cfg = AnalysisConfig(
    bright_limits         = [-20.5, -20.75, -21.0, -21.25, -21.5, -21.75, -22.0],
    faint_limits          = [-17.0, -17.1, -17.2, -17.3, -17.4, -17.5, -17.6, -17.7,
                             -17.8, -17.9, -18.0, -18.1, -18.2, -18.3, -18.4, -18.5,
                             -18.6, -18.7, -18.8, -18.9, -19.0, -19.1, -19.2],
    preselect_faint_limit = -17.0,
    survey_area_arcmin2   = 12.24,
)

d1s_cfg = D1sConfig(bw_fid=0.18, bw_stoc=0.11)

CACHE_ROOT = Path("/groups/astro/ivannik/projects/Neighbors/cache")
OUTPUT_DIR = Path("/groups/astro/ivannik/projects/Neighbors/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cache files per redshift
CACHE = {
    z: (
        CACHE_ROOT / f"z{z}" / f"d1s_fiducial_real{n}.npz",
        CACHE_ROOT / f"z{z}" / f"d1s_stochastic_real{n}.npz",
    )
    for z, n in [(8.0, 1), (10.5, 5), (12.0, 50), (14.0, 100)]
}

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-ticks")
plt.rcParams.update({
    "font.size": 14, "xtick.top": True, "ytick.right": True,
    "xtick.direction": "in", "ytick.direction": "in",
})

COLOR_FID  = d1s_cfg.color_fid
COLOR_STOC = d1s_cfg.color_stoc
LABEL_FID  = d1s_cfg.label_fid
LABEL_STOC = d1s_cfg.label_stoc
BW_FID     = d1s_cfg.bw_fid
BW_STOC    = d1s_cfg.bw_stoc
X          = np.linspace(0, 8, 300)
XLIM       = (0, 8)
YLIM       = (0, 1.7)


def plot_kde(ax, arr, color, label, bw):
    """Plot KDE if array is valid, silently skip otherwise."""
    if len(arr) < 2:
        return
    try:
        kde = gaussian_kde(arr, bw_method=bw)
        ax.plot(X, kde(X), color=color, lw=2.5, label=label)
    except Exception:
        pass


def style_ax(ax, title, show_xlabel=False, show_ylabel=False, show_legend=False):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_title(title, fontsize=13)
    if show_xlabel:
        ax.set_xlabel(r"$d_{12}$ [cMpc]", fontsize=13)
    if show_ylabel:
        ax.set_ylabel(r"PDF$(d_{12})$", fontsize=13)
    if show_legend:
        ax.legend(fontsize=11, framealpha=0.8)


def save(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Load z=10.5 once — used in Fig 1 and Fig 2
# ---------------------------------------------------------------------------
fid_105, stoc_105 = [load_d1s(p, cfg) for p in CACHE[10.5]]

# ---------------------------------------------------------------------------
# Fig 1: vary Muv,lim — fixed Muv,0=-21.5, z=10.5
# ---------------------------------------------------------------------------
BRIGHT_KEY  = "M21.5"
FAINT_KEYS  = ["M18", "M18.5", "M19"]
FAINT_LABS  = [r"$M_{\rm UV,lim}=-18.0$",
               r"$M_{\rm UV,lim}=-18.5$",
               r"$M_{\rm UV,lim}=-19.0$"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for i, (ax, fkey, lab) in enumerate(zip(axes, FAINT_KEYS, FAINT_LABS)):
    plot_kde(ax, fid_105[BRIGHT_KEY][fkey],  COLOR_FID,  LABEL_FID,  BW_FID)
    plot_kde(ax, stoc_105[BRIGHT_KEY][fkey], COLOR_STOC, LABEL_STOC, BW_STOC)
    style_ax(ax, lab,
             show_xlabel=True,
             show_ylabel=(i == 0),
             show_legend=(i == 0))

fig.suptitle(
    r"$M_{\rm UV,0}=-21.5$, $z=10.5$  —  varying $M_{\rm UV,lim}$",
    fontsize=14, y=1.02,
)
fig.tight_layout()
save(fig, "fig1_vary_faint_z10p5_M21p5.pdf")

# ---------------------------------------------------------------------------
# Fig 2: vary Muv,0 — fixed Muv,lim=-18.5, z=10.5
# ---------------------------------------------------------------------------
FAINT_KEY   = "M18.5"
BRIGHT_KEYS = ["M21", "M21.5", "M22"]
BRIGHT_LABS = [r"$M_{\rm UV,0}=-21.0$",
               r"$M_{\rm UV,0}=-21.5$",
               r"$M_{\rm UV,0}=-22.0$"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for i, (ax, bkey, lab) in enumerate(zip(axes, BRIGHT_KEYS, BRIGHT_LABS)):
    plot_kde(ax, fid_105[bkey][FAINT_KEY],  COLOR_FID,  LABEL_FID,  BW_FID)
    plot_kde(ax, stoc_105[bkey][FAINT_KEY], COLOR_STOC, LABEL_STOC, BW_STOC)
    style_ax(ax, lab,
             show_xlabel=True,
             show_ylabel=(i == 0),
             show_legend=(i == 0))

fig.suptitle(
    r"$M_{\rm UV,lim}=-18.5$, $z=10.5$  —  varying $M_{\rm UV,0}$",
    fontsize=14, y=1.02,
)
fig.tight_layout()
save(fig, "fig2_vary_bright_z10p5_M18p5.pdf")

# ---------------------------------------------------------------------------
# Fig 3: vary z — fixed Muv,0=-21.5, Muv,lim=-18.5
# ---------------------------------------------------------------------------
BRIGHT_KEY = "M21.5"
FAINT_KEY  = "M18.5"
REDSHIFTS  = [8.0, 12.0, 14.0]
Z_LABELS   = [r"$z=8$", r"$z=12$", r"$z=14$"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for i, (ax, z, zlab) in enumerate(zip(axes, REDSHIFTS, Z_LABELS)):
    fid_z, stoc_z = [load_d1s(p, cfg) for p in CACHE[z]]
    plot_kde(ax, fid_z[BRIGHT_KEY][FAINT_KEY],  COLOR_FID,  LABEL_FID,  BW_FID)
    plot_kde(ax, stoc_z[BRIGHT_KEY][FAINT_KEY], COLOR_STOC, LABEL_STOC, BW_STOC)
    style_ax(ax, zlab,
             show_xlabel=True,
             show_ylabel=(i == 0),
             show_legend=(i == 0))

fig.suptitle(
    r"$M_{\rm UV,0}=-21.5$, $M_{\rm UV,lim}=-18.5$  —  varying $z$",
    fontsize=14, y=1.02,
)
fig.tight_layout()
save(fig, "fig3_vary_z_M21p5_M18p5.pdf")
