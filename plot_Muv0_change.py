#!/usr/bin/env python
"""
plot_d1s_comparison.py
----------------------
3-panel horizontal plot of d1 distributions for fixed Muv,faint = -18.5,
varying Muv,0 = -21.0, -21.5, -22.0, at z=10.5.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

from galaxy_neighbors import AnalysisConfig
from galaxy_d1s import load_d1s, D1sConfig

# ---------------------------------------------------------------------------
# Config — must match what was used to compute the cache
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

CACHE_DIR = Path("/groups/astro/ivannik/projects/Neighbors/cache/z10.5")
OUTPUT    = Path("/groups/astro/ivannik/projects/Neighbors/z10.5/d1s_fixed_faint_vary_bright.pdf")

BRIGHT_KEYS  = ["M21", "M21.5", "M22"]       # Muv,0 = -21.0, -21.5, -22.0
FAINT_KEY    = "M18.5"                         # Muv,faint = -18.5
REDSHIFT     = 10.5

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
d1s_fid  = load_d1s(CACHE_DIR / "d1s_fiducial_real5.npz",  cfg)
d1s_stoc = load_d1s(CACHE_DIR / "d1s_stochastic_real5.npz", cfg)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-ticks")
plt.rcParams.update({
    "font.size": 14, "xtick.top": True, "ytick.right": True,
    "xtick.direction": "in", "ytick.direction": "in",
})

x = np.linspace(0, 8, 300)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, bkey in zip(axes, BRIGHT_KEYS):
    arr_fid  = d1s_fid[bkey][FAINT_KEY]
    arr_stoc = d1s_stoc[bkey][FAINT_KEY]

    if len(arr_fid) > 1:
        kde_fid = gaussian_kde(arr_fid, bw_method=d1s_cfg.bw_fid)
        ax.plot(x, kde_fid(x), color=d1s_cfg.color_fid, lw=3, label=d1s_cfg.label_fid)

    if len(arr_stoc) > 1:
        kde_stoc = gaussian_kde(arr_stoc, bw_method=d1s_cfg.bw_stoc)
        ax.plot(x, kde_stoc(x), color=d1s_cfg.color_stoc, lw=3, label=d1s_cfg.label_stoc)

    ax.set_xlabel(r"$d_{12}$ [cMpc]", fontsize=14)
    ax.set_title(rf"$M_{{UV,0}} < -{bkey.replace('M','')}$", fontsize=15)
    ax.legend(fontsize=12)
    ax.set_xlim(0, 8)

axes[0].set_ylabel(r"PDF (separation)", fontsize=14)

fig.suptitle(
    rf"$M_{{UV,\rm faint}} < -18.5$,  $z={REDSHIFT}$",
    fontsize=16, y=1.02
)
fig.tight_layout()
fig.savefig(OUTPUT, bbox_inches="tight")
print(f"Saved: {OUTPUT}")
