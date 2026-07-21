#!/usr/bin/env python
"""
plot_d1s_threepanel.py
----------------------
One figure, three panels:
  Left:   vary Muv,lim = -17.5, -18.5, -19.5  (fixed Muv,0=-21.5, z=10.5)
  Middle: vary Muv,0   = -21.0, -21.5, -22.0  (fixed Muv,lim=-18.5, z=10.5)
  Right:  vary z       = 8, 12, 14             (fixed Muv,0=-21.5, Muv,lim=-18.5)

Color encodes the model (intrinsically bright vs increased stochasticity,
shared across all three panels); linestyle encodes the varying parameter
within each panel.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.lines import Line2D

from galaxy_neighbors import AnalysisConfig
from galaxy_d1s import load_d1s, D1sConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
cfg = AnalysisConfig(
    bright_limits         = [-20.5, -20.75, -21.0, -21.25, -21.5, -21.75, -22.0],
    faint_limits          = [-16.5, -16.6, -16.7, -16.8, -16.9, -17.0, -17.1, -17.2,
                             -17.3, -17.4, -17.5, -17.6, -17.7, -17.8, -17.9, -18.0,
                             -18.1, -18.2, -18.3, -18.4, -18.5, -18.6, -18.7, -18.8,
                             -18.9, -19.0, -19.1, -19.2, -19.3, -19.4, -19.5, -19.6],
    preselect_faint_limit = -16.5,
    survey_area_arcmin2   = 12.24,
)

d1s_cfg = D1sConfig()

CACHE_ROOT = Path("/groups/astro/ivannik/projects/Neighbors/cache")
OUTPUT     = Path("/groups/astro/ivannik/projects/Neighbors/plots/d1s_threepanel.pdf")

CACHE = {
    8.0:  (CACHE_ROOT / "z8.0"  / "d1s_fiducial_real2.npz",
            CACHE_ROOT / "z8.0"  / "d1s_stochastic_real2.npz"),
    10.5: (CACHE_ROOT / "z10.5" / "d1s_fiducial_real20.npz",
            CACHE_ROOT / "z10.5" / "d1s_stochastic_real20.npz"),
    12.0: (CACHE_ROOT / "z12.0" / "d1s_fiducial_real100.npz",
            CACHE_ROOT / "z12.0" / "d1s_stochastic_real100.npz"),
    14.0: (CACHE_ROOT / "z14.0" / "d1s_fiducial_real200.npz",
            CACHE_ROOT / "z14.0" / "d1s_stochastic_real200.npz"),
}

# ---------------------------------------------------------------------------
# Colors — one per model, shared across all panels
# ---------------------------------------------------------------------------
COLOR_FID  = '#fd8d3c'
COLOR_STOC = '#08519c'

# Linestyles — one per varying-parameter value within a panel. Each panel
# uses LINESTYLES_REV (reversed) so that its most extreme value -- the one
# expected to show the clearest fiducial/stochastic difference -- gets the
# solid line, since solid draws the eye first.
LINESTYLES = ['-', '--', ':']
LINESTYLES_REV = LINESTYLES[::-1]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
x = np.linspace(0, 8, 300)

def plot_kde(ax, arr, color, label=None, lw=2.5, bw=0.2, ls='-'):
    if len(arr) < 2:
        return
    try:
        kde = gaussian_kde(arr, bw_method=bw)
        ax.plot(x, kde(x), color=color, lw=lw, ls=ls, label=label)
    except Exception:
        pass

def add_param_legend(ax, labels, linestyles=None, loc='upper right'):
    """Black-line legend mapping this panel's linestyles -> parameter values."""
    ls_list = LINESTYLES_REV if linestyles is None else linestyles
    handles = [Line2D([0], [0], color='black', lw=2.5, ls=ls, label=lab)
               for ls, lab in zip(ls_list, labels)]
    return ax.legend(handles=handles, loc=loc, fontsize=12, frameon=False, handlelength=3)

def add_model_legend(ax, loc='upper left'):
    """Color legend mapping model -> color, meant to be shown on one panel only."""
    handles = [
        Line2D([0], [0], color=COLOR_FID,  lw=3, label='intrinsically bright'),
        Line2D([0], [0], color=COLOR_STOC, lw=3, label='increased stochasticity'),
    ]
    return ax.legend(handles=handles, loc=loc, fontsize=13, frameon=False, handlelength=2.5)

def style_ax(ax, title, show_ylabel=False):
    ax.set_xlim(0.1, 4)
    #ax.set_xscale('log')
    ax.set_ylim(0, 1.4)
    ax.set_xlabel(r"$d_1$ [cMpc]", fontsize=13)
    if show_ylabel:
        ax.set_ylabel(r"PDF$(d_1)$", fontsize=13)
    ax.set_title(title, fontsize=13)
    # ticks = [0.6, 1, 2, 3, 4]
    # ax.xaxis.set_major_locator(FixedLocator(ticks))
    # ax.xaxis.set_major_formatter(FixedFormatter(['0.6', '1', '2', '3', '4']))


# ---------------------------------------------------------------------------
# Load z=10.5 once — used in panels 1 and 2
# ---------------------------------------------------------------------------
fid_105, stoc_105 = [load_d1s(p, cfg) for p in CACHE[10.5]]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-ticks")
plt.rcParams.update({"font.size": 13, "xtick.top": True, "ytick.right": True,
                     "xtick.direction": "in", "ytick.direction": "in"})

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

# --- Panel 1: vary Muv,lim ---
ax = axes[1]
FAINT_KEYS = ["M17.5", "M18.5", "M19.5"]
FAINT_LABS = [r"$M_{\rm UV,lim}=-17.5$",
              r"$M_{\rm UV,lim}=-18.5$",
              r"$M_{\rm UV,lim}=-19.5$"]
BKEY = "M21.5"

for i, (fkey, lab) in enumerate(zip(FAINT_KEYS, FAINT_LABS)):
    plot_kde(ax, fid_105[BKEY][fkey],  COLOR_FID,  bw=0.3, ls=LINESTYLES_REV[i])
    plot_kde(ax, stoc_105[BKEY][fkey], COLOR_STOC, bw=0.3, ls=LINESTYLES_REV[i])

style_ax(ax, r"Varying $M_{\rm UV,lim}$" + "\n" + r"$M_{\rm UV,0}=-21.5$, $z=10.5$")
add_param_legend(ax, FAINT_LABS)

# --- Panel 2: vary Muv,0 ---
ax = axes[0]
BRIGHT_KEYS = ["M21", "M21.5", "M22"]
BRIGHT_LABS = [r"$M_{\rm UV,0}=-21.0$",
               r"$M_{\rm UV,0}=-21.5$",
               r"$M_{\rm UV,0}=-22.0$"]
FKEY = "M18.5"

for i, (bkey, lab) in enumerate(zip(BRIGHT_KEYS, BRIGHT_LABS)):
    if bkey == "M22":
        plot_kde(ax, fid_105[bkey][FKEY], COLOR_FID,  bw=0.4, ls=LINESTYLES_REV[i])
        plot_kde(ax, stoc_105[bkey][FKEY], COLOR_STOC, bw=0.3, ls=LINESTYLES_REV[i])
    else:
        plot_kde(ax, fid_105[bkey][FKEY],  COLOR_FID,  bw = 0.3, ls=LINESTYLES_REV[i])
        plot_kde(ax, stoc_105[bkey][FKEY], COLOR_STOC, bw = 0.3, ls=LINESTYLES_REV[i])

style_ax(ax, r"Varying $M_{\rm UV,0}$" + "\n" + r"$M_{\rm UV,lim}=-18.5$, $z=10.5$", show_ylabel=True)
# Shared model-color legend lives on this panel only.
model_leg = add_model_legend(ax, loc='upper left')
ax.add_artist(model_leg)
add_param_legend(ax, BRIGHT_LABS)

# --- Panel 3: vary z ---
ax = axes[2]
REDSHIFTS = [8.0, 12.0, 14.0]
Z_LABS    = [r"$z=8$", r"$z=12$", r"$z=14$"]
BKEY = "M21.5"
FKEY = "M18.5"

for i, (z, zlab) in enumerate(zip(REDSHIFTS, Z_LABS)):
    fid_z, stoc_z = [load_d1s(p, cfg) for p in CACHE[z]]
    if z==12:
        BW_FID = 0.3
        BW_STOC = 0.3
    elif z==14:
        BW_FID = 0.3
        BW_STOC = 0.3
    else:
        BW_FID = 0.15
        BW_STOC = 0.15
    plot_kde(ax, fid_z[BKEY][FKEY],  COLOR_FID,  bw = BW_FID, ls=LINESTYLES_REV[i])
    plot_kde(ax, stoc_z[BKEY][FKEY], COLOR_STOC, bw = BW_STOC, ls=LINESTYLES_REV[i])

style_ax(ax, r"Varying $z$" + "\n" + r"$M_{\rm UV,0}=-21.5$, $M_{\rm UV,lim}=-18.5$")
add_param_legend(ax, Z_LABS)

fig.subplots_adjust(wspace=0.04, hspace=0.04)

#fig.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, bbox_inches="tight")
print(f"Saved: {OUTPUT}")