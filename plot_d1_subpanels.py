#!/usr/bin/env python
"""
plot_d1s_threepanel.py
----------------------
One figure, three panels:
  Left:   vary Muv,lim = -17.5, -18.5, -19.5  (fixed Muv,0=-21.5, z=10.5)
  Middle: vary Muv,0   = -21.0, -21.5, -22.0  (fixed Muv,lim=-18.5, z=10.5)
  Right:  vary z       = 8, 12, 14             (fixed Muv,0=-21.5, Muv,lim=-18.5)

Color intensity encodes the varying parameter within each panel.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

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
    8.0:  (CACHE_ROOT / "z8.0"  / "d1s_fiducial_real1.npz",
            CACHE_ROOT / "z8.0"  / "d1s_stochastic_real1.npz"),
    10.5: (CACHE_ROOT / "z10.5" / "d1s_fiducial_real5.npz",
            CACHE_ROOT / "z10.5" / "d1s_stochastic_real5.npz"),
    12.0: (CACHE_ROOT / "z12.0" / "d1s_fiducial_real50.npz",
            CACHE_ROOT / "z12.0" / "d1s_stochastic_real50.npz"),
    14.0: (CACHE_ROOT / "z14.0" / "d1s_fiducial_real100.npz",
            CACHE_ROOT / "z14.0" / "d1s_stochastic_real100.npz"),
}

# ---------------------------------------------------------------------------
# Colors — light to dark as parameter increases in magnitude
# ---------------------------------------------------------------------------
colors_fid  = np.flip(['#d94701', '#fd8d3c', '#fdbe85'])
colors_stoc = np.flip(['#2171b5', '#6baed6', '#bdd7e7'])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
x = np.linspace(0, 8, 300)

def plot_kde(ax, arr, color, label=None, lw=2.5, bw=0.15):
    if len(arr) < 2:
        return
    try:
        kde = gaussian_kde(arr, bw_method=bw)
        ax.plot(x, kde(x), color=color, lw=lw, label=label)
    except Exception:
        pass

def style_ax(ax, title, show_ylabel=False):
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.7)
    ax.set_xlabel(r"$d_1$ [cMpc]", fontsize=13)
    if show_ylabel:
        ax.set_ylabel(r"PDF$(d_1)$", fontsize=13)
    ax.set_title(title, fontsize=13)

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
ax = axes[0]
FAINT_KEYS = ["M17.5", "M18.5", "M19.5"]
FAINT_LABS = [r"$M_{\rm UV,lim}=-17.5$",
              r"$M_{\rm UV,lim}=-18.5$",
              r"$M_{\rm UV,lim}=-19.5$"]
BKEY = "M21.5"

for i, (fkey, lab) in enumerate(zip(FAINT_KEYS, FAINT_LABS)):
    plot_kde(ax, fid_105[BKEY][fkey],  colors_fid[i],  label=lab if i == 0 else None)
    plot_kde(ax, stoc_105[BKEY][fkey], colors_stoc[i])

# Custom legend showing both models + parameter variation
from matplotlib.lines import Line2D
legend_elements = (
    [Line2D([0], [0], color=colors_fid[i],  lw=2.5, label=lab) for i, lab in enumerate(FAINT_LABS)] +
    [Line2D([0], [0], color='gray', lw=2.5, ls='-',  label='intrinsically bright'),
     Line2D([0], [0], color='gray', lw=2.5, ls='--', label='increased stochasticity')]
)
style_ax(ax, r"Varying $M_{\rm UV,lim}$" + "\n" + r"$M_{\rm UV,0}=-21.5$, $z=10.5$",
         show_ylabel=True)

# --- Panel 2: vary Muv,0 ---
ax = axes[1]
BRIGHT_KEYS = ["M21", "M21.5", "M22"]
BRIGHT_LABS = [r"$M_{\rm UV,0}=-21.0$",
               r"$M_{\rm UV,0}=-21.5$",
               r"$M_{\rm UV,0}=-22.0$"]
FKEY = "M18.5"

for i, (bkey, lab) in enumerate(zip(BRIGHT_KEYS, BRIGHT_LABS)):
    plot_kde(ax, fid_105[bkey][FKEY],  colors_fid[i])
    plot_kde(ax, stoc_105[bkey][FKEY], colors_stoc[i])

style_ax(ax, r"Varying $M_{\rm UV,0}$" + "\n" + r"$M_{\rm UV,lim}=-18.5$, $z=10.5$")

# --- Panel 3: vary z ---
ax = axes[2]
REDSHIFTS = [8.0, 12.0, 14.0]
Z_LABS    = [r"$z=8$", r"$z=12$", r"$z=14$"]
BKEY = "M21.5"
FKEY = "M18.5"

for i, (z, zlab) in enumerate(zip(REDSHIFTS, Z_LABS)):
    fid_z, stoc_z = [load_d1s(p, cfg) for p in CACHE[z]]
    plot_kde(ax, fid_z[BKEY][FKEY],  colors_fid[i],  label=zlab)
    plot_kde(ax, stoc_z[BKEY][FKEY], colors_stoc[i])

style_ax(ax, r"Varying $z$" + "\n" + r"$M_{\rm UV,0}=-21.5$, $M_{\rm UV,lim}=-18.5$")

# --- Shared legend ---
# One legend for all panels: color = parameter value, solid = fid, dashed = stoc
# Add dashed versions for stoc on panel 3
from matplotlib.patches import Patch
legend_fid  = [Patch(color=c, label=lab) for c, lab in zip(colors_fid,  Z_LABS)]
legend_stoc = [Patch(color=c, label=lab) for c, lab in zip(colors_stoc, Z_LABS)]

import matplotlib.patches as mpatches

param_labels = [
    [r"$-17.5$", r"$-18.5$", r"$-19.5$"],   # panel 0: Muv,lim
    [r"$-21.0$", r"$-21.5$", r"$-22.0$"],    # panel 1: Muv,0
    [r"$8$",     r"$12$",    r"$14$"],        # panel 2: z
]
param_xlabels = [
    r"$M_{\rm UV,lim}=$",
    r"$M_{\rm UV,0}=$",
    r"         $z=$",
]

for ax, plabs, pxlab in zip(axes, param_labels, param_xlabels):
    x0, y0 = 0.35, 0.85
    dx = 0.11

    # Fiducial row
    for i, c in enumerate(colors_fid):
        rect = mpatches.Rectangle(
            (x0 + i*dx, y0 - 0.03), 0.04, 0.02,
            transform=ax.transAxes, facecolor=c, edgecolor='none'
        )
        ax.add_patch(rect)
    ax.text(x0 + 3*dx + 0.02, y0 - 0.04, 'intrinsically\nbright',
            fontsize=12, transform=ax.transAxes)

    # Stochastic row
    for i, c in enumerate(colors_stoc):
        rect = mpatches.Rectangle(
            (x0 + i*dx, y0 - 0.15), 0.04, 0.02,
            transform=ax.transAxes, facecolor=c, edgecolor='none'
        )
        ax.add_patch(rect)
    ax.text(x0 + 3*dx + 0.02, y0 - 0.18, 'increased\nstochasticity',
            fontsize=12, transform=ax.transAxes)

    # Parameter value labels
    ax.text(x0 - 0.22, y0 - 0.25, pxlab, fontsize=11, transform=ax.transAxes)
    for i, lab in enumerate(plabs):
        ax.text(x0 + i*dx -0.01 , y0 - 0.25, lab, fontsize=8, transform=ax.transAxes)

# Global note on line meaning
# fig.text(0.5, -0.02,
#          "Solid = intrinsically bright,  Dashed = increased stochasticity  [dashed not shown — colors match]",
#          ha='center', fontsize=10, style='italic')
fig.subplots_adjust(wspace=0.02, hspace=0.02)

#fig.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, bbox_inches="tight")
print(f"Saved: {OUTPUT}")