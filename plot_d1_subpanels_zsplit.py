#!/usr/bin/env python
"""
plot_d1_subpanels_zsplit.py
-----------------------------
Revision of plot_d1_subpanels.py's Fig. 6 (per Charlotte's feedback that the
original single combined z-dependence panel "doesn't look very nice"):
panels 1-2 (varying M_UV,lim and M_UV,0 at z=10.5) are unchanged, but the
third panel -- which used to overlay all 6 curves (3 redshifts x 2 models)
in one axes -- is now split into two panels, one per model, each showing
only its own 3-redshift fan.

Why: varying z isn't just a rescaling of each model's curve the way varying
M_UV,lim/M_UV,0 is -- it's a regime change (the high luminosity model
develops an extended tail while the high stochasticity model goes nearly
flat), so the two hues' curves cross each other repeatedly in the original
combined panel. Splitting by model removes the cross-hue tangle entirely;
each sub-panel is then a clean, non-crossing 3-shade fan, same visual
grammar as panels 1-2.

This is now a 4-panel figure. A separate script (not a modification of
plot_d1_subpanels.py) so the original 3-panel version stays available for
comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path
from matplotlib.lines import Line2D

from galaxy_neighbors import AnalysisConfig
from galaxy_d1s import load_d1s, D1sConfig

# ---------------------------------------------------------------------------
# Config -- identical to plot_d1_subpanels.py
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
OUTPUT     = Path("/groups/astro/ivannik/projects/Neighbors/plots/d1s_fourpanel_zsplit.pdf")

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
# Colors -- same ramps as plot_d1_subpanels.py, for consistency with panels 1-2
# ---------------------------------------------------------------------------
COLORS_FID  = ['#fdbe85', '#fd8d3c', '#d94701']   # light -> dark orange
COLORS_STOC = ['#bdd7e7', '#6baed6', '#2171b5']   # light -> dark blue
COLORS_FID_REV  = COLORS_FID[::-1]
COLORS_STOC_REV = COLORS_STOC[::-1]

LINEWIDTHS = [2.4, 3.2, 4.0]
LINEWIDTHS_REV = LINEWIDTHS[::-1]

GRAYS = ['#bdbdbd', '#636363', '#252525']
GRAYS_REV = GRAYS[::-1]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
x = np.linspace(0, 8, 300)

def plot_kde(ax, arr, color, label=None, lw=2.5, bw=0.2):
    if len(arr) < 2:
        return
    try:
        kde = gaussian_kde(arr, bw_method=bw)
        ax.plot(x, kde(x), color=color, lw=lw, label=label)
    except Exception:
        pass

def add_param_legend(ax, labels, colors, linewidths=None, loc='upper right'):
    """Line legend mapping shade+linewidth -> parameter values. Unlike the
    cross-hue panels (1-2), each z-sub-panel here is single-hue, so the
    legend swatches use that panel's own color ramp directly instead of a
    neutral gray -- there's no cross-hue ambiguity left to avoid."""
    lw_list = LINEWIDTHS if linewidths is None else linewidths
    handles = [Line2D([0], [0], color=c, lw=lw, label=lab)
               for c, lw, lab in zip(colors, lw_list, labels)]
    return ax.legend(handles=handles, loc=loc, fontsize=13, frameon=False, handlelength=1.5)

def add_model_legend(ax, loc='upper right'):
    handles = [
        Line2D([0], [0], color=COLORS_FID[1],  lw=3, label='High\nluminosity'),
        Line2D([0], [0], color=COLORS_STOC[1], lw=3, label='High\nstochasticity'),
    ]
    return ax.legend(handles=handles, loc=loc, fontsize=15, frameon=False, handlelength=1.5)

def style_ax(ax, title, show_ylabel=False):
    ax.set_xlim(0.1, 4)
    ax.set_ylim(0, 1.4)
    ax.set_xlabel(r"$d_1$ [cMpc]", fontsize=13)
    if show_ylabel:
        ax.set_ylabel(r"p$(d_1)$", fontsize=13)
    ax.set_title(title, fontsize=13)

# ---------------------------------------------------------------------------
# Load z=10.5 once -- used in panels 1 and 2
# ---------------------------------------------------------------------------
fid_105, stoc_105 = [load_d1s(p, cfg) for p in CACHE[10.5]]

# ---------------------------------------------------------------------------
# Plot -- now 4 panels
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-ticks")
plt.rcParams.update({"font.size": 13, "xtick.top": True, "ytick.right": True,
                     "xtick.direction": "in", "ytick.direction": "in"})

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)

# --- Panel 1: vary Muv,0 (unchanged from plot_d1_subpanels.py) ---
ax = axes[0]
BRIGHT_KEYS = ["M22", "M21.5", "M21"]
BRIGHT_LABS = [r"$M_{\rm UV,0}=-22.0$",
               r"$M_{\rm UV,0}=-21.5$",
               r"$M_{\rm UV,0}=-21.0$"]
FKEY = "M18.5"

for i, (bkey, lab) in enumerate(zip(BRIGHT_KEYS, BRIGHT_LABS)):
    bw_fid = 0.4 if bkey == "M22" else 0.3
    plot_kde(ax, fid_105[bkey][FKEY],  COLORS_FID_REV[i],  lw=LINEWIDTHS_REV[i], bw=bw_fid)
    plot_kde(ax, stoc_105[bkey][FKEY], COLORS_STOC_REV[i], lw=LINEWIDTHS_REV[i], bw=0.3)

style_ax(ax, "Varying UV magnitude of the\nbright galaxy, " + r"$M_{\rm UV,0}$" + "\n" + r"$M_{\rm UV,lim}=-18.5$, $z=10.5$", show_ylabel=True)
model_leg = add_model_legend(ax)
ax.add_artist(model_leg)
add_param_legend(ax, BRIGHT_LABS, colors=GRAYS_REV, linewidths=LINEWIDTHS_REV)

# --- Panel 2: vary Muv,lim (unchanged from plot_d1_subpanels.py) ---
ax = axes[1]
FAINT_KEYS = ["M17.5", "M18.5", "M19.5"]
FAINT_LABS = [r"$M_{\rm UV,lim}=-17.5$",
              r"$M_{\rm UV,lim}=-18.5$",
              r"$M_{\rm UV,lim}=-19.5$"]
BKEY = "M21.5"

for i, (fkey, lab) in enumerate(zip(FAINT_KEYS, FAINT_LABS)):
    plot_kde(ax, fid_105[BKEY][fkey],  COLORS_FID_REV[i],  lw=LINEWIDTHS_REV[i], bw=0.3)
    plot_kde(ax, stoc_105[BKEY][fkey], COLORS_STOC_REV[i], lw=LINEWIDTHS_REV[i], bw=0.3)

style_ax(ax, "Varying UV magnitude of \nphotometric neighbors, " + r"$M_{\rm UV,lim}$" + "\n" + r"$M_{\rm UV,0}=-21.5$, $z=10.5$")
add_param_legend(ax, FAINT_LABS, colors=GRAYS_REV, linewidths=LINEWIDTHS_REV)

# --- Panels 3 & 4: vary z, split by model instead of overlaid ---
REDSHIFTS = [14.0, 12.0, 8.0]
Z_LABS    = [r"$z=14$", r"$z=12$", r"$z=8$"]
BKEY = "M21.5"
FKEY = "M18.5"

# Bandwidths per redshift, matching plot_d1_subpanels.py
BW_BY_Z = {14.0: 0.3, 12.0: 0.3, 8.0: 0.15}

z_data = {}
for z in REDSHIFTS:
    fid_z, stoc_z = [load_d1s(p, cfg) for p in CACHE[z]]
    z_data[z] = (fid_z, stoc_z)

ax = axes[2]
for i, (z, zlab) in enumerate(zip(REDSHIFTS, Z_LABS)):
    fid_z, _ = z_data[z]
    plot_kde(ax, fid_z[BKEY][FKEY], COLORS_FID_REV[i], lw=LINEWIDTHS_REV[i], bw=BW_BY_Z[z])
style_ax(ax, "High luminosity model:\nvarying redshift, $z$" + "\n" + r"$M_{\rm UV,0}=-21.5$, $M_{\rm UV,lim}=-18.5$")
add_param_legend(ax, Z_LABS, colors=COLORS_FID_REV, linewidths=LINEWIDTHS_REV)

ax = axes[3]
for i, (z, zlab) in enumerate(zip(REDSHIFTS, Z_LABS)):
    _, stoc_z = z_data[z]
    plot_kde(ax, stoc_z[BKEY][FKEY], COLORS_STOC_REV[i], lw=LINEWIDTHS_REV[i], bw=BW_BY_Z[z])
style_ax(ax, "High stochasticity model:\nvarying redshift, $z$" + "\n" + r"$M_{\rm UV,0}=-21.5$, $M_{\rm UV,lim}=-18.5$")
add_param_legend(ax, Z_LABS, colors=COLORS_STOC_REV, linewidths=LINEWIDTHS_REV)

fig.subplots_adjust(wspace=0.04, hspace=0.04)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, bbox_inches="tight")
print(f"Saved: {OUTPUT}")
