#!/usr/bin/env python
"""
plot_d1s_threepanel.py
----------------------
One figure, three panels:
  Left:   vary Muv,lim = -17.5, -18.5, -19.5  (fixed Muv,0=-21.5, z=10.5)
  Middle: vary Muv,0   = -21.0, -21.5, -22.0  (fixed Muv,lim=-18.5, z=10.5)
  Right:  vary z       = 8, 12, 14             (fixed Muv,0=-21.5, Muv,lim=-18.5)

Color hue encodes the model (intrinsically bright vs increased stochasticity,
shared across all three panels); shade of that hue -- plus linewidth, tied
together -- encodes the varying parameter within each panel. The
darkest/thickest line in each panel marks whichever value shows the
clearest fiducial-vs-stochastic difference, which is why the shade/width
order is reversed in the z panel relative to the other two (see comments
at each panel below).
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
# Colors — one 3-shade sequential ramp per model (light -> dark), shared hue
# across all panels; shade within a panel encodes the varying parameter.
# Exact ramps restored from the pre-linestyle version of this figure
# (commit bfb1f44's predecessor): ColorBrewer 3-class Oranges / Blues.
# ---------------------------------------------------------------------------
COLORS_FID  = ['#fdbe85', '#fd8d3c', '#d94701']   # light -> dark orange
COLORS_STOC = ['#bdd7e7', '#6baed6', '#2171b5']   # light -> dark blue
COLORS_FID_REV  = COLORS_FID[::-1]
COLORS_STOC_REV = COLORS_STOC[::-1]

# Linewidths — tied to shade: lightest shade is thinnest, darkest is
# thickest, so the two encodings reinforce the same visual hierarchy.
LINEWIDTHS = [2.4, 3.2, 4.0]
LINEWIDTHS_REV = LINEWIDTHS[::-1]

# Neutral grayscale ramp used only for the parameter-value legend swatches,
# so that legend doesn't imply the shade encoding is specific to one hue
# (ColorBrewer 3-class Greys, same design as the color ramps above).
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

def add_param_legend(ax, labels, grays=None, linewidths=None, loc='center right'):
    """Grayscale-line legend mapping this panel's shade+linewidth -> parameter
    values (gray rather than the model hue, since the shade/width encoding
    applies the same way to both models)."""
    gray_list = GRAYS if grays is None else grays
    lw_list   = LINEWIDTHS if linewidths is None else linewidths
    handles = [Line2D([0], [0], color=g, lw=lw, label=lab)
               for g, lw, lab in zip(gray_list, lw_list, labels)]
    return ax.legend(handles=handles, loc=loc, fontsize=14, frameon=False, handlelength=1.5)

def add_model_legend(ax, loc='upper right'):
    """Color legend mapping model -> hue, meant to be shown on one panel only.
    Uses the medium shade of each model's ramp as the representative swatch."""
    handles = [
        Line2D([0], [0], color=COLORS_FID[1],  lw=3, label='High\nluminosity'),
        Line2D([0], [0], color=COLORS_STOC[1], lw=3, label='High\nstochasticity'),
    ]
    return ax.legend(handles=handles, loc=loc, fontsize=15, frameon=False, handlelength=1.5)

def style_ax(ax, title, show_ylabel=False):
    ax.set_xlim(0.1, 4)
    #ax.set_xscale('log')
    ax.set_ylim(0, 1.4)
    ax.set_xlabel(r"$d_1$ [cMpc]", fontsize=13)
    if show_ylabel:
        ax.set_ylabel(r"p$(d_1)$", fontsize=13)
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

# Prominent (darkest/thickest) = first-listed = M_UV,lim=-17.5 -> use the
# REV ramps so index 0 (first-listed) maps to the darkest/thickest entry.
for i, (fkey, lab) in enumerate(zip(FAINT_KEYS, FAINT_LABS)):
    plot_kde(ax, fid_105[BKEY][fkey],  COLORS_FID_REV[i],  lw=LINEWIDTHS_REV[i], bw=0.3)
    plot_kde(ax, stoc_105[BKEY][fkey], COLORS_STOC_REV[i], lw=LINEWIDTHS_REV[i], bw=0.3)

style_ax(ax, "Varying UV magnitude of \nphotometric neighbors, " + r"$M_{\rm UV,lim}$" + "\n" + r"$M_{\rm UV,0}=-21.5$, $z=10.5$")
# Labels are already listed darkest-first (matches the REV plotting order above).
add_param_legend(ax, FAINT_LABS, grays=GRAYS_REV, linewidths=LINEWIDTHS_REV)

# --- Panel 2: vary Muv,0 ---
ax = axes[0]
BRIGHT_KEYS = ["M22", "M21.5", "M21"]
BRIGHT_LABS = [r"$M_{\rm UV,0}=-22.0$",
               r"$M_{\rm UV,0}=-21.5$",
               r"$M_{\rm UV,0}=-21.0$"]
FKEY = "M18.5"

# Prominent (darkest/thickest) = first-listed = M_UV,0=-22.0 (brightest) -> use
# the REV ramps so index 0 (first-listed) maps to the darkest/thickest entry.
for i, (bkey, lab) in enumerate(zip(BRIGHT_KEYS, BRIGHT_LABS)):
    if bkey == "M22":
        plot_kde(ax, fid_105[bkey][FKEY], COLORS_FID_REV[i],  lw=LINEWIDTHS_REV[i], bw=0.4)
        plot_kde(ax, stoc_105[bkey][FKEY], COLORS_STOC_REV[i], lw=LINEWIDTHS_REV[i], bw=0.3)
    else:
        plot_kde(ax, fid_105[bkey][FKEY],  COLORS_FID_REV[i],  lw=LINEWIDTHS_REV[i], bw = 0.3)
        plot_kde(ax, stoc_105[bkey][FKEY], COLORS_STOC_REV[i], lw=LINEWIDTHS_REV[i], bw = 0.3)

style_ax(ax, "Varying UV magnitude of the\nbright galaxy, "+r"$M_{\rm UV,0}$" + "\n" + r"$M_{\rm UV,lim}=-18.5$, $z=10.5$", show_ylabel=True)
# Shared model-color legend lives on this panel only.
model_leg = add_model_legend(ax)
ax.add_artist(model_leg)
# Labels are already listed darkest-first (matches the REV plotting order above).
add_param_legend(ax, BRIGHT_LABS, grays=GRAYS_REV, linewidths=LINEWIDTHS_REV)

# --- Panel 3: vary z ---
ax = axes[2]
REDSHIFTS = [14.0, 12.0, 8.0]
Z_LABS    = [r"$z=14$", r"$z=12$", r"$z=8$"]
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
    # Prominent (darkest/thickest) = z=14, which is first-listed here -> REV
    # ramps, so index 0 (first-listed, z=14) lands on the darkest entry.
    plot_kde(ax, fid_z[BKEY][FKEY],  COLORS_FID_REV[i],  lw=LINEWIDTHS_REV[i], bw = BW_FID)
    plot_kde(ax, stoc_z[BKEY][FKEY], COLORS_STOC_REV[i], lw=LINEWIDTHS_REV[i], bw = BW_STOC)

style_ax(ax, r"Varying redshift, $z$" + "\n" + r"$M_{\rm UV,0}=-21.5$, $M_{\rm UV,lim}=-18.5$")
# Z_LABS is already listed darkest-first (z=14 first, matches the REV plotting
# order above), so no reversal needed here -- same pattern as panels 1 and 2.
add_param_legend(ax, Z_LABS, grays=GRAYS_REV, linewidths=LINEWIDTHS_REV)

fig.subplots_adjust(wspace=0.04, hspace=0.04)

#fig.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, bbox_inches="tight")
print(f"Saved: {OUTPUT}")