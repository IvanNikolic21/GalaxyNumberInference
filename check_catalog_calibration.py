#!/usr/bin/env python
"""
check_catalog_calibration.py
-----------------------------
Network-independent sanity check: does the freshly generated catalog for
(Muv_add, sigmaUV_a, sigmaUV_b) actually realize that scatter/shift, or is
there a generation-side bug? Bins halos near a target log(Mh) and compares
the empirical M_UV median/std against the intended pMuv_Mh prediction.

Usage
-----
    python check_catalog_calibration.py
"""

import numpy as np

from galaxy_neighbors import load_halo_catalog, load_muv_catalog
from generate_catalog_database import load_muv_mh_dict, median_muv, sigma_uv

HALO_CATALOG_PATH = "/lustre/astro/ivannik/21cmFAST_cache/d12b21e80b7885d62d31717c2c2d8421/1952/ffa852ccaa39d8f82951cc98ff798ab4/10.5000/HaloCatalog.h5"
MUV_MH_FILE       = "/groups/astro/ivannik/notebooks/clustering_project/Muv_Mh_z=10.txt"
CATALOG_PATH      = "/lustre/astro/ivannik/catalogs_grid_prior/catalog_Madd0p30_sa-0p34_sb0p60.h5"

MUV_ADD, SIGMA_A, SIGMA_B = 0.3, -0.34, 0.6
LOGMH_TARGET = 11.0
LOGMH_HALF_WIDTH = 0.1   # bin: [target-halfwidth, target+halfwidth]

# ---------------------------------------------------------------------------
_, logMh = load_halo_catalog(HALO_CATALOG_PATH)
muvs = load_muv_catalog(CATALOG_PATH, index=0)
print(f"Loaded {len(logMh)} halos, {len(muvs)} Muv samples "
      f"({'ALIGNED' if len(logMh) == len(muvs) else 'MISMATCHED LENGTH -- STOP, something is wrong'})")

muv_mh_dict = load_muv_mh_dict(MUV_MH_FILE)

sel = np.abs(logMh - LOGMH_TARGET) < LOGMH_HALF_WIDTH
n_sel = sel.sum()
print(f"\n{n_sel} halos with log(Mh) in [{LOGMH_TARGET-LOGMH_HALF_WIDTH:.2f}, "
      f"{LOGMH_TARGET+LOGMH_HALF_WIDTH:.2f}]")

if n_sel < 30:
    print("WARNING: very few halos in this bin -- widen LOGMH_HALF_WIDTH before trusting the numbers below.")

empirical_median = np.median(muvs[sel])
empirical_std    = np.std(muvs[sel])

expected_median = median_muv(np.array([LOGMH_TARGET]), muv_mh_dict)[0] + MUV_ADD
expected_sigma  = sigma_uv(np.array([LOGMH_TARGET]), SIGMA_A, SIGMA_B)[0]

print(f"\nAt log(Mh) = {LOGMH_TARGET}:")
print(f"  median M_UV : empirical = {empirical_median:+.3f}   expected = {expected_median:+.3f}   "
      f"diff = {empirical_median - expected_median:+.3f}")
print(f"  sigma_UV    : empirical = {empirical_std:.3f}   expected = {expected_sigma:.3f}   "
      f"ratio = {empirical_std / expected_sigma:.2f}x")

if abs(empirical_std / expected_sigma - 1) > 0.15 or abs(empirical_median - expected_median) > 0.15:
    print("\n  ==> MISMATCH: the catalog does not realize the intended (Muv_add, sigmaUV_a, sigmaUV_b).")
    print("      This points to a bug in catalog generation, not the trained networks.")
else:
    print("\n  ==> Catalog matches its intended parameters within tolerance.")
    print("      The bias is NOT a catalog-generation problem -- look elsewhere "
          "(training-database settings, network calibration).")