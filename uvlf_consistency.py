"""
uvlf_consistency.py
-------------------
UVLF consistency of theta = (Muv_add, sigmaUV_a, sigmaUV_b) against the
Donnan+24 z~10 UVLF, as an equivalent Gaussian sigma. Shared by the
halo-mass diagnostics and build_nre_database.py (--max-uvlf-sigma); kept
free of torch so the database builder doesn't need it.
"""
import numpy as np
from scipy.stats import chi2, norm

from uvlf import Mason15, Observations


def uvlf_sigma_per_theta(params: np.ndarray) -> np.ndarray:
    """Equivalent Gaussian sigma of each theta's UVLF fit to Donnan+24.

    Identical chi2 -> p -> sigma conversion as plot_eval_vs_truth_uvlf_consistent.py
    and HaloMassDataset's --uvlf-reweight block (see the former's docstring
    for why the /2 is needed); cached per distinct theta, since every
    environment of one database file shares that file's theta.
    """
    mason15 = Mason15(z=10.0)
    obs = Observations(ang=False, uvlf=True)
    muv_obs, phi_obs, (sig_p, sig_m) = obs.get_obs_uvlf_z10_Donnan24()
    muv_grid = np.linspace(-25, -13, 100)
    dof = len(muv_obs)
    cache = {}
    out = np.empty(len(params))
    for k, (muv_add, sa, sb) in enumerate(params):
        key = (round(float(muv_add), 8), round(float(sa), 8), round(float(sb), 8))
        if key not in cache:
            phi_pred = np.interp(muv_obs, muv_grid,
                                 mason15.calculate_UVLF(muv_add, sa, sb, Muv_grid=muv_grid))
            sigma_obs = np.where(phi_pred >= phi_obs, sig_p, sig_m)
            chi2_val = np.sum(((phi_obs - phi_pred) / sigma_obs) ** 2)
            pval = np.clip(chi2.sf(chi2_val, df=dof) / 2.0, 0.0, 1.0 - 1e-16)
            cache[key] = max(0.0, float(norm.isf(pval)))
        out[k] = cache[key]
    return out
