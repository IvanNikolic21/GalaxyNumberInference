#!/usr/bin/env python
"""
compare_hmf_envs.py
-------------------
Diagnostic (2026-10-23): hmf 3.5.2 (galaxy-neighbors) and 3.6.1 (UVLF_clust)
report identical defaults (Planck18, CAMB, Tinker08, SOMean(200),
sigma_8 = 0.8159) yet dn/dm at z = 10 differs by x1.6 at 1e10 and x2.4 at
1e11 Msun/h. This prints every intermediate of the HMF chain
  P(k) -> sigma(M) -> f(sigma) -> dn/dm
plus the CAMB version, at z = 0 and z = 10, with the default CAMB transfer
and with the analytic Eisenstein & Hu (1998) transfer (no CAMB). The first
quantity that differs between the two envs is where the change lives.
Run in both envs and diff the outputs.
"""
import importlib.metadata as md
import numpy as np
import hmf

def ver(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"

print(f"hmf {hmf.__version__} | camb {ver('camb')} | astropy {ver('astropy')} | numpy {np.__version__}")
for transfer in ("CAMB", "EH"):
    for z in (0.0, 10.0):
        mf = hmf.MassFunction(z=z, Mmin=8, Mmax=14, dlog10m=0.05, transfer_model=transfer)
        i = [np.argmin(abs(np.log10(mf.m) - lm)) for lm in (10, 11)]
        k_i = [np.argmin(abs(np.log(mf.k) - np.log(kk))) for kk in (0.1, 1.0, 10.0)]
        print(f"[{transfer:4s} z={z:4.1f}] growth={mf.growth_factor:.5f}  "
              f"P(k=0.1,1,10)={np.array2string(mf.power[k_i], precision=5)}  "
              f"sigma(1e10,1e11)={np.array2string(mf.sigma[i], precision=5)}  "
              f"f(sigma)={np.array2string(mf.fsigma[i], precision=5)}  "
              f"dndm={np.array2string(mf.dndm[i], precision=4)}")
print("Tinker08 params:", hmf.MassFunction(z=10.0).hmf.params)
