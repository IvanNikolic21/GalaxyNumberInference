#!/usr/bin/env python
"""
check_uvlf_consistency.py
---------------------------
Is the theta point that's been breaking NRE calibration (Muv_add=0.3,
sigmaUV_a=-0.34, sigmaUV_b=2.4) actually consistent with the observed UVLF,
or is it a physically implausible corner of the (uniform) prior that we've
been stress-testing the network against for no good physical reason?

Reuses the exact Mason15/Observations machinery and UVLF log-likelihood
already used in infer_nre.py's --use-uvlf path, so the "badness of fit"
number here is directly comparable to what the inference pipeline itself
would compute.

Usage
-----
    python check_uvlf_consistency.py
"""

import numpy as np
import matplotlib.pyplot as plt

from uvlf import Mason15, Observations

# The problem truth, and two points already calibrated to match the observed
# UVLF for comparison: Paper I's high-luminosity and high-stochasticity
# reference models (see Paper I Sec. 2 / this paper's Sec. 2.1).
POINTS = {
    "sigma_b=2.4 (the problem truth)": (0.3, -0.34, 2.4),
    "high-stochasticity ref. (sigma_b=0.6)": (0.3, -0.34, 0.6),
    "high-luminosity ref.": (-0.8, 0.0, 0.3),
}


def main():
    mason15 = Mason15(z=10.0)
    obs = Observations(ang=False, uvlf=True)
    muv_obs, phi_obs, (sig_p, sig_m) = obs.get_obs_uvlf_z10_Donnan24()

    muv_grid = np.linspace(-25, -13, 100)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.errorbar(muv_obs, phi_obs, yerr=(sig_m, sig_p), fmt='ko', label='Donnan+24 (z=10)', zorder=5)

    print(f"{'theta':<40s} {'chi2-like log-L':>16s}  {'phi at Muv=-20.75':>20s}  {'ratio to obs':>14s}")
    print("-" * 95)
    for label, (muv_add, sa, sb) in POINTS.items():
        phi_pred_grid = mason15.calculate_UVLF(muv_add, sa, sb, Muv_grid=muv_grid)
        phi_pred = np.interp(muv_obs, muv_grid, phi_pred_grid)
        sigma = np.where(phi_pred >= phi_obs, sig_p, sig_m)
        ll = -0.5 * np.sum(((phi_obs - phi_pred) / sigma) ** 2)

        # Compare at the brightest, best-constrained observed point
        phi_pred_bright = np.interp(muv_obs[0], muv_grid, phi_pred_grid)
        ratio = phi_pred_bright / phi_obs[0]

        print(f"{label:<40s} {ll:16.2f}  {phi_pred_bright:20.3e}  {ratio:14.2f}x")
        ax.plot(muv_grid, phi_pred_grid, label=f"{label}\n(Madd={muv_add}, sa={sa}, sb={sb})")

    ax.set_yscale('log')
    ax.set_xlabel(r'$M_{\rm UV}$')
    ax.set_ylabel(r'$\phi$ [mag$^{-1}$ cMpc$^{-3}$]')
    ax.legend(fontsize=8)
    ax.set_title('UVLF consistency check: is the problem truth physically plausible?')
    fig.tight_layout()
    fig.savefig("uvlf_consistency_check.png", dpi=150)
    print("\nSaved: uvlf_consistency_check.png")
    print("\nlog-L is the same UVLF log-likelihood infer_nre.py's --use-uvlf path computes "
          "(more negative = worse fit). Compare the problem truth's value to the two reference "
          "models', which were explicitly calibrated to match this same observed UVLF.")


if __name__ == "__main__":
    main()