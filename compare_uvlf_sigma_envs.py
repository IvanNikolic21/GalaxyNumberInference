#!/usr/bin/env python
"""
compare_uvlf_sigma_envs.py
--------------------------
Diagnostic (2026-10-23): the same theta gets a different UVLF-consistency
sigma in diagnose_heldout_calibration.py (run in the galaxy-neighbors env,
hmf 3.5.2) and diagnose_information_floor.py (UVLF_clust env, hmf 3.6.1),
although both call uvlf_consistency.uvlf_sigma_per_theta on the repo's
uvlf.py. Suspected cause: the hmf version (transfer-function extrapolation
default, 'extrapolate_with_eh'). Run this in BOTH envs and compare.

theta below: A = passed the floor script's cut but got > 5 sigma in the
held-out npz; B = the reverse; C = both agree (<= 5). 'heldout' is the
sigma stored by the galaxy-neighbors run.

    conda activate galaxy-neighbors && python compare_uvlf_sigma_envs.py
    conda activate UVLF_clust      && python compare_uvlf_sigma_envs.py
"""
import numpy as np
import hmf
from uvlf_consistency import uvlf_sigma_per_theta

THETA = [  # (label, theta, sigma stored by the held-out run)
    ("A", [-0.13740962, 0.62263019, 2.66901319], 6.34),
    ("A", [-0.21753181, -0.03181574, 1.29334267], 10.87),
    ("A", [0.56439693, -0.53177901, 0.20031353], 7.44),
    ("B", [1.49610783, 0.11584202, 1.59171734], 4.93),
    ("B", [1.8367387, -0.20718671, 1.10166635], 4.77),
    ("B", [0.30404191, -0.18905759, 0.27182546], 4.34),
    ("C", [0.25376943, 0.41916913, 2.22427323], 1.42),
    ("C", [1.39526705, 0.11393325, 1.77419668], 3.19),
    ("C", [-0.02752331, 0.16310532, 1.40711142], 0.30),
]

sig = uvlf_sigma_per_theta(np.array([t for _, t, _ in THETA]))
print(f"hmf {hmf.__version__}")
for (lab, t, s_old), s in zip(THETA, sig):
    print(f"  {lab}  theta={t}  sigma here {s:6.2f}   held-out npz {s_old:6.2f}")
