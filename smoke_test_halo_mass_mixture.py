#!/usr/bin/env python
"""
smoke_test_halo_mass_mixture.py
-------------------------------
Fast (~1 min, CPU is fine) sanity checks for the --n-components / --use-bright-muv
additions to train_halo_mass.py, run before the v11 training job:
  1. K = 1: mixture_nll reproduces the original single-Gaussian (beta-)NLL exactly.
  2. Output shapes for the Deep Sets net with one extra scalar input + a 3-component
     head, and the unchanged 2-output head for K = 1 (old checkpoints still load).
  3. A toy skewed problem (Gaussian core + exponential low tail, like the v10 PIT):
     a K = 3 mixture should reach a lower NLL than K = 1 and one-sided coverage
     close to the diagonal; K = 1 should show the low-tail excess at small q.
"""
import math, numpy as np, torch
from train_halo_mass import (HaloMassNetwork, HaloMassSetNetwork, mixture_outputs, mixture_nll,
                             mixture_mean_std, mixture_cdf, _n_features)
torch.manual_seed(0)
# 1) K=1: mixture_nll == old Gaussian NLL (incl. beta)
mu = torch.randn(64); ls = torch.randn(64)*0.3; t = torch.randn(64); w = torch.rand(64)
for beta in (0.0, 0.5):
    s = torch.exp(ls)
    old = ls + 0.5*math.log(2*math.pi) + 0.5*((t-mu)/s)**2
    if beta: old = old * s**(2*beta)
    old = (old*w).sum()/w.sum()
    new = mixture_nll(torch.zeros(64,1), mu[:,None], ls[:,None], t, w, beta=beta)
    print(f"K=1 beta={beta}: old {old:.6f} new {new:.6f}")
    assert torch.allclose(old, new, atol=1e-5), "K=1 mixture NLL != original Gaussian NLL"
# 2) shapes: deepset with extra input + mixture; old-layout K=1 state_dict keys unchanged
n_feat = _n_features(True, False); D = 10*n_feat + 1 + 3 + 1
net = HaloMassSetNetwork(10, n_feat, [16,16], [32,32], 0.0, n_extra=1, n_components=3)
out = mixture_outputs(net, torch.randn(5, D)); print("deepset K=3 shapes", [tuple(o.shape) for o in out])
assert all(tuple(o.shape) == (5, 3) for o in out)
k1 = HaloMassNetwork(34, [32,32], 0.0); print("K=1 head", tuple(k1.output.weight.shape), "outputs", [tuple(o.shape) for o in k1(torch.randn(5,34))])
assert tuple(k1.output.weight.shape) == (2, 32)
# 3) toy skewed problem: y = x + skewed noise (long low tail); fit K=1 vs K=3, check one-sided PIT
N = 40000
x = torch.rand(N,1)*2-1
noise = torch.where(torch.rand(N,1) < 0.25, -torch.distributions.Exponential(2.0).sample((N,1)), 0.1*torch.randn(N,1))
y = (10 + x + noise).squeeze(1)
nlls = {}
for K in (1, 3):
    torch.manual_seed(1)
    m = HaloMassNetwork(1, [64,64], 0.0, n_components=K)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for it in range(1500):
        idx = torch.randint(0, N, (512,))
        loss = mixture_nll(*mixture_outputs(m, x[idx]), y[idx], torch.ones(512), beta=0.0)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        o = mixture_outputs(m, x); pit = mixture_cdf(*o, y).numpy()
        nll = mixture_nll(*o, y, torch.ones(N)).item()
    nlls[K] = nll
    q = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    print(f"K={K}: NLL {nll:.3f}  one-sided cov at q={q.tolist()}: {np.round([np.mean(pit<qq) for qq in q],3).tolist()}")
assert nlls[3] < nlls[1], "3-component mixture did not beat a single Gaussian on skewed toy data"
print("Smoke test passed.")
