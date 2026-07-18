"""RUN THIS FIRST (needs torch).  Verifies:
 1. torch classical pipeline (params=0) matches the verified NumPy mirror,
 2. gradients flow from Chamfer loss to network parameters,
 3. one optimization step reduces the loss on a single object.

Usage:  python scripts/smoke_test.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

from nssr.synthetic import make_sample
from nssr.train import to_torch, forward_object
from nssr.geometry import hermite_surface, zero_params, surface_points
from nssr.networks import ParamNet, contour_features
from nssr.losses import chamfer
from tests.numpy_sanity_check import hermite_surface_np

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    s = make_sample(seed=7, N=9)
    obj = to_torch(s, m=128, device=dev, dtype=torch.float64)

    # --- 1. torch classical == numpy classical --------------------------
    p0 = zero_params(obj["R"].shape[0], obj["R"].shape[1],
                     device=dev, dtype=torch.float64)
    S_t = hermite_surface(obj["R"], obj["Z"], obj["RB"], obj["RC"],
                          obj["Bh"], obj["Th"], p0, n_u=16)
    np_args = [obj[k].cpu().numpy() for k in ("R", "Z", "RB", "RC")]
    S_n = hermite_surface_np(*np_args, float(obj["Bh"]), float(obj["Th"]),
                             n_u=16)
    # NOTE: the numpy mirror uses the analytic radial cap direction while
    # torch uses Eq. 22-24 boundary directions -> compare interior patches.
    diff = (S_t[1:-1].cpu().numpy() - S_n[1:-1])
    err = np.abs(diff).max()
    print(f"1) interior torch-vs-numpy max abs diff: {err:.2e}",
          "OK" if err < 1e-10 else "MISMATCH -- investigate")

    # --- 2. gradient flow ------------------------------------------------
    net = ParamNet().to(device=dev, dtype=torch.float64)
    S, pts, nrms, params = forward_object(net, obj, n_u=16)
    loss = chamfer(pts, obj["gt_pts"])
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for p in net.parameters()
                if p.grad is not None)
    print(f"2) loss {loss.item():.6f}, grad-norm sum {gnorm:.3e}",
          "OK" if gnorm > 0 else "NO GRADIENT -- investigate")

    # --- 3. one step reduces loss on this object -------------------------
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    l0 = None
    for it in range(30):
        opt.zero_grad()
        _, pts, _, _ = forward_object(net, obj, n_u=16)
        loss = chamfer(pts, obj["gt_pts"])
        if l0 is None:
            l0 = loss.item()
        loss.backward(); opt.step()
    print(f"3) chamfer {l0:.6f} -> {loss.item():.6f} after 30 steps",
          "OK" if loss.item() < l0 else "NOT DECREASING -- investigate")
    print("\nSMOKE TEST COMPLETE")

if __name__ == "__main__":
    main()
