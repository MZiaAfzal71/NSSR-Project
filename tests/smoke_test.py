"""Fast NSSR-V2 smoke test.

Runs:
- imports
- synthetic sample generation
- preprocessing
- classical surface reconstruction
- zero-init ParamNet forward
- V2 geometry evaluation
- one backward pass through the legacy-compatible loss

Usage:
    python scripts/smoke_test.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from nssr.synthetic import make_sample
from nssr.train import to_torch, forward_object
from nssr.networks import ParamNet
from nssr.geometry import evaluate_geometry, zero_params, surface_points
from nssr.losses import total_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print("device:", device)

    sample = make_sample(0, N=5)
    obj = to_torch(
        sample,
        m=64,
        device=device,
        dtype=dtype,
        gt_subsample=2000,
        seed=0,
    )

    N, m = obj["R"].shape[:2]
    p0 = zero_params(N, m, device=device, dtype=dtype)

    classical = evaluate_geometry(
        obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"],
        p0,
        n_u=8,
        closed_top=obj.get("closed_top", True),
        base_circular=obj.get("base_circular", True),
        crown_circular=obj.get("crown_circular", True),
        compute_jacobian=True,
        compute_curvature=True,
        run_validation=False,
    )

    assert torch.isfinite(classical.surface.xyz).all()
    assert torch.isfinite(classical.surface.normals).all()

    net = ParamNet().to(device=device, dtype=dtype)
    S, pts, nrms, params, _ = forward_object(
        net,
        obj,
        n_u=8,
    )

    assert pts.ndim == 2 and pts.shape[-1] == 3
    assert nrms.shape == pts.shape

    loss, parts = total_loss(
        pts,
        nrms,
        obj["gt_pts"],
        obj["gt_normals"],
        params,
        surf_sub=2000,
        gt_sub=2000,
    )

    loss.backward()

    finite_grad = True
    for p in net.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            finite_grad = False
            break

    assert finite_grad

    print("surface shape:", tuple(S.shape))
    print("points:", pts.shape[0])
    print("loss:", float(loss.detach()))
    print("parts:", parts)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
