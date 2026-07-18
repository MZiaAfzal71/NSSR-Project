"""Parameter networks Phi_theta (METHOD.md sec. 4).

Input : invariant per-point features  (N, m, F)
Output: the learnable fields of geometry.py, bounded by c*tanh so that
        weights stay within e^{+-c} of classical.

The final linear layer is zero-initialized => at initialization the model
IS the classical pipeline.
"""
from __future__ import annotations
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
def contour_features(R: torch.Tensor, Z: torch.Tensor,
                     RB: torch.Tensor, RC: torch.Tensor,
                     Bh: torch.Tensor, Th: torch.Tensor) -> torch.Tensor:
    """(N, m, 9) rotation/translation-invariant features (METHOD.md sec. 4)."""
    N, m, _ = R.shape
    eps = 1e-9
    # chords to previous / next slice, with virtual base & crown rows
    prev = torch.cat([RB.expand(1, m, 2), R[:-1]], dim=0)
    nxt = torch.cat([R[1:], RC.expand(1, m, 2)], dim=0)
    d_prev = (R - prev).norm(dim=-1)
    d_next = (nxt - R).norm(dim=-1)
    Zfull_prev = torch.cat([Bh.view(1), Z[:-1]])
    Zfull_next = torch.cat([Z[1:], Th.view(1)])
    dz_prev = (Z - Zfull_prev).unsqueeze(-1).expand(N, m)
    dz_next = (Zfull_next - Z).unsqueeze(-1).expand(N, m)
    # in-plane discrete curvature (Eq. 1) along each contour
    Rm = torch.roll(R, 1, dims=1)
    Rp = torch.roll(R, -1, dims=1)
    u, v = R - Rm, Rp - R
    cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    w = Rp - Rm
    kappa = 2 * cross / (u.norm(dim=-1) * v.norm(dim=-1) * w.norm(dim=-1) + eps)
    # radial distance to per-contour centroid, normalized by mean radius
    c = R.mean(dim=1, keepdim=True)
    rad = (R - c).norm(dim=-1)
    rad = rad / (rad.mean(dim=1, keepdim=True) + eps)
    # height position encoding
    t = torch.linspace(0, 1, N, device=R.device, dtype=R.dtype)
    t = t.unsqueeze(-1).expand(N, m)
    feats = torch.stack([
        torch.log(d_prev + eps), torch.log(d_next + eps),
        dz_prev, dz_next, kappa, rad,
        t, torch.sin(torch.pi * t), torch.cos(torch.pi * t)], dim=-1)
    return feats                                            # (N, m, 9)


# ----------------------------------------------------------------------
class CircConv(nn.Module):
    """1-D conv along j with circular padding (closed contours)."""

    def __init__(self, cin, cout, k=5):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, padding=k // 2,
                              padding_mode="circular")

    def forward(self, x):        # x: (B, C, m)
        return self.conv(x)


class ParamNet(nn.Module):
    """Predicts (s_a, s_b, s_tau) per (i, j) and (s_fB, s_fC) per j.

    Every contour row i is processed by the same circular CNN (weight
    sharing across rows and objects); the height encoding in the features
    lets rows behave differently.
    """

    def __init__(self, fdim=9, hidden=64, c_bound=2.0, free_residual=False):
        super().__init__()
        self.c = c_bound
        self.free_residual = free_residual
        self.body = nn.Sequential(
            CircConv(fdim, hidden), nn.SiLU(),
            CircConv(hidden, hidden), nn.SiLU(),
            CircConv(hidden, hidden), nn.SiLU(),
        )
        nout = 3 + (2 if free_residual else 0)
        self.head = nn.Conv1d(hidden, nout, 1)
        self.bhead = nn.Conv1d(hidden, 2, 1)          # s_fB, s_fC
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.bhead.weight); nn.init.zeros_(self.bhead.bias)

    def forward(self, feats):                          # feats: (N, m, F)
        N, m, F = feats.shape
        h = self.body(feats.permute(0, 2, 1))          # (N, hidden, m)
        out = self.head(h)                             # (N, nout, m)
        s = self.c * torch.tanh(out)
        params = {"s_a": s[:, 0], "s_b": s[:, 1], "s_tau": s[:, 2]}
        if self.free_residual:
            params["rho"] = 0.1 * s[:, 3:5].permute(0, 2, 1)   # (N, m, 2)
        b = self.c * torch.tanh(self.bhead(h))          # (N, 2, m)
        params["s_fB"] = b[0, 0]                        # base row
        params["s_fC"] = b[-1, 1]                       # crown row
        return params


def param_l2(params) -> torch.Tensor:
    """Proximity regularizer L_reg (mean of squared fields)."""
    total, cnt = 0.0, 0
    for k, v in params.items():
        total = total + (v ** 2).sum()
        cnt += v.numel()
    return total / max(cnt, 1)


def param_smoothness(params) -> torch.Tensor:
    """Circumferential smoothness L_sm on the (N, m) fields."""
    tot, cnt = 0.0, 0
    for k in ("s_a", "s_b", "s_tau"):
        v = params[k]
        d = torch.roll(v, -1, dims=-1) - v
        tot = tot + (d ** 2).sum(); cnt += d.numel()
    return tot / max(cnt, 1)
