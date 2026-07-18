# curves_pytorch.py
# -----------------
# Fully vectorized PyTorch implementation of Goodman’s 2D curve smoothing.
# This version avoids Python loops entirely and supports both CPU and GPU
# execution via PyTorch tensors.
#
# ⚠️ This module uses PyTorch ONLY (no NumPy loops, no Numba).
# It is designed for high-throughput and GPU-accelerated workflows.
#
# Loop-based (pure Python) and Numba-accelerated versions are provided
# in separate modules.

import torch


def _cross2d_torch(a, b):
    """2D cross product scalar for arrays of 2-vectors: a,b shape (...,2) -> (...,)"""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def curve_goodman_torch(I, seg_pts, r_coef=0.25, s_coef=0.25, m=0.5, n=0.5, device=None):
    """
    Smooth curve through given data points using Goodman’s method (Torch version).

    Parameters
    ----------
    I : (n,2) torch.Tensor
        Input polyline points (closed loop expected).
    seg_pts : int
        Number of interpolation points per segment.

    Returns
    -------
    r : (n*seg_pts,2) torch.Tensor
        Smoothed curve points.
    """
    if device is None:
        device = I.device
    else:
        I = I.to(device)

    dtype = I.dtype
    no_pts = I.shape[0]

    t_vals = torch.linspace(0, 1, seg_pts + 1, dtype=dtype, device=device)[:-1]  # exclude endpoint
    idx = torch.arange(no_pts, dtype=torch.int, device=device)


    # --- Circular indexing (wrap-around)
    im1 = (idx - 1) % no_pts
    ip1 = (idx + 1) % no_pts
    im2 = (idx - 2) % no_pts
    ip2 = (idx + 2) % no_pts


    # --- Initialize tensors
    lineleft = torch.zeros(no_pts, dtype=torch.bool, device=device)
    lineright = torch.zeros_like(lineleft)
    kleft = torch.zeros(no_pts, dtype=dtype, device=device)
    kright = torch.zeros_like(kleft)
    # --- Collinearity checks
    def collinear(p, q, r):
        return torch.abs((q[:,1]-p[:,1])*(r[:,0]-p[:,0]) - (r[:,1]-p[:,1])*(q[:,0]-p[:,0])) < 1e-9

    lineleft |= collinear(I[im1], I[idx], I[ip1])
    lineright |= collinear(I[im1], I[idx], I[ip1])
    lineright |= collinear(I[ip1], I[idx], I[ip2])
    lineleft |= collinear(I[im1], I[idx], I[im2])

    # --- Curvatures (kleft, kright)
    num = 2 * _cross2d_torch(I[idx]-I[im1], I[ip1]-I[idx])
    den = torch.linalg.norm(I[idx]-I[im1], dim=-1) * torch.linalg.norm(I[ip1]-I[idx], dim=-1) * torch.linalg.norm(I[ip1]-I[im1], dim=-1)
    kleft = torch.where(~lineleft, num / torch.clamp(den, min=1e-12), torch.zeros_like(num))
    kright = torch.where(~lineright, num / torch.clamp(den, min=1e-12), torch.zeros_like(num))
    kleft = torch.where(den!=0, kleft, torch.zeros_like(num))
    kright = torch.where(den!=0, kright, torch.zeros_like(num))

    # --- Tangents
    T = torch.zeros_like(I)
    Tu = torch.zeros_like(I)
    a = torch.abs(kleft[ip1]) * torch.linalg.norm(I[ip1]-I[idx], dim=-1)**2
    b = torch.abs(kright[im1]) * torch.linalg.norm(I[idx]-I[im1], dim=-1)**2
    T = a.unsqueeze(-1)*(I[idx]-I[im1]) + b.unsqueeze(-1)*(I[ip1]-I[idx])
    normT = torch.linalg.norm(T, dim=-1, keepdim=True)
    Tu = torch.where(normT > 0, T / normT, torch.zeros_like(T))
    mask = (kleft != 0) | (kright != 0)
    T = T * mask.unsqueeze(-1)
    Tu = Tu * mask.unsqueeze(-1)

    # --- Bezier control points
    A, B, C, D = I.clone(), torch.zeros_like(I), torch.zeros_like(I), I[ip1].clone()
    lengthab = torch.zeros(no_pts, dtype=dtype, device=device)
    lengthcd = torch.zeros_like(lengthab)

    k_prod = kright * kleft[ip1]
    convex_mask = k_prod > 0
    inflect_mask = k_prod < 0

    # Shared distance
    L = torch.linalg.norm(I[ip1]-I[idx], dim=-1)

    # Convex case
    sina = _cross2d_torch(T, I[ip1]-I[idx]) / (torch.linalg.norm(T, dim=-1) * torch.linalg.norm(I[ip1]-I[idx], dim=-1))
    sinb = _cross2d_torch(I[ip1]-I[idx], T[ip1]) / (torch.linalg.norm(T[ip1], dim=-1) * torch.linalg.norm(I[ip1]-I[idx], dim=-1))
    sinab = _cross2d_torch(T, T[ip1]) / (torch.linalg.norm(T, dim=-1) * torch.linalg.norm(T[ip1], dim=-1))

    p = 2*torch.abs(sinb) / (2*m*torch.abs(sinb)+(1-m)*L*torch.abs(kleft[ip1])+2*torch.abs(sinab))
    q = 2*torch.abs(sina) / (2*n*torch.abs(sina)+(1-n)*L*torch.abs(kright)+2*torch.abs(sinab))
    lengthab = torch.where(convex_mask, p*L, lengthab)
    lengthcd = torch.where(convex_mask, q*L, lengthcd)

    B = torch.where(convex_mask.unsqueeze(-1), A + lengthab.unsqueeze(-1)*Tu, B)
    C = torch.where(convex_mask.unsqueeze(-1), D - lengthcd.unsqueeze(-1)*Tu[ip1], C)

    # Inflection case
    lengthab = torch.where(inflect_mask, r_coef*L, lengthab)
    lengthcd = torch.where(inflect_mask, s_coef*L, lengthcd)
    B = torch.where(inflect_mask.unsqueeze(-1), A + lengthab.unsqueeze(-1)*Tu, B)
    C = torch.where(inflect_mask.unsqueeze(-1), D - lengthcd.unsqueeze(-1)*Tu[ip1], C)

    # --- Weights
    alpha = torch.zeros(no_pts, dtype=dtype, device=device)
    beta = torch.zeros_like(alpha)
    denom1 = 2 * _cross2d_torch(B - A, C - B)
    denom2 = 2 * _cross2d_torch(C - B, D - C)

    alpha = torch.where(denom1 != 0, kright * lengthab**3 / denom1, torch.zeros_like(alpha))
    beta = torch.where(denom2 != 0, kleft[ip1] * lengthcd**3 / denom2, torch.zeros_like(beta))

    # --- Build curve points ---
    tj = t_vals.view(1, seg_pts, 1)                    # (1, seg_pts, 1)
    Ai, Bi, Ci, Di = A[:, None, :], B[:, None, :], C[:, None, :], D[:, None, :]  # (no_pts, 1, 2)
    alph = alpha[:, None, None]
    bet = beta[:, None, None]
    mask_nl = (kright != 0).view(-1, 1, 1)             # (no_pts, 1, 1)

    # --- nonlinear branch (vectorized) ---
    num = (Ai * alph * (1 - tj)**3 +
          Bi * tj * (1 - tj)**2 +
          Ci * tj**2 * (1 - tj) +
          Di * bet * tj**3)
    den = alph * (1 - tj)**3 + tj * (1 - tj)**2 + tj**2 * (1 - tj) + bet * tj**3
    r_nl = num / den                                   # (no_pts, seg_pts, 2)

    # --- linear branch (vectorized) ---
    r_lin = (1 - tj) * Ai + tj * Di                    # (no_pts, seg_pts, 2)

    # --- select between them without loop ---
    r_all = torch.where(mask_nl, r_nl, r_lin)          # (no_pts, seg_pts, 2)

    # --- flatten to match your original shape ---
    r = r_all.reshape(no_pts * seg_pts, 2)

    return r