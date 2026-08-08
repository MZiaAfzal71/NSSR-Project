"""Evaluation metrics (no gradients needed)."""
from __future__ import annotations
import torch
from .losses import _nn_sqdist


@torch.no_grad()
def evaluate_surface(pred_pts, gt_pts, pred_normals=None, gt_normals=None,
                     max_pts=200000):
    # Cap point counts so full-resolution eval (large m) stays within memory.
    if pred_pts.shape[0] > max_pts:
        idx = torch.randperm(pred_pts.shape[0], device=pred_pts.device)[:max_pts]
        pred_pts = pred_pts[idx]
        pred_normals = pred_normals[idx] if pred_normals is not None else None
    if gt_pts.shape[0] > max_pts:
        idx = torch.randperm(gt_pts.shape[0], device=gt_pts.device)[:max_pts]
        gt_pts = gt_pts[idx]
        gt_normals = gt_normals[idx] if gt_normals is not None else None
    d_pg, idx_pg = _nn_sqdist(pred_pts, gt_pts)
    d_gp, idx_gp = _nn_sqdist(gt_pts, pred_pts)
    out = {
        "chamfer_l2": (d_pg.mean() + d_gp.mean()).item(),
        "chamfer_l1": (d_pg.sqrt().mean() + d_gp.sqrt().mean()).item() / 2,
        "hausdorff": max(d_pg.max().item(), d_gp.max().item()) ** 0.5,
        "hausdorff95": max(torch.quantile(d_pg.sqrt(), 0.95).item(),
                            torch.quantile(d_gp.sqrt(), 0.95).item()),
    }
    if pred_normals is not None and gt_normals is not None:
        cos = (gt_normals * pred_normals[idx_gp]).sum(-1).abs()
        out["normal_consistency"] = cos.mean().item()
    return out


@torch.no_grad()
def c1_diagnostic(gR, gZ):
    """Minimum tangent magnitude over every contour junction — the figure
    from your CiSE paper (Fig. 4 top), now for the LEARNED tangent field.
    Values well clear of zero => no near-cusp junctions => C1 claim holds."""
    mag = torch.sqrt(gR.norm(dim=-1) ** 2 + gZ ** 2)     # (N, m)
    return {"min_per_contour": mag.min(dim=1).values.cpu().numpy(),
            "global_min": mag.min().item()}


@torch.no_grad()
def axis_clearance(S, R):
    """Smallest distance from the interior surface to the object's LOCAL
    axis, relative to the narrowest input contour.

    Why this matters: the learned tangent multiplier is bounded by
    e^{+-c_bound}. At c_bound=2 (7.4x) an amplified tangent can push the
    interpolated surface THROUGH the central axis on narrow features --
    measured on the vase, whose 0.191-radius neck collapses to 0.002 --
    producing a pinch-to-a-point and an apparent "cone" in the wireframe.
    The Chamfer number barely notices (it is a small region), so this
    failure is invisible in the aggregate metrics and must be checked
    separately.

    The axis is taken PER PATCH, interpolated between the two contour
    centroids that patch spans. A single global centre would be wrong for
    bent objects: the banana false-flags at ratio 0.01 under a global
    centre even in the exact classical configuration.

    Returns (min_clearance, ratio_to_narrowest_contour); ratio < 0.1 means
    the surface is collapsing onto the axis.
    """
    cen = R.mean(dim=1)                                  # (N, 2) centroids
    n_int = S.shape[0] - 2 if S.shape[0] > 2 else S.shape[0]
    if n_int <= 0:
        return float("inf"), float("inf")
    n_u = S.shape[1]
    u = torch.linspace(0, 1, n_u, device=S.device, dtype=S.dtype)
    worst = None
    for k in range(n_int):
        patch = S[1 + k]                                 # (n_u, m, 3)
        c0 = cen[min(k, cen.shape[0] - 1)]
        c1 = cen[min(k + 1, cen.shape[0] - 1)]
        axis = (1 - u).unsqueeze(-1) * c0 + u.unsqueeze(-1) * c1   # (n_u,2)
        r = torch.linalg.norm(patch[..., :2] - axis.unsqueeze(1), dim=-1)
        mn = r.min()
        worst = mn if worst is None else torch.minimum(worst, mn)
    contour_r = torch.linalg.norm(R - cen.unsqueeze(1), dim=-1).mean(dim=1).min()
    return worst.item(), (worst / contour_r.clamp_min(1e-9)).item()
