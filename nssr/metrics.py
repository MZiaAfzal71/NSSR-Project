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
