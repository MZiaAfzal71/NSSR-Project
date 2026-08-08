"""NSSR-V2 training losses and nearest-neighbour utilities.

Preserves the existing NSSR loss API while adding optional Jacobian and
curvature regularizers. The V2 terms default to zero weight, so legacy
training behavior is retained until explicitly enabled.
"""

from __future__ import annotations

from typing import Optional

import torch

from .core.curvature import curvature_barrier_loss
from .core.jacobian import jacobian_barrier_loss
from .core.types import GeometryOutput


def _nn_sqdist(a, b, chunk_q=4096, chunk_t=16384):
    """Squared nearest-neighbour distance from each point in a to b."""
    Na = a.shape[0]
    out_d = torch.empty(Na, device=a.device, dtype=a.dtype)
    out_i = torch.empty(Na, device=a.device, dtype=torch.long)

    for i in range(0, Na, chunk_q):
        aq = a[i:i + chunk_q]
        best_d = torch.full(
            (aq.shape[0],), float("inf"),
            device=a.device, dtype=a.dtype
        )
        best_i = torch.zeros(
            aq.shape[0], device=a.device, dtype=torch.long
        )

        for j in range(0, b.shape[0], chunk_t):
            bt = b[j:j + chunk_t]
            d = torch.cdist(aq, bt)
            dmin, imin = d.min(dim=1)
            upd = dmin < best_d
            best_d = torch.where(upd, dmin, best_d)
            best_i = torch.where(upd, imin + j, best_i)

        out_d[i:i + chunk_q] = best_d ** 2
        out_i[i:i + chunk_q] = best_i

    return out_d, out_i


def chamfer(pred, gt, surf_sub=None, gt_sub=None):
    """Two-sided mean squared Chamfer distance."""
    if surf_sub is not None and pred.shape[0] > surf_sub:
        idx = torch.randperm(pred.shape[0], device=pred.device)[:surf_sub]
        pred = pred[idx]

    if gt_sub is not None and gt.shape[0] > gt_sub:
        idx = torch.randperm(gt.shape[0], device=gt.device)[:gt_sub]
        gt = gt[idx]

    d_pg, _ = _nn_sqdist(pred, gt)
    d_gp, _ = _nn_sqdist(gt, pred)
    return d_pg.mean() + d_gp.mean()


def normal_loss(pred_pts, pred_normals, gt_pts, gt_normals, gt_sub=None):
    """Legacy unoriented nearest-neighbour normal loss."""
    if gt_sub is not None and gt_pts.shape[0] > gt_sub:
        idx = torch.randperm(gt_pts.shape[0], device=gt_pts.device)[:gt_sub]
        gt_pts = gt_pts[idx]
        gt_normals = gt_normals[idx]

    _, idx = _nn_sqdist(gt_pts, pred_pts)
    cos = (gt_normals * pred_normals[idx]).sum(-1).abs()
    return (1.0 - cos).mean()


def chamfer_weighted(pred, gt, pred_w=None, surf_sub=None, gt_sub=None):
    """Two-sided Chamfer with optional per-predicted-point weights."""
    if surf_sub is not None and pred.shape[0] > surf_sub:
        idx = torch.randperm(pred.shape[0], device=pred.device)[:surf_sub]
        pred = pred[idx]
        if pred_w is not None:
            pred_w = pred_w[idx]

    if gt_sub is not None and gt.shape[0] > gt_sub:
        idx = torch.randperm(gt.shape[0], device=gt.device)[:gt_sub]
        gt = gt[idx]

    d_pg, _ = _nn_sqdist(pred, gt)
    d_gp, idx_gp = _nn_sqdist(gt, pred)

    if pred_w is None:
        return d_pg.mean() + d_gp.mean()

    w_pg = pred_w
    w_gp = pred_w[idx_gp]

    return (
        (d_pg * w_pg).sum() / w_pg.sum().clamp_min(1e-8)
        + (d_gp * w_gp).sum() / w_gp.sum().clamp_min(1e-8)
    )


def geometry_regularization_loss(
    geometry: GeometryOutput,
    *,
    lam_jacobian: float = 0.0,
    lam_curvature: float = 0.0,
    jacobian_margin: float = 1.0e-4,
    max_abs_curvature: float = 100.0,
    curvature_power: float = 2.0,
):
    """Compute optional Jacobian and curvature penalties."""
    device = geometry.surface.xyz.device
    dtype = geometry.surface.xyz.dtype
    zero = torch.zeros((), device=device, dtype=dtype)

    l_jac = zero
    l_curv = zero

    if lam_jacobian != 0.0:
        if geometry.jacobian is None:
            raise ValueError(
                "lam_jacobian is non-zero but geometry.jacobian is missing"
            )

        l_jac = jacobian_barrier_loss(
            geometry.jacobian.signed,
            margin=jacobian_margin,
            valid_mask=geometry.jacobian.valid_mask,
        )

    if lam_curvature != 0.0:
        if geometry.curvature is None:
            raise ValueError(
                "lam_curvature is non-zero but geometry.curvature is missing"
            )

        valid_mask = (
            geometry.jacobian.valid_mask
            if geometry.jacobian is not None
            else None
        )

        l_curv = curvature_barrier_loss(
            geometry.curvature,
            max_abs_curvature=max_abs_curvature,
            valid_mask=valid_mask,
            power=curvature_power,
        )

    total = lam_jacobian * l_jac + lam_curvature * l_curv

    return total, {
        "jacobian": l_jac,
        "curvature": l_curv,
    }


def total_loss(
    pred_pts,
    pred_normals,
    gt_pts,
    gt_normals,
    params,
    lam_n=0.1,
    lam_r=1e-3,
    lam_s=1e-3,
    surf_sub=20000,
    gt_sub=20000,
    cap_mask=None,
    cap_weight=1.0,
    *,
    geometry: Optional[GeometryOutput] = None,
    lam_jacobian: float = 0.0,
    lam_curvature: float = 0.0,
    jacobian_margin: float = 1.0e-4,
    max_abs_curvature: float = 100.0,
    curvature_power: float = 2.0,
):
    """Legacy NSSR objective plus optional V2 geometry terms."""
    from .networks import param_l2, param_smoothness

    w = None
    if cap_mask is not None and cap_weight != 1.0:
        w = torch.ones(
            pred_pts.shape[0],
            device=pred_pts.device,
            dtype=pred_pts.dtype,
        )
        w = torch.where(
            cap_mask,
            torch.full_like(w, cap_weight),
            w,
        )

    l_cd = chamfer_weighted(
        pred_pts,
        gt_pts,
        pred_w=w,
        surf_sub=surf_sub,
        gt_sub=gt_sub,
    )

    if gt_normals is not None:
        l_n = normal_loss(
            pred_pts,
            pred_normals,
            gt_pts,
            gt_normals,
            gt_sub=gt_sub,
        )
    else:
        l_n = torch.zeros(
            (),
            device=pred_pts.device,
            dtype=pred_pts.dtype,
        )

    l_r = param_l2(params)
    l_s = param_smoothness(params)

    l_geo = torch.zeros(
        (),
        device=pred_pts.device,
        dtype=pred_pts.dtype,
    )
    l_jac = torch.zeros_like(l_geo)
    l_curv = torch.zeros_like(l_geo)

    if lam_jacobian != 0.0 or lam_curvature != 0.0:
        if geometry is None:
            raise ValueError(
                "geometry must be provided when V2 geometry losses are enabled"
            )

        l_geo, geo_parts = geometry_regularization_loss(
            geometry,
            lam_jacobian=lam_jacobian,
            lam_curvature=lam_curvature,
            jacobian_margin=jacobian_margin,
            max_abs_curvature=max_abs_curvature,
            curvature_power=curvature_power,
        )
        l_jac = geo_parts["jacobian"]
        l_curv = geo_parts["curvature"]

    loss = (
        l_cd
        + lam_n * l_n
        + lam_r * l_r
        + lam_s * l_s
        + l_geo
    )

    return loss, {
        "chamfer": float(l_cd.detach()),
        "normal": float(l_n.detach()),
        "reg": float(l_r.detach()),
        "smooth": float(l_s.detach()),
        "jacobian": float(l_jac.detach()),
        "curvature": float(l_curv.detach()),
        "geometry": float(l_geo.detach()),
        "total": float(loss.detach()),
    }


__all__ = [
    "_nn_sqdist",
    "chamfer",
    "normal_loss",
    "chamfer_weighted",
    "geometry_regularization_loss",
    "total_loss",
]
