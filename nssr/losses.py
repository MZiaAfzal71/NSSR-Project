"""NSSR-V2 losses.

Active geometry safeguards
--------------------------
1. Signed-Jacobian barrier:
   discourages local orientation reversal / foldover.
2. Cap radial-fold barrier:
   discourages the base/crown cap from turning back toward its pole and
   producing a visible loop while remaining locally Jacobian-valid.

Curvature is intentionally NOT part of the training objective.  The sampled
finite-difference curvature estimator remains available as a diagnostic under
``nssr.core.curvature`` but is not used here.
"""

from __future__ import annotations

from typing import Optional

import torch

from .core.jacobian import jacobian_barrier_loss
from .core.types import GeometryOutput


def _nn_sqdist(a, b, chunk_q=4096, chunk_t=16384):
    """Squared nearest-neighbour distance from each point in ``a`` to ``b``."""
    Na = a.shape[0]
    out_d = torch.empty(Na, device=a.device, dtype=a.dtype)
    out_i = torch.empty(Na, device=a.device, dtype=torch.long)

    for i in range(0, Na, chunk_q):
        aq = a[i:i + chunk_q]
        best_d = torch.full(
            (aq.shape[0],), float("inf"),
            device=a.device, dtype=a.dtype,
        )
        best_i = torch.zeros(
            aq.shape[0], device=a.device, dtype=torch.long,
        )

        for j in range(0, b.shape[0], chunk_t):
            bt = b[j:j + chunk_t]
            d = torch.cdist(aq, bt)
            dmin, imin = d.min(dim=1)
            upd = dmin < best_d
            best_d = torch.where(upd, dmin, best_d)
            best_i = torch.where(upd, imin + j, best_i)

        out_d[i:i + chunk_q] = best_d.square()
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


# ---------------------------------------------------------------------------
# Geometry-safety losses
# ---------------------------------------------------------------------------

def intentional_pole_mask(
    surface: torch.Tensor,
    *,
    closed_top: bool = True,
) -> torch.Tensor:
    """Return samples that are singular by cap construction.

    ``surface`` is ``(P, n_u, m, 3)``.

    The exact base pole is ``surface[0, 0]``.
    For a closed crown, the exact crown pole is ``surface[-1, -1]``.

    These locations are intentionally collapsed in the parameterization and
    must not be treated as accidental Jacobian degeneracies.
    """
    P, n_u, m, _ = surface.shape
    mask = torch.zeros(P, n_u, m, dtype=torch.bool, device=surface.device)
    mask[0, 0, :] = True
    if closed_top:
        mask[-1, -1, :] = True
    return mask


def cap_radial_fold_measure(
    surface: torch.Tensor,
    RB: torch.Tensor,
    RC: torch.Tensor,
    *,
    closed_top: bool = True,
    eps: float = 1e-8,
):
    """Measure normalized radial turn-back in the cap patches.

    A valid base cap should move radially OUT from RB as u increases:
        r(u+du) - r(u) >= 0.

    A valid crown cap should move radially IN toward RC as u increases:
        r(u+du) - r(u) <= 0.

    Returns
    -------
    base_violation, crown_violation
        Non-negative tensors shaped roughly ``(n_u-1, m)`` and normalized by
        the corresponding input-cap radius.  Zero means monotone radial travel.

    Notes
    -----
    This is deliberately a *turn-back* measure, not a generic smoothness or
    curvature measure.  It therefore targets the visible cap loop directly.
    """
    if surface.ndim != 4 or surface.shape[-1] != 3:
        raise ValueError("surface must have shape (P, n_u, m, 3)")

    # Base: pole -> first contour.
    base_xy = surface[0, :, :, :2]
    base_r = torch.linalg.vector_norm(base_xy - RB.reshape(1, 1, 2), dim=-1)
    base_scale = base_r[-1].mean().clamp_min(eps)
    base_dr = (base_r[1:] - base_r[:-1]) / base_scale
    base_violation = torch.relu(-base_dr)

    if closed_top:
        # Crown: last contour -> pole.
        crown_xy = surface[-1, :, :, :2]
        crown_r = torch.linalg.vector_norm(
            crown_xy - RC.reshape(1, 1, 2), dim=-1
        )
        crown_scale = crown_r[0].mean().clamp_min(eps)
        crown_dr = (crown_r[1:] - crown_r[:-1]) / crown_scale
        crown_violation = torch.relu(crown_dr)
    else:
        crown_violation = torch.zeros(
            0,
            surface.shape[2],
            device=surface.device,
            dtype=surface.dtype,
        )

    return base_violation, crown_violation


def cap_radial_fold_loss(
    surface: torch.Tensor,
    RB: torch.Tensor,
    RC: torch.Tensor,
    *,
    closed_top: bool = True,
    power: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Differentiable loss for cap radial turn-back / loops.

    The measure is dimensionless because each cap is normalized by its boundary
    radius.  Classical monotone caps therefore have loss approximately zero,
    while a radial reversal produces a positive penalty.

    ``power=2`` is recommended.  The default does NOT force a minimum radial
    slope; it only penalizes reversal, so flat-but-valid regions are allowed.
    """
    if power <= 0:
        raise ValueError("power must be > 0")

    b, c = cap_radial_fold_measure(
        surface,
        RB,
        RC,
        closed_top=closed_top,
    )

    pieces = [b.reshape(-1)]
    if c.numel():
        pieces.append(c.reshape(-1))

    v = torch.cat(pieces)
    p = v.pow(power)

    if reduction == "mean":
        return p.mean() if p.numel() else surface.new_zeros(())
    if reduction == "sum":
        return p.sum()
    if reduction == "max":
        return p.max() if p.numel() else surface.new_zeros(())
    if reduction == "none":
        return p
    raise ValueError("reduction must be mean, sum, max, or none")


def cap_radial_fold_max(
    surface: torch.Tensor,
    RB: torch.Tensor,
    RC: torch.Tensor,
    *,
    closed_top: bool = True,
) -> torch.Tensor:
    """Maximum normalized radial reversal, useful for inference diagnostics."""
    b, c = cap_radial_fold_measure(
        surface, RB, RC, closed_top=closed_top
    )
    vals = [b.reshape(-1)]
    if c.numel():
        vals.append(c.reshape(-1))
    v = torch.cat(vals)
    return v.max() if v.numel() else surface.new_zeros(())


def geometry_regularization_loss(
    geometry: GeometryOutput,
    *,
    lam_jacobian: float = 0.0,
    jacobian_margin: float = 1.0e-4,
    closed_top: bool = True,
):
    """Signed-Jacobian penalty with intentional poles excluded.

    Important difference from the earlier implementation:
    ``geometry.jacobian.valid_mask`` is NOT used as the barrier mask because it
    excludes degenerate samples.  A degenerate sample has signed Jacobian near
    zero and should be penalized.  Only known pole singularities (and non-finite
    values that cannot safely enter arithmetic) are excluded.
    """
    device = geometry.surface.xyz.device
    dtype = geometry.surface.xyz.dtype
    zero = torch.zeros((), device=device, dtype=dtype)

    if lam_jacobian == 0.0:
        return zero, {"jacobian": zero}

    if geometry.jacobian is None:
        raise ValueError(
            "lam_jacobian is non-zero but geometry.jacobian is missing"
        )

    signed = geometry.jacobian.signed
    evaluable = ~intentional_pole_mask(
        geometry.surface.xyz,
        closed_top=closed_top,
    )
    evaluable = evaluable & torch.isfinite(signed)

    l_jac = jacobian_barrier_loss(
        signed,
        margin=jacobian_margin,
        valid_mask=evaluable,
    )

    return lam_jacobian * l_jac, {"jacobian": l_jac}


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
    surface: Optional[torch.Tensor] = None,
    RB: Optional[torch.Tensor] = None,
    RC: Optional[torch.Tensor] = None,
    closed_top: bool = True,
    geometry: Optional[GeometryOutput] = None,
    lam_jacobian: float = 0.0,
    jacobian_margin: float = 1.0e-4,
    lam_cap_fold: float = 0.0,
    cap_fold_power: float = 2.0,
):
    """NSSR objective with active local- and cap-safety terms.

    ``lam_cap_fold`` is the new targeted safeguard against visible cap loops.
    It acts directly on the sampled cap geometry and therefore complements the
    local signed-Jacobian term.
    """
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
        l_n = pred_pts.new_zeros(())

    l_r = param_l2(params)
    l_s = param_smoothness(params)

    l_geo = pred_pts.new_zeros(())
    l_jac = pred_pts.new_zeros(())
    if lam_jacobian != 0.0:
        if geometry is None:
            raise ValueError(
                "geometry must be provided when lam_jacobian is non-zero"
            )
        l_geo, geo_parts = geometry_regularization_loss(
            geometry,
            lam_jacobian=lam_jacobian,
            jacobian_margin=jacobian_margin,
            closed_top=closed_top,
        )
        l_jac = geo_parts["jacobian"]

    l_cap = pred_pts.new_zeros(())
    if lam_cap_fold != 0.0:
        if surface is None or RB is None or RC is None:
            raise ValueError(
                "surface, RB, and RC are required when lam_cap_fold is non-zero"
            )
        l_cap = cap_radial_fold_loss(
            surface,
            RB,
            RC,
            closed_top=closed_top,
            power=cap_fold_power,
        )

    loss = (
        l_cd
        + lam_n * l_n
        + lam_r * l_r
        + lam_s * l_s
        + l_geo
        + lam_cap_fold * l_cap
    )

    return loss, {
        "chamfer": float(l_cd.detach()),
        "normal": float(l_n.detach()),
        "reg": float(l_r.detach()),
        "smooth": float(l_s.detach()),
        "jacobian": float(l_jac.detach()),
        "cap_fold": float(l_cap.detach()),
        "geometry": float(l_geo.detach()),
        "total": float(loss.detach()),
    }


__all__ = [
    "_nn_sqdist",
    "chamfer",
    "normal_loss",
    "chamfer_weighted",
    "intentional_pole_mask",
    "cap_radial_fold_measure",
    "cap_radial_fold_loss",
    "cap_radial_fold_max",
    "geometry_regularization_loss",
    "total_loss",
]
