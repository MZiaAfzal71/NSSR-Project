"""NSSR-V2 losses.

Active geometry safeguards
--------------------------
1. Signed-Jacobian barrier:
   discourages local orientation reversal / foldover.
2. Cap radial/meridional turn-back barrier:
   discourages the base/crown cap from reversing progress and producing a
   visible loop while remaining locally Jacobian-valid.

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


def _cap_progress_violation(
    cap: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Backward motion along the cap endpoint chord.

    ``cap`` has shape ``(n_u, m, 3)`` and is parameterized from endpoint 0 to
    endpoint 1.  For each circumferential column we project every sampled point
    onto the straight chord joining its two cap endpoints:

        t(u) = <S(u)-P0, P1-P0> / ||P1-P0||^2.

    A non-self-turning cap must not move *backwards* along that chord as u
    increases.  Lateral Hermite bulge is still allowed; only negative progress
    increments are penalized.

    This catches the axial/meridional turn-back visible in the apple cap even
    when radial distance alone remains monotone.
    """
    p0 = cap[0:1]
    p1 = cap[-1:]
    chord = p1 - p0                                      # (1,m,3)
    denom = chord.square().sum(dim=-1).clamp_min(eps)   # (1,m)
    t = ((cap - p0) * chord).sum(dim=-1) / denom        # (n_u,m)
    dt = t[1:] - t[:-1]
    return torch.relu(-dt)


def cap_radial_fold_measure(
    surface: torch.Tensor,
    RB: torch.Tensor,
    RC: torch.Tensor,
    *,
    closed_top: bool = True,
    eps: float = 1e-8,
):
    """Combined cap turn-back measure.

    Historical name retained for API compatibility.  The measure now catches
    TWO independent loop mechanisms:

    1. radial reversal about the cap pole in the xy plane;
    2. backward meridional/chord progress in full 3-D.

    The returned violation is the pointwise maximum of those two normalized,
    dimensionless tests.  Therefore existing validation/reconstruction code
    using ``cap_radial_fold_*`` automatically receives the stronger test.

    Base cap
    --------
    Pole -> first contour:
        radial distance should not decrease;
        chord progress should not decrease.

    Crown cap
    ---------
    Last contour -> pole:
        radial distance should not increase;
        chord progress should not decrease.
    """
    if surface.ndim != 4 or surface.shape[-1] != 3:
        raise ValueError("surface must have shape (P, n_u, m, 3)")

    # ----- base ------------------------------------------------------------
    base = surface[0]
    base_xy = base[:, :, :2]
    base_r = torch.linalg.vector_norm(
        base_xy - RB.reshape(1, 1, 2), dim=-1
    )
    base_scale = base_r[-1].mean().clamp_min(eps)
    base_dr = (base_r[1:] - base_r[:-1]) / base_scale
    base_radial = torch.relu(-base_dr)
    base_progress = _cap_progress_violation(base, eps=eps)
    base_violation = torch.maximum(base_radial, base_progress)

    # ----- crown -----------------------------------------------------------
    if closed_top:
        crown = surface[-1]
        crown_xy = crown[:, :, :2]
        crown_r = torch.linalg.vector_norm(
            crown_xy - RC.reshape(1, 1, 2), dim=-1
        )
        crown_scale = crown_r[0].mean().clamp_min(eps)
        crown_dr = (crown_r[1:] - crown_r[:-1]) / crown_scale
        crown_radial = torch.relu(crown_dr)
        crown_progress = _cap_progress_violation(crown, eps=eps)
        crown_violation = torch.maximum(crown_radial, crown_progress)
    else:
        crown_violation = torch.zeros(
            0,
            surface.shape[2],
            device=surface.device,
            dtype=surface.dtype,
        )

    return base_violation, crown_violation


def _topk_mean(
    values: torch.Tensor,
    *,
    fraction: float = 0.05,
) -> torch.Tensor:
    """Mean of the largest ``fraction`` of a non-negative tensor.

    Local geometric failures should not be diluted by thousands of safe
    samples. ``fraction=1`` reproduces a global mean; the recommended training
    value is 0.05 (worst 5%).
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError("top-k fraction must lie in (0, 1]")

    flat = values.reshape(-1)
    if flat.numel() == 0:
        return values.new_zeros(())

    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return values.new_tensor(float("inf"))

    k = max(1, int(round(fraction * finite.numel())))
    k = min(k, finite.numel())
    return torch.topk(
        finite, k=k, largest=True, sorted=False
    ).values.mean()


def cap_radial_fold_loss(
    surface: torch.Tensor,
    RB: torch.Tensor,
    RC: torch.Tensor,
    *,
    closed_top: bool = True,
    margin: float = 1e-3,
    power: float = 2.0,
    topk_fraction: float = 0.05,
    reduction: str = "topk",
) -> torch.Tensor:
    """Dimensionless barrier for localized cap loop / turn-back violations.

    ``cap_radial_fold_measure`` returns a dimensionless turn-back measure ``v``.
    Training is aligned with validation through the *relative* excess

        excess = relu(v / margin - 1)

    rather than the earlier absolute excess ``relu(v - margin)``.

    This scaling is important: a cap with turn-back 0.015 against a safety
    threshold of 0.001 is a 15x violation and receives a large penalty rather
    than a tiny ~1e-4 squared absolute error.

    The default reduction averages only the worst 5% of cap samples.
    """
    if margin <= 0:
        raise ValueError(
            "cap-fold margin must be > 0 for the normalized safety barrier"
        )
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

    # Relative-to-threshold violation: 1.0 means exactly at the allowed limit.
    excess = torch.relu(v / margin - 1.0)
    penalty = excess.pow(power)

    if reduction == "topk":
        return _topk_mean(penalty, fraction=topk_fraction)
    if reduction == "mean":
        return penalty.mean() if penalty.numel() else surface.new_zeros(())
    if reduction == "sum":
        return penalty.sum()
    if reduction == "max":
        return penalty.max() if penalty.numel() else surface.new_zeros(())
    if reduction == "none":
        return penalty

    raise ValueError("reduction must be topk, mean, sum, max, or none")


def cap_radial_fold_max(
    surface: torch.Tensor,
    RB: torch.Tensor,
    RC: torch.Tensor,
    *,
    closed_top: bool = True,
) -> torch.Tensor:
    """Maximum combined normalized cap turn-back diagnostic."""
    b, c = cap_radial_fold_measure(
        surface,
        RB,
        RC,
        closed_top=closed_top,
    )
    vals = [b.reshape(-1)]
    if c.numel():
        vals.append(c.reshape(-1))
    v = torch.cat(vals)
    return v.max() if v.numel() else surface.new_zeros(())


def jacobian_orientation_barrier_loss(
    signed_jacobian: torch.Tensor,
    area_scale: torch.Tensor,
    *,
    margin: float = 0.05,
    valid_mask: Optional[torch.Tensor] = None,
    power: float = 2.0,
    topk_fraction: float = 0.05,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Scale-independent local orientation barrier.

    ``signed_jacobian`` is the oriented area relative to the fixed classical
    reference normal and ``area_scale`` is the unsigned local area magnitude.
    Their ratio is therefore an orientation cosine-like quantity in [-1, 1]:

        orientation = signed_jacobian / area_scale

    Positive values preserve orientation; negative values are folded.
    Penalizing this normalized quantity avoids dependence on object scale or
    local parameterization density.

    ``margin=0.05`` asks for a small positive orientation reserve instead of
    merely J > 0.
    """
    if margin < 0:
        raise ValueError("orientation margin must be >= 0")
    if power <= 0:
        raise ValueError("power must be > 0")

    if valid_mask is None:
        mask = torch.ones_like(signed_jacobian, dtype=torch.bool)
    else:
        mask = valid_mask.to(
            dtype=torch.bool,
            device=signed_jacobian.device,
        )

    mask = (
        mask
        & torch.isfinite(signed_jacobian)
        & torch.isfinite(area_scale)
        & (area_scale > eps)
    )

    signed = signed_jacobian[mask]
    area = area_scale[mask]

    if signed.numel() == 0:
        return signed_jacobian.new_zeros(())

    orientation = signed / area.clamp_min(eps)
    violation = torch.relu(margin - orientation)
    penalty = violation.pow(power)

    return _topk_mean(
        penalty,
        fraction=topk_fraction,
    )


def jacobian_area_barrier_loss(
    area_scale: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    relative_margin: float = 0.05,
    power: float = 2.0,
    topk_fraction: float = 0.05,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Explicit normalized area-degeneracy barrier.

    The local unsigned area is normalized by the median evaluable area of the
    current surface (detached from autograd):

        area_ratio = area_scale / median(area_scale)

    and samples below ``relative_margin`` are penalized.  This makes the
    degeneracy term dimensionless and object-scale independent.

    Exact cap poles must be removed by ``valid_mask`` before calling.
    """
    if relative_margin <= 0:
        raise ValueError("relative area margin must be > 0")
    if power <= 0:
        raise ValueError("power must be > 0")

    if valid_mask is None:
        mask = torch.ones_like(area_scale, dtype=torch.bool)
    else:
        mask = valid_mask.to(
            dtype=torch.bool,
            device=area_scale.device,
        )

    mask = mask & torch.isfinite(area_scale) & (area_scale >= 0)

    area = area_scale[mask]
    if area.numel() == 0:
        return area_scale.new_zeros(())

    positive = area[area > eps]
    if positive.numel() == 0:
        # Entire evaluable surface has collapsed.
        return area_scale.new_tensor(1.0)

    # Reference scale is used only as normalization; detaching prevents the
    # optimizer from gaming the denominator by globally inflating all areas.
    ref = positive.detach().median().clamp_min(eps)
    ratio = area / ref

    # Relative form again: ratio == relative_margin is exactly at threshold.
    violation = torch.relu(1.0 - ratio / relative_margin)
    penalty = violation.pow(power)

    return _topk_mean(
        penalty,
        fraction=topk_fraction,
    )


def jacobian_hard_barrier_loss(
    signed_jacobian: torch.Tensor,
    *,
    margin: float = 0.05,
    valid_mask: Optional[torch.Tensor] = None,
    power: float = 2.0,
    topk_fraction: float = 0.05,
) -> torch.Tensor:
    """Compatibility wrapper for older imports.

    This function cannot form the new normalized orientation term without an
    ``area_scale`` tensor, so it retains a simple signed barrier only for
    backward compatibility.  The active NSSR-V2 training path uses
    ``geometry_regularization_loss`` below and therefore uses normalized
    orientation + explicit area-degeneracy penalties.
    """
    if margin < 0:
        raise ValueError("margin must be >= 0")
    if power <= 0:
        raise ValueError("power must be > 0")

    signed = signed_jacobian
    if valid_mask is None:
        mask = torch.isfinite(signed)
    else:
        mask = valid_mask.to(dtype=torch.bool, device=signed.device)
        mask = mask & torch.isfinite(signed)

    selected = signed[mask]
    if selected.numel() == 0:
        return signed.new_zeros(())

    violation = torch.relu(margin - selected)
    return _topk_mean(
        violation.pow(power),
        fraction=topk_fraction,
    )


def geometry_regularization_loss(
    geometry: GeometryOutput,
    *,
    lam_jacobian: float = 0.0,
    jacobian_margin: float = 0.05,
    jacobian_power: float = 2.0,
    geometry_topk_fraction: float = 0.05,
    closed_top: bool = True,
):
    """Dimensionless local orientation safety loss.

    Active training now uses ONLY the normalized orientation barrier

        orientation = signed_jacobian / area_scale

        relu(jacobian_margin - orientation)^power

    reduced over the worst ``geometry_topk_fraction`` of evaluable samples.

    Why the area-degeneracy term is excluded from training
    -------------------------------------------------------
    NSSR cap patches intentionally taper toward collapsed poles.  Even after
    excluding the exact pole rows, near-pole samples naturally have much
    smaller area than the global body median.  A global normalized-area barrier
    therefore penalizes valid classical cap tapering and produces a non-zero
    Jacobian loss even for the 100%-valid classical baseline.

    Degeneracy remains a validation diagnostic via ``jacobian.degenerate_mask``;
    it is simply not part of the active optimization objective unless future
    experiments demonstrate a real degeneracy failure mode.
    """
    device = geometry.surface.xyz.device
    dtype = geometry.surface.xyz.dtype
    zero = torch.zeros((), device=device, dtype=dtype)

    if lam_jacobian == 0.0:
        return zero, {
            "jacobian": zero,
            "jacobian_orientation": zero,
            "jacobian_area": zero,
        }

    if geometry.jacobian is None:
        raise ValueError(
            "lam_jacobian is non-zero but geometry.jacobian is missing"
        )

    jac = geometry.jacobian

    evaluable = ~intentional_pole_mask(
        geometry.surface.xyz,
        closed_top=closed_top,
    )

    l_orientation = jacobian_orientation_barrier_loss(
        jac.signed,
        jac.area_scale,
        margin=jacobian_margin,
        valid_mask=evaluable,
        power=jacobian_power,
        topk_fraction=geometry_topk_fraction,
    )

    # Kept in the returned diagnostics for API/logging compatibility.
    # It is deliberately zero in the active loss.
    l_area = zero
    l_jac = l_orientation

    return lam_jacobian * l_jac, {
        "jacobian": l_jac,
        "jacobian_orientation": l_orientation,
        "jacobian_area": l_area,
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
    surface: Optional[torch.Tensor] = None,
    RB: Optional[torch.Tensor] = None,
    RC: Optional[torch.Tensor] = None,
    closed_top: bool = True,
    geometry: Optional[GeometryOutput] = None,
    lam_jacobian: float = 0.0,
    jacobian_margin: float = 0.05,
    jacobian_power: float = 2.0,
    lam_cap_fold: float = 0.0,
    cap_fold_margin: float = 1.0e-3,
    cap_fold_power: float = 2.0,
    geometry_topk_fraction: float = 0.05,
):
    """NSSR objective with active local- and cap-safety terms.

    ``lam_cap_fold`` targets visible cap loops directly. Both geometry terms
    use hard-sample top-k reduction so localized failures cannot be diluted by
    the many safe samples on the surface.
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
            jacobian_power=jacobian_power,
            geometry_topk_fraction=geometry_topk_fraction,
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
            margin=cap_fold_margin,
            power=cap_fold_power,
            topk_fraction=geometry_topk_fraction,
            reduction="topk",
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
    "jacobian_orientation_barrier_loss",
    "jacobian_area_barrier_loss",
    "jacobian_hard_barrier_loss",
    "geometry_regularization_loss",
    "total_loss",
]
