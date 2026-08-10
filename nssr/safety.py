"""Shared sampled safety evaluation and projection for NSSR-V2.

This module is the single source of truth for the final sampled safety rule:

    SAFE = Jacobian-valid AND cap-safe

where Jacobian-valid means no flipped or accidentally degenerate samples
outside the intentional exact cap poles, and cap-safe means the maximum
cap turn-back measure is at or below ``max_cap_fold``.

Curvature is deliberately not part of active safety.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from .geometry import evaluate_geometry, surface_points, zero_params
from .losses import (
    cap_radial_fold_max,
    cap_radial_fold_measure,
    intentional_pole_mask,
)
from .metrics import evaluate_surface
from .networks import contour_features


def zero_params_like(obj):
    """Classical zero-correction parameter dictionary for an object."""
    N, m = obj["R"].shape[:2]
    return zero_params(
        N,
        m,
        device=obj["R"].device,
        dtype=obj["R"].dtype,
    )


@torch.no_grad()
def classical_geometry_and_reference(obj, n_u):
    """Return classical scored geometry and its fixed pointwise normal field.

    The first classical evaluation supplies the orientation reference.
    The second evaluates the classical surface against exactly that same
    reference, matching learned evaluation semantics.
    """
    p0 = zero_params_like(obj)

    ref_geom = evaluate_geometry(
        obj["R"], obj["Z"], obj["RB"], obj["RC"],
        obj["Bh"], obj["Th"],
        p0,
        n_u=n_u,
        closed_top=obj.get("closed_top", True),
        base_circular=obj.get("base_circular", True),
        crown_circular=obj.get("crown_circular", True),
        compute_jacobian=False,
        compute_curvature=False,
        run_validation=False,
    )
    reference_normal = ref_geom.surface.normals.detach()

    classical = evaluate_geometry(
        obj["R"], obj["Z"], obj["RB"], obj["RC"],
        obj["Bh"], obj["Th"],
        p0,
        n_u=n_u,
        closed_top=obj.get("closed_top", True),
        base_circular=obj.get("base_circular", True),
        crown_circular=obj.get("crown_circular", True),
        compute_jacobian=True,
        compute_curvature=False,
        run_validation=False,
        reference_normal=reference_normal,
    )
    return classical, reference_normal


@torch.no_grad()
def geometry_from_params(obj, params, n_u, reference_normal):
    """Evaluate learned/projected parameters with the fixed reference field."""
    return evaluate_geometry(
        obj["R"], obj["Z"], obj["RB"], obj["RC"],
        obj["Bh"], obj["Th"],
        params,
        n_u=n_u,
        closed_top=obj.get("closed_top", True),
        base_circular=obj.get("base_circular", True),
        crown_circular=obj.get("crown_circular", True),
        compute_jacobian=True,
        compute_curvature=False,
        run_validation=False,
        reference_normal=reference_normal,
    )


@torch.no_grad()
def params_from_net(net, obj):
    feats = contour_features(
        obj["R"], obj["Z"], obj["RB"], obj["RC"],
        obj["Bh"], obj["Th"],
    )
    return net(feats)


def geometry_safety_summary(geom, obj, max_cap_fold=1e-3):
    """Compute the canonical NSSR-V2 sampled safety summary."""
    if max_cap_fold <= 0:
        raise ValueError("max_cap_fold must be > 0")

    jac = geom.jacobian
    if jac is None:
        raise RuntimeError("Jacobian diagnostics were not computed")

    eval_mask = ~intentional_pole_mask(
        geom.surface.xyz,
        closed_top=obj.get("closed_top", True),
    )

    neg_mask = jac.flipped_mask & eval_mask
    deg_mask = jac.degenerate_mask & eval_mask

    denom = max(int(eval_mask.sum().item()), 1)
    neg_count = int(neg_mask.sum().item())
    deg_count = int(deg_mask.sum().item())

    j_valid = neg_count == 0 and deg_count == 0

    base_v, crown_v = cap_radial_fold_measure(
        geom.surface.xyz,
        obj["RB"],
        obj["RC"],
        closed_top=obj.get("closed_top", True),
    )
    base_max = float(base_v.max().item()) if base_v.numel() else 0.0
    crown_max = float(crown_v.max().item()) if crown_v.numel() else 0.0
    cap_max = float(
        cap_radial_fold_max(
            geom.surface.xyz,
            obj["RB"],
            obj["RC"],
            closed_top=obj.get("closed_top", True),
        ).item()
    )
    cap_safe = cap_max <= max_cap_fold

    signed_eval = jac.signed[eval_mask]
    finite_signed = signed_eval[torch.isfinite(signed_eval)]
    minimum_signed = (
        float(finite_signed.min().item())
        if finite_signed.numel()
        else float("nan")
    )

    area_eval = jac.area_scale[eval_mask]
    finite_area = area_eval[torch.isfinite(area_eval)]
    minimum_area = (
        float(finite_area.min().item())
        if finite_area.numel()
        else float("nan")
    )

    return {
        "jacobian_valid": bool(j_valid),
        "negative_jacobian": neg_count,
        "negative_fraction": float(neg_count / denom),
        "degenerate": deg_count,
        "degenerate_fraction": float(deg_count / denom),
        "minimum_signed_jacobian": minimum_signed,
        "minimum_area_scale": minimum_area,
        "base_cap_fold_max": base_max,
        "crown_cap_fold_max": crown_max,
        "cap_fold_max": cap_max,
        "cap_safe": bool(cap_safe),
        "safe": bool(j_valid and cap_safe),
    }


@torch.no_grad()
def reconstruction_metrics(obj, geom):
    """Accuracy metrics using the geometry's sampled points/normals."""
    if obj.get("gt_pts") is None:
        return {}

    pts = surface_points(geom.surface.xyz)
    nrms = geom.surface.normals.reshape(-1, 3)
    result = evaluate_surface(
        pts,
        obj["gt_pts"],
        nrms,
        obj.get("gt_normals"),
    )
    return {k: float(v) for k, v in result.items()}


def scaled_params(params, alpha):
    """Scale all learned corrections toward the classical zero endpoint."""
    return {k: alpha * v for k, v in params.items()}


@torch.no_grad()
def evaluate_params(
    obj,
    params,
    n_u,
    reference_normal,
    max_cap_fold=1e-3,
    with_metrics=True,
):
    """Evaluate one parameter state using the canonical safety definition."""
    geom = geometry_from_params(
        obj, params, n_u, reference_normal
    )
    safety = geometry_safety_summary(
        geom, obj, max_cap_fold=max_cap_fold
    )
    metrics = reconstruction_metrics(obj, geom) if with_metrics else {}
    return geom, safety, metrics


@torch.no_grad()
def project_all_to_safe(
    obj,
    params,
    n_u,
    reference_normal,
    max_cap_fold=1e-3,
    max_iter=40,
    with_metrics=True,
):
    """Project all corrections toward classical until sampled SAFE.

    Returns:
        projected_params, alpha, raw_safety, raw_metrics,
        post_safety, post_metrics

    ``alpha=1`` means unchanged.  ``alpha=0`` is the classical endpoint.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    _, raw_safety, raw_metrics = evaluate_params(
        obj,
        params,
        n_u,
        reference_normal,
        max_cap_fold=max_cap_fold,
        with_metrics=with_metrics,
    )
    if raw_safety["safe"]:
        return (
            params, 1.0,
            raw_safety, raw_metrics,
            raw_safety, raw_metrics,
        )

    p0 = scaled_params(params, 0.0)
    _, zero_safety, zero_metrics = evaluate_params(
        obj,
        p0,
        n_u,
        reference_normal,
        max_cap_fold=max_cap_fold,
        with_metrics=with_metrics,
    )
    if not zero_safety["safe"]:
        raise RuntimeError(
            "Classical endpoint is not sampled-safe under the shared "
            "Jacobian/cap safety definition."
        )

    lo, hi = 0.0, 1.0
    best_params = p0
    best_safety = zero_safety
    best_metrics = zero_metrics

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cand = scaled_params(params, mid)
        _, cand_safety, cand_metrics = evaluate_params(
            obj,
            cand,
            n_u,
            reference_normal,
            max_cap_fold=max_cap_fold,
            with_metrics=with_metrics,
        )
        if cand_safety["safe"]:
            lo = mid
            best_params = cand
            best_safety = cand_safety
            best_metrics = cand_metrics
        else:
            hi = mid

    return (
        best_params, lo,
        raw_safety, raw_metrics,
        best_safety, best_metrics,
    )



def _scaled_selected_params(params, alpha, keys):
    """Scale only selected correction groups; preserve all other parameters."""
    keyset = set(keys)
    out = {}
    for k, v in params.items():
        out[k] = alpha * v if k in keyset else v
    return out


@torch.no_grad()
def _binary_search_projection(
    obj,
    params,
    keys,
    n_u,
    reference_normal,
    max_cap_fold,
    max_iter,
    with_metrics,
):
    """Find the largest safe alpha when scaling only ``keys``.

    Returns ``None`` if the alpha=0 endpoint for this stage is not safe.
    """
    p0 = _scaled_selected_params(params, 0.0, keys)
    _, s0, m0 = evaluate_params(
        obj,
        p0,
        n_u,
        reference_normal,
        max_cap_fold=max_cap_fold,
        with_metrics=with_metrics,
    )
    if not s0["safe"]:
        return None

    lo, hi = 0.0, 1.0
    best_params = p0
    best_safety = s0
    best_metrics = m0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cand = _scaled_selected_params(params, mid, keys)
        _, cs, cm = evaluate_params(
            obj,
            cand,
            n_u,
            reference_normal,
            max_cap_fold=max_cap_fold,
            with_metrics=with_metrics,
        )
        if cs["safe"]:
            lo = mid
            best_params = cand
            best_safety = cs
            best_metrics = cm
        else:
            hi = mid

    return best_params, lo, best_safety, best_metrics


@torch.no_grad()
def project_staged_to_safe(
    obj,
    params,
    n_u,
    reference_normal,
    max_cap_fold=1e-3,
    max_iter=40,
    with_metrics=True,
):
    """Staged sampled-safety projection.

    Projection policy
    -----------------
    1. If raw reconstruction is already SAFE, keep it unchanged.
    2. For any unsafe reconstruction, first try scaling only tangent controls
       ``s_a``, ``s_b``, and ``s_tau``.
    3. If tangent-only scaling cannot reach a safe endpoint, fall back to
       scaling all learned correction parameters toward the classical solution.

    This preserves the current global projection as a guaranteed fallback
    whenever the classical endpoint is sampled-safe.

    Returns:
        projected_params,
        alpha,
        stage,               # "none", "tangent", or "all"
        raw_safety,
        raw_metrics,
        post_safety,
        post_metrics
    """
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    _, raw_safety, raw_metrics = evaluate_params(
        obj,
        params,
        n_u,
        reference_normal,
        max_cap_fold=max_cap_fold,
        with_metrics=with_metrics,
    )

    if raw_safety["safe"]:
        return (
            params,
            1.0,
            "none",
            raw_safety,
            raw_metrics,
            raw_safety,
            raw_metrics,
        )

    tangent_keys = ("s_a", "s_b", "s_tau")
    tangent = _binary_search_projection(
        obj,
        params,
        tangent_keys,
        n_u,
        reference_normal,
        max_cap_fold,
        max_iter,
        with_metrics,
    )
    if tangent is not None:
        p, alpha, safety_out, metrics_out = tangent
        return (
            p,
            alpha,
            "tangent",
            raw_safety,
            raw_metrics,
            safety_out,
            metrics_out,
        )

    # Final guaranteed fallback: scale all learned corrections.
    all_keys = tuple(params.keys())
    all_result = _binary_search_projection(
        obj,
        params,
        all_keys,
        n_u,
        reference_normal,
        max_cap_fold,
        max_iter,
        with_metrics,
    )
    if all_result is None:
        raise RuntimeError(
            "Classical endpoint is not sampled-safe under the shared "
            "Jacobian/cap safety definition."
        )

    p, alpha, safety_out, metrics_out = all_result
    return (
        p,
        alpha,
        "all",
        raw_safety,
        raw_metrics,
        safety_out,
        metrics_out,
    )


__all__ = [
    "zero_params_like",
    "classical_geometry_and_reference",
    "geometry_from_params",
    "params_from_net",
    "geometry_safety_summary",
    "reconstruction_metrics",
    "scaled_params",
    "evaluate_params",
    "project_all_to_safe",
    "project_staged_to_safe",
]
