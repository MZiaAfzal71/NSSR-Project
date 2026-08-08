"""NSSR-V2 surface Jacobian and fold-orientation diagnostics.

For a parametric surface S(u,v): R^2 -> R^3, the differential is 3x2.
The meaningful local quantities are:

    area_vector = Su x Sv
    area_scale  = ||Su x Sv||

A signed value requires an expected orientation n_ref:

    signed_jacobian = dot(Su x Sv, n_ref)

Negative signed values indicate orientation reversal relative to the chosen
reference. During training, prefer a fixed classical/reference orientation
rather than estimating orientation from each prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass(slots=True)
class JacobianEvaluation:
    """Local surface-area and orientation diagnostics."""

    area_vector: ArrayLike
    area_scale: ArrayLike
    signed: ArrayLike
    unit_normal: ArrayLike
    reference_normal: ArrayLike
    valid_mask: ArrayLike
    degenerate_mask: ArrayLike
    flipped_mask: ArrayLike
    near_flip_mask: ArrayLike

    @property
    def negative_fraction(self):
        return _masked_fraction(self.flipped_mask, self.valid_mask)

    @property
    def degenerate_fraction(self):
        if isinstance(self.degenerate_mask, torch.Tensor):
            return self.degenerate_mask.to(self.area_scale.dtype).mean()
        return np.asarray(self.degenerate_mask, dtype=float).mean()

    @property
    def minimum_signed(self):
        return _masked_min(self.signed, self.valid_mask)


def _is_torch(x):
    return isinstance(x, torch.Tensor)


def _cross(a, b):
    return torch.linalg.cross(a, b, dim=-1) if _is_torch(a) else np.cross(a, b, axis=-1)


def _dot(a, b):
    return torch.sum(a * b, dim=-1) if _is_torch(a) else np.sum(a * b, axis=-1)


def _norm(v, keepdim=False):
    return (
        torch.linalg.vector_norm(v, dim=-1, keepdim=keepdim)
        if _is_torch(v)
        else np.linalg.norm(v, axis=-1, keepdims=keepdim)
    )


def _normalize(v, eps=1e-12):
    n = _norm(v, keepdim=True)
    if _is_torch(v):
        out = v / torch.clamp(n, min=eps)
        return torch.where(n > eps, out, torch.zeros_like(out))
    out = v / np.maximum(n, eps)
    return np.where(n > eps, out, np.zeros_like(out))


def _isfinite(x):
    return torch.isfinite(x) if _is_torch(x) else np.isfinite(x)


def _masked_fraction(mask, valid):
    if _is_torch(mask):
        num = (mask & valid).to(torch.float32).sum()
        den = valid.to(torch.float32).sum().clamp_min(1.0)
        return num / den
    return np.logical_and(mask, valid).sum() / max(float(np.asarray(valid).sum()), 1.0)


def _masked_min(values, valid):
    if _is_torch(values):
        if not bool(valid.any()):
            return torch.full((), float("nan"), dtype=values.dtype, device=values.device)
        return torch.where(valid, values, torch.full_like(values, float("inf"))).amin()
    if not np.any(valid):
        return np.asarray(np.nan, dtype=values.dtype)
    return np.min(np.where(valid, values, np.inf))


def _broadcast_reference(reference_normal, target):
    if type(reference_normal) is not type(target):
        raise TypeError("reference_normal and derivatives must use the same backend")
    if reference_normal.shape[-1] != 3:
        raise ValueError("reference_normal must have final dimension 3")
    ref = _normalize(reference_normal)
    if _is_torch(target):
        return torch.broadcast_to(ref, target.shape)
    return np.broadcast_to(ref, target.shape)


def estimate_reference_normal(area_vector, *, sample_dims=None, eps=1e-12):
    """Estimate a broadcastable orientation reference from area vectors.

    For area vectors shaped (P,U,V,3), use sample_dims=(1,2) to estimate one
    orientation per patch. This is intended for diagnostics; training should
    usually use a fixed reference surface.
    """
    if area_vector.shape[-1] != 3:
        raise ValueError("area_vector must have final dimension 3")
    if sample_dims is None:
        sample_dims = tuple(range(area_vector.ndim - 1))
    if _is_torch(area_vector):
        ref = area_vector.sum(dim=sample_dims, keepdim=True)
    else:
        ref = area_vector.sum(axis=sample_dims, keepdims=True)
    return _normalize(ref, eps)


def surface_jacobian(
    Su,
    Sv,
    *,
    reference_normal=None,
    reference_sample_dims=None,
    min_area=1e-8,
    orientation_tol=1e-8,
    normal_eps=1e-12,
):
    """Compute local area scaling and orientation validity.

    Su and Sv must have shape (...,3). If reference_normal is omitted, an
    orientation is estimated from the current area vectors.
    """
    if type(Su) is not type(Sv):
        raise TypeError("Su and Sv must use the same backend")
    if Su.shape != Sv.shape or Su.shape[-1] != 3:
        raise ValueError("Su and Sv must share shape (...,3)")

    area_vector = _cross(Su, Sv)
    area_scale = _norm(area_vector)
    unit_normal = _normalize(area_vector, normal_eps)

    finite_vec = _isfinite(area_vector)
    finite_vec = finite_vec.all(dim=-1) if _is_torch(finite_vec) else finite_vec.all(axis=-1)
    finite_area = _isfinite(area_scale)

    degenerate = (~finite_vec) | (~finite_area) | (area_scale <= min_area)
    valid = ~degenerate

    if reference_normal is None:
        reference_normal = estimate_reference_normal(
            area_vector,
            sample_dims=reference_sample_dims,
            eps=normal_eps,
        )

    reference = _broadcast_reference(reference_normal, area_vector)
    signed = _dot(area_vector, reference)

    flipped = valid & (signed < -orientation_tol)
    near_flip = valid & (signed >= -orientation_tol) & (signed <= orientation_tol)

    return JacobianEvaluation(
        area_vector,
        area_scale,
        signed,
        unit_normal,
        reference,
        valid,
        degenerate,
        flipped,
        near_flip,
    )


def jacobian_barrier_loss(
    signed_jacobian,
    *,
    margin=1e-4,
    valid_mask=None,
    power=2.0,
):
    """Penalize signed Jacobians below a positive margin."""
    if power <= 0:
        raise ValueError("power must be positive")

    if _is_torch(signed_jacobian):
        target = torch.as_tensor(
            margin,
            dtype=signed_jacobian.dtype,
            device=signed_jacobian.device,
        )
        penalty = torch.relu(target - signed_jacobian).pow(power)
        if valid_mask is None:
            return penalty.mean()
        w = valid_mask.to(penalty.dtype)
        return (penalty * w).sum() / w.sum().clamp_min(1.0)

    penalty = np.maximum(margin - signed_jacobian, 0.0) ** power
    if valid_mask is None:
        return penalty.mean()
    w = np.asarray(valid_mask, dtype=penalty.dtype)
    return (penalty * w).sum() / max(float(w.sum()), 1.0)


def normal_orientation_loss(unit_normal, reference_normal, *, valid_mask=None):
    """Penalize orientation disagreement without using absolute cosine."""
    reference = _broadcast_reference(reference_normal, unit_normal)
    penalty = 1.0 - _dot(unit_normal, reference)

    if valid_mask is None:
        return penalty.mean()

    if _is_torch(penalty):
        w = valid_mask.to(penalty.dtype)
        return (penalty * w).sum() / w.sum().clamp_min(1.0)

    w = np.asarray(valid_mask, dtype=penalty.dtype)
    return (penalty * w).sum() / max(float(w.sum()), 1.0)


def fold_count(result):
    """Count reversed-orientation samples."""
    if _is_torch(result.flipped_mask):
        return int(result.flipped_mask.sum().detach().cpu().item())
    return int(np.asarray(result.flipped_mask).sum())


def patch_fold_fraction(result, *, patch_dim=0):
    """Return flipped fraction independently for each patch."""
    mask = result.flipped_mask & result.valid_mask
    valid = result.valid_mask
    reduce_dims = tuple(d for d in range(mask.ndim) if d != patch_dim)

    if _is_torch(mask):
        num = mask.to(torch.float32).sum(dim=reduce_dims)
        den = valid.to(torch.float32).sum(dim=reduce_dims).clamp_min(1.0)
        return num / den

    num = mask.astype(np.float64).sum(axis=reduce_dims)
    den = np.maximum(valid.astype(np.float64).sum(axis=reduce_dims), 1.0)
    return num / den


__all__ = [
    "JacobianEvaluation",
    "estimate_reference_normal",
    "surface_jacobian",
    "jacobian_barrier_loss",
    "normal_orientation_loss",
    "fold_count",
    "patch_fold_fraction",
]
