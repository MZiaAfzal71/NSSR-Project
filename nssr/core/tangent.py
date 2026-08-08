"""NSSR-V2 adaptive tangent-field construction.

This module converts sampled contour points into stable parameter-space
tangents for Hermite surface construction.

The key design goal is to prevent the classic cubic-Hermite overshoot that
can produce loops near an apple crown/shoulder or other rapidly changing
regions.  The algorithm therefore combines:

1. centered/one-sided finite differences,
2. non-uniform spacing awareness,
3. a monotonicity/overshoot limiter,
4. magnitude limiting,
5. optional curvature-aware damping.

No neural-network or surface-specific policy is embedded here.  The module
operates on ordered contour samples and can therefore be unit tested
independently.

Input convention
----------------
A contour is an array with shape (..., N, D), normally D=2 or D=3.
The N dimension is the contour/slice direction.

For a 2-D meridian contour the coordinates may be (r, z), while a 3-D
contour may be used directly.  The tangent returned has the same shape.

The parameter coordinate is supplied separately when available.  This is
important: using sample indices as if spacing were uniform is a common source
of excessive Hermite derivatives.

The limiter is inspired by shape-preserving cubic interpolation: when a
component changes sign across a local interval, the tangent at the shared
node is suppressed rather than allowed to create an extremum that the data
do not support.  A generalized vector version additionally limits tangent
magnitude relative to adjacent secants.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _is_torch(x: ArrayLike) -> bool:
    return isinstance(x, torch.Tensor)


def _stack(values, axis: int = -2):
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values, dim=axis)
    return np.stack(values, axis=axis)


def _zeros_like(x):
    return torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)


def _ones_like(x):
    return torch.ones_like(x) if isinstance(x, torch.Tensor) else np.ones_like(x)


def _sqrt(x):
    return torch.sqrt(x) if isinstance(x, torch.Tensor) else np.sqrt(x)


def _abs(x):
    return torch.abs(x) if isinstance(x, torch.Tensor) else np.abs(x)


def _maximum(a, b):
    return torch.maximum(a, b) if isinstance(a, torch.Tensor) else np.maximum(a, b)


def _minimum(a, b):
    return torch.minimum(a, b) if isinstance(a, torch.Tensor) else np.minimum(a, b)


def _where(condition, a, b):
    return torch.where(condition, a, b) if isinstance(a, torch.Tensor) else np.where(condition, a, b)


def _norm(x, eps: float):
    if isinstance(x, torch.Tensor):
        return torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def _dot(a, b):
    if isinstance(a, torch.Tensor):
        return torch.sum(a * b, dim=-1, keepdim=True)
    return np.sum(a * b, axis=-1, keepdims=True)


def _clip(x, lo, hi):
    if isinstance(x, torch.Tensor):
        return torch.clamp(x, min=lo, max=hi)
    return np.clip(x, lo, hi)


def _validate_points(points: ArrayLike) -> None:
    if not isinstance(points, (np.ndarray, torch.Tensor)):
        raise TypeError("points must be a NumPy array or PyTorch tensor")
    if points.ndim < 2:
        raise ValueError("points must have shape (..., N, D)")
    if points.shape[-2] < 2:
        raise ValueError("at least two contour samples are required")
    if points.shape[-1] < 1:
        raise ValueError("point dimension must be positive")


def _validate_parameter(t: ArrayLike, points: ArrayLike) -> None:
    if not isinstance(t, (np.ndarray, torch.Tensor)):
        raise TypeError("parameter must be a NumPy array or PyTorch tensor")
    if t.shape[-1] != points.shape[-2]:
        raise ValueError(
            "parameter must have one coordinate per point: "
            f"expected {points.shape[-2]}, got {t.shape[-1]}"
        )


def _broadcast_parameter(t: ArrayLike, points: ArrayLike) -> ArrayLike:
    """Convert parameter coordinates to (..., N, 1) for vector arithmetic."""
    if t.ndim == 1:
        if isinstance(t, torch.Tensor):
            return t.reshape(1, -1, 1).expand(*points.shape[:-2], -1, 1)
        return np.broadcast_to(
            t.reshape((1,) * (points.ndim - 2) + (t.shape[0], 1)),
            points.shape[:-1] + (1,),
        )
    return t[..., :, None]


def secant_slopes(
    points: ArrayLike,
    parameter: Optional[ArrayLike] = None,
    *,
    eps: float = 1.0e-8,
) -> ArrayLike:
    """Compute interval secants ``(P[i+1]-P[i]) / (t[i+1]-t[i])``.

    Parameters
    ----------
    points:
        Shape ``(..., N, D)``.
    parameter:
        Shape ``(N,)`` or ``(..., N)``.  If omitted, integer-spaced samples
        are assumed.
    eps:
        Minimum spacing used to avoid division by zero.

    Returns
    -------
    array
        Shape ``(..., N-1, D)``.
    """
    _validate_points(points)

    n = points.shape[-2]
    if parameter is None:
        if isinstance(points, torch.Tensor):
            dt = torch.ones(
                (n - 1,), dtype=points.dtype, device=points.device
            )
        else:
            dt = np.ones((n - 1,), dtype=points.dtype)
        dt = dt.reshape((1,) * (points.ndim - 2) + (n - 1, 1))
    else:
        _validate_parameter(parameter, points)
        t = _broadcast_parameter(parameter, points)
        dt = t[..., 1:, :] - t[..., :-1, :]
        if isinstance(dt, torch.Tensor):
            dt = torch.sign(dt) * dt.abs().clamp_min(eps)
        else:
            dt = np.where(np.abs(dt) < eps, np.where(dt < 0, -eps, eps), dt)

    return (points[..., 1:, :] - points[..., :-1, :]) / dt


def _endpoint_tangent(
    secants: ArrayLike,
    *,
    at_start: bool,
    monotone: bool,
) -> ArrayLike:
    """One-sided endpoint derivative with optional overshoot protection."""
    if at_start:
        s0 = secants[..., 0, :]
        if secants.shape[-2] == 1:
            return s0
        s1 = secants[..., 1, :]
        # Quadratic one-sided estimate: 1.5*s0 - 0.5*s1.
        tangent = 1.5 * s0 - 0.5 * s1
        if monotone:
            # If the first two secants reverse direction componentwise,
            # do not manufacture an endpoint extremum.
            same = s0 * s1 > 0
            tangent = _where(same, tangent, _zeros_like(tangent))
        return tangent

    s0 = secants[..., -1, :]
    if secants.shape[-2] == 1:
        return s0
    sm1 = secants[..., -2, :]
    tangent = 1.5 * s0 - 0.5 * sm1
    if monotone:
        same = s0 * sm1 > 0
        tangent = _where(same, tangent, _zeros_like(tangent))
    return tangent


def _fritsch_carlson_component(
    left: ArrayLike,
    right: ArrayLike,
    *,
    eps: float,
) -> ArrayLike:
    """Shape-preserving derivative for two neighboring secants.

    This is the weighted harmonic mean form used for monotone cubic
    interpolation.  It returns zero if the neighboring slopes have
    incompatible signs.
    """
    same_sign = left * right > 0
    if isinstance(left, torch.Tensor):
        w1 = torch.ones_like(left)
        w2 = torch.ones_like(right)
    else:
        w1 = np.ones_like(left)
        w2 = np.ones_like(right)

    # Equal weights are deliberately used here because interval spacing has
    # already been incorporated into the secants.  The resulting harmonic
    # mean is robust and easy to audit.
    denom = left + right
    harmonic = 2.0 * left * right / _where(
        _abs(denom) > eps,
        denom,
        _ones_like(denom),
    )
    return _where(same_sign, harmonic, _zeros_like(harmonic))


def _limit_component(
    tangent: ArrayLike,
    left: ArrayLike,
    right: ArrayLike,
    *,
    factor: float,
    eps: float,
) -> ArrayLike:
    """Limit a tangent against adjacent scalar secants."""
    same = left * right > 0
    bound = factor * _minimum(_abs(left), _abs(right))
    limited = _where(_abs(tangent) > bound, _clip(tangent, -bound, bound), tangent)
    return _where(same, limited, _zeros_like(tangent))


def monotone_tangents(
    points: ArrayLike,
    parameter: Optional[ArrayLike] = None,
    *,
    componentwise: bool = True,
    limit_factor: float = 3.0,
    eps: float = 1.0e-8,
) -> ArrayLike:
    """Compute shape-preserving tangents from ordered samples.

    Interior tangents use a monotone cubic derivative when the adjacent
    secants have compatible signs.  At a local extremum, the tangent is zero.
    This is intentionally conservative: a cubic Hermite segment cannot create
    a loop from an unconstrained derivative if the derivative is prevented
    from exceeding the neighboring data trend.

    Parameters
    ----------
    componentwise:
        If true, each coordinate is limited independently.  This is useful
        for meridian (r,z) data because it preserves monotonicity of each
        coordinate.  If false, the derivative is subsequently treated as a
        vector field and only its magnitude is limited.
    limit_factor:
        Upper tangent/secant multiplier.  Values around 3 are conservative
        for cubic Hermite interpolation.
    """
    _validate_points(points)
    sec = secant_slopes(points, parameter, eps=eps)
    n = points.shape[-2]

    out_shape = points.shape
    tangents = _zeros_like(points)

    # Endpoints.
    tangents[..., 0, :] = _endpoint_tangent(
        sec, at_start=True, monotone=True
    )
    tangents[..., -1, :] = _endpoint_tangent(
        sec, at_start=False, monotone=True
    )

    if n > 2:
        left = sec[..., :-1, :]
        right = sec[..., 1:, :]
        interior = _fritsch_carlson_component(left, right, eps=eps)

        if componentwise:
            interior = _limit_component(
                interior, left, right, factor=limit_factor, eps=eps
            )

        tangents[..., 1:-1, :] = interior

    return tangents


def vector_limited_tangents(
    points: ArrayLike,
    parameter: Optional[ArrayLike] = None,
    *,
    base_limit: float = 3.0,
    eps: float = 1.0e-8,
) -> ArrayLike:
    """Compute tangents using a vector-direction limiter.

    This is preferable when the contour is geometrically smooth but its
    individual coordinates are not monotone.  The tangent direction is based
    on neighboring secants, while its magnitude is bounded by the smaller
    neighboring secant magnitude.

    Unlike a scalar component limiter, this does not independently distort
    the coordinate directions.
    """
    _validate_points(points)
    sec = secant_slopes(points, parameter, eps=eps)
    n = points.shape[-2]
    tangent = _zeros_like(points)

    tangent[..., 0, :] = _endpoint_tangent(sec, at_start=True, monotone=False)
    tangent[..., -1, :] = _endpoint_tangent(sec, at_start=False, monotone=False)

    if n <= 2:
        return tangent

    left = sec[..., :-1, :]
    right = sec[..., 1:, :]

    ln = _norm(left, eps)
    rn = _norm(right, eps)

    # Normalized average direction.  If the directions oppose one another,
    # the average becomes small and the tangent is suppressed.
    direction = left / ln + right / rn
    direction_norm = _norm(direction, eps)
    direction = direction / direction_norm

    magnitude = base_limit * _minimum(ln, rn)
    tangent[..., 1:-1, :] = direction * magnitude

    # If neighboring secants are nearly opposite, force a stationary tangent.
    cosine = _dot(left, right) / (ln * rn)
    valid_direction = cosine > 0.0
    tangent[..., 1:-1, :] = _where(
        valid_direction,
        tangent[..., 1:-1, :],
        _zeros_like(tangent[..., 1:-1, :]),
    )

    return tangent


def adaptive_tangents(
    points: ArrayLike,
    parameter: Optional[ArrayLike] = None,
    *,
    mode: str = "componentwise",
    limit_factor: float = 3.0,
    damping: Optional[ArrayLike] = None,
    eps: float = 1.0e-8,
) -> ArrayLike:
    """Build a stable tangent field.

    Parameters
    ----------
    points:
        Ordered contour samples, shape ``(...,N,D)``.
    parameter:
        Non-uniform parameter coordinates.  Supplying these is strongly
        recommended when slice spacing is non-uniform.
    mode:
        ``"componentwise"`` for conservative monotonicity control or
        ``"vector"`` for direction-preserving limiting.
    limit_factor:
        Tangent magnitude multiplier.
    damping:
        Optional scalar or per-point factor. Values below one damp tangents;
        values above one are allowed but are clipped to one for safety.
        This hook is intended for later curvature/confidence integration.
    eps:
        Numerical tolerance.

    Returns
    -------
    array
        Tangent field with shape ``(...,N,D)``.
    """
    if mode == "componentwise":
        tangent = monotone_tangents(
            points, parameter,
            componentwise=True,
            limit_factor=limit_factor,
            eps=eps,
        )
    elif mode == "vector":
        tangent = vector_limited_tangents(
            points, parameter,
            base_limit=limit_factor,
            eps=eps,
        )
    else:
        raise ValueError("mode must be 'componentwise' or 'vector'")

    if damping is not None:
        d = damping
        if isinstance(d, torch.Tensor):
            d = torch.clamp(d, min=0.0, max=1.0)
            while d.ndim < tangent.ndim:
                d = d.unsqueeze(-1)
        else:
            d = np.clip(d, 0.0, 1.0)
            while d.ndim < tangent.ndim:
                d = np.expand_dims(d, axis=-1)
        tangent = tangent * d

    return tangent


def tangent_energy(tangent: ArrayLike, *, eps: float = 1.0e-8) -> ArrayLike:
    """Return squared tangent magnitude, useful for diagnostics/losses."""
    if isinstance(tangent, torch.Tensor):
        return torch.sum(tangent * tangent, dim=-1)
    return np.sum(tangent * tangent, axis=-1)


def tangent_magnitude(tangent: ArrayLike, *, eps: float = 1.0e-8) -> ArrayLike:
    """Return tangent magnitude."""
    if isinstance(tangent, torch.Tensor):
        return torch.linalg.vector_norm(tangent, dim=-1)
    return np.linalg.norm(tangent, axis=-1)


__all__ = [
    "secant_slopes",
    "monotone_tangents",
    "vector_limited_tangents",
    "adaptive_tangents",
    "tangent_energy",
    "tangent_magnitude",
]
