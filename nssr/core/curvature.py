"""NSSR-V2 differential surface curvature.

This module computes Gaussian, mean, and principal curvatures from the first
and second derivatives returned by ``nssr.core.hermite``.

For a parametric surface S(u,v):

First fundamental form
----------------------
    E = <Su, Su>
    F = <Su, Sv>
    G = <Sv, Sv>

Second fundamental form
-----------------------
    e = <n, Suu>
    f = <n, Suv>
    g = <n, Svv>

Gaussian curvature
------------------
    K = (e g - f^2) / (E G - F^2)

Mean curvature
--------------
    H = (E g - 2 F f + G e) / (2 (E G - F^2))

Principal curvatures
--------------------
    k1, k2 = H +/- sqrt(max(H^2 - K, 0))

The PyTorch path is differentiable and suitable for curvature regularization.

This module intentionally does not decide whether a given curvature is
"good" or "bad". Thresholds and repair policy belong in validation.py.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch

from .types import CurvatureResult, SurfaceEvaluation

ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _is_torch(x: ArrayLike) -> bool:
    return isinstance(x, torch.Tensor)


def _dot(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    if _is_torch(a):
        return torch.sum(a * b, dim=-1)
    return np.sum(a * b, axis=-1)


def _cross(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    if _is_torch(a):
        return torch.linalg.cross(a, b, dim=-1)
    return np.cross(a, b, axis=-1)


def _norm(v: ArrayLike, *, keepdim: bool = False) -> ArrayLike:
    if _is_torch(v):
        return torch.linalg.vector_norm(v, dim=-1, keepdim=keepdim)
    return np.linalg.norm(v, axis=-1, keepdims=keepdim)


def _normalize(v: ArrayLike, eps: float = 1.0e-12) -> ArrayLike:
    n = _norm(v, keepdim=True)
    if _is_torch(v):
        out = v / torch.clamp(n, min=eps)
        return torch.where(n > eps, out, torch.zeros_like(out))
    out = v / np.maximum(n, eps)
    return np.where(n > eps, out, np.zeros_like(out))


def _sqrt_nonnegative(x: ArrayLike) -> ArrayLike:
    if _is_torch(x):
        return torch.sqrt(torch.clamp(x, min=0.0))
    return np.sqrt(np.maximum(x, 0.0))


def _isfinite(x: ArrayLike) -> ArrayLike:
    return torch.isfinite(x) if _is_torch(x) else np.isfinite(x)


def _where(mask: ArrayLike, a: ArrayLike, b: ArrayLike) -> ArrayLike:
    if _is_torch(a):
        return torch.where(mask, a, b)
    return np.where(mask, a, b)


def _zeros_like(x: ArrayLike) -> ArrayLike:
    return torch.zeros_like(x) if _is_torch(x) else np.zeros_like(x)


def _ones_like(x: ArrayLike) -> ArrayLike:
    return torch.ones_like(x) if _is_torch(x) else np.ones_like(x)


def _abs(x: ArrayLike) -> ArrayLike:
    return torch.abs(x) if _is_torch(x) else np.abs(x)


# ---------------------------------------------------------------------------
# Fundamental forms
# ---------------------------------------------------------------------------

def first_fundamental_form(
    Su: ArrayLike,
    Sv: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Return coefficients ``E, F, G`` of the first fundamental form."""
    if type(Su) is not type(Sv):
        raise TypeError("Su and Sv must use the same backend")
    if Su.shape != Sv.shape or Su.shape[-1] != 3:
        raise ValueError("Su and Sv must share shape (...,3)")

    E = _dot(Su, Su)
    F = _dot(Su, Sv)
    G = _dot(Sv, Sv)
    return E, F, G


def surface_normal(
    Su: ArrayLike,
    Sv: ArrayLike,
    *,
    eps: float = 1.0e-12,
) -> ArrayLike:
    """Compute robust unit normals from first derivatives."""
    return _normalize(_cross(Su, Sv), eps=eps)


def second_fundamental_form(
    Suu: ArrayLike,
    Suv: ArrayLike,
    Svv: ArrayLike,
    normal: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Return coefficients ``e, f, g`` of the second fundamental form."""
    if not (
        type(Suu) is type(Suv)
        and type(Suv) is type(Svv)
        and type(Svv) is type(normal)
    ):
        raise TypeError("all inputs must use the same backend")

    if not (
        Suu.shape == Suv.shape == Svv.shape == normal.shape
        and Suu.shape[-1] == 3
    ):
        raise ValueError(
            "Suu, Suv, Svv, and normal must all have shape (...,3)"
        )

    e = _dot(normal, Suu)
    f = _dot(normal, Suv)
    g = _dot(normal, Svv)
    return e, f, g


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------

def curvatures_from_derivatives(
    Su: ArrayLike,
    Sv: ArrayLike,
    Suu: ArrayLike,
    Suv: ArrayLike,
    Svv: ArrayLike,
    *,
    normal: Optional[ArrayLike] = None,
    metric_eps: float = 1.0e-12,
    normal_eps: float = 1.0e-12,
    invalid_value: float = 0.0,
) -> CurvatureResult:
    """Compute Gaussian, mean, and principal curvatures.

    Parameters
    ----------
    Su, Sv
        First derivatives, shape ``(...,3)``.
    Suu, Suv, Svv
        Second derivatives, shape ``(...,3)``.
    normal
        Optional unit normal. If omitted it is computed from ``Su x Sv``.
    metric_eps
        Minimum valid value of ``EG - F^2``. Near-zero values correspond to a
        degenerate surface parameterization.
    normal_eps
        Tolerance used when normalizing ``Su x Sv``.
    invalid_value
        Value written at degenerate/non-finite samples.

    Returns
    -------
    CurvatureResult
        ``mean``, ``gaussian``, ``principal_1``, ``principal_2``.

    Notes
    -----
    Principal curvature ordering is

        principal_1 = H + sqrt(H^2-K)
        principal_2 = H - sqrt(H^2-K)

    so principal_1 >= principal_2 for valid real-valued samples.
    """
    arrays = (Su, Sv, Suu, Suv, Svv)
    first_type = type(Su)

    if not all(type(x) is first_type for x in arrays):
        raise TypeError("all derivative arrays must use the same backend")

    shape = Su.shape
    if not all(x.shape == shape for x in arrays):
        raise ValueError("all derivative arrays must have identical shapes")

    if Su.shape[-1] != 3:
        raise ValueError("surface derivatives must have final dimension 3")

    if normal is None:
        normal = surface_normal(Su, Sv, eps=normal_eps)
    else:
        if type(normal) is not first_type:
            raise TypeError("normal must use the same backend as derivatives")
        if normal.shape != shape:
            raise ValueError("normal must have the same shape as derivatives")
        normal = _normalize(normal, eps=normal_eps)

    E, F, G = first_fundamental_form(Su, Sv)
    e, f, g = second_fundamental_form(Suu, Suv, Svv, normal)

    metric_det = E * G - F * F

    finite = (
        _isfinite(E)
        & _isfinite(F)
        & _isfinite(G)
        & _isfinite(e)
        & _isfinite(f)
        & _isfinite(g)
        & _isfinite(metric_det)
    )

    valid = finite & (metric_det > metric_eps)

    if _is_torch(metric_det):
        safe_metric = torch.where(
            valid,
            metric_det,
            torch.ones_like(metric_det),
        )
    else:
        safe_metric = np.where(
            valid,
            metric_det,
            np.ones_like(metric_det),
        )

    gaussian = (e * g - f * f) / safe_metric
    mean = (E * g - 2.0 * F * f + G * e) / (2.0 * safe_metric)

    discriminant = mean * mean - gaussian
    root = _sqrt_nonnegative(discriminant)

    k1 = mean + root
    k2 = mean - root

    fill = _zeros_like(mean) + invalid_value

    gaussian = _where(valid, gaussian, fill)
    mean = _where(valid, mean, fill)
    k1 = _where(valid, k1, fill)
    k2 = _where(valid, k2, fill)

    return CurvatureResult(
        mean=mean,
        gaussian=gaussian,
        principal_1=k1,
        principal_2=k2,
    )


def curvature_from_evaluation(
    evaluation: SurfaceEvaluation,
    *,
    metric_eps: float = 1.0e-12,
    normal_eps: float = 1.0e-12,
    invalid_value: float = 0.0,
) -> CurvatureResult:
    """Compute curvature directly from ``SurfaceEvaluation``."""
    return curvatures_from_derivatives(
        evaluation.Su,
        evaluation.Sv,
        evaluation.Suu,
        evaluation.Suv,
        evaluation.Svv,
        normal=evaluation.normals,
        metric_eps=metric_eps,
        normal_eps=normal_eps,
        invalid_value=invalid_value,
    )


# ---------------------------------------------------------------------------
# Validity and diagnostic helpers
# ---------------------------------------------------------------------------

def curvature_valid_mask(
    Su: ArrayLike,
    Sv: ArrayLike,
    *,
    metric_eps: float = 1.0e-12,
) -> ArrayLike:
    """Return samples where the first fundamental form is non-degenerate."""
    E, F, G = first_fundamental_form(Su, Sv)
    determinant = E * G - F * F
    return _isfinite(determinant) & (determinant > metric_eps)


def curvature_magnitude(result: CurvatureResult) -> ArrayLike:
    """Return ``max(|k1|, |k2|)`` at each sample."""
    if result.principal_1 is None or result.principal_2 is None:
        raise ValueError("principal curvatures are required")

    if _is_torch(result.principal_1):
        return torch.maximum(
            torch.abs(result.principal_1),
            torch.abs(result.principal_2),
        )

    return np.maximum(
        np.abs(result.principal_1),
        np.abs(result.principal_2),
    )


def curvedness(result: CurvatureResult) -> ArrayLike:
    """Compute Koenderink curvedness.

    C = sqrt((k1^2 + k2^2) / 2)

    Curvedness measures curvature magnitude without distinguishing convex,
    concave, or saddle behavior.
    """
    if result.principal_1 is None or result.principal_2 is None:
        raise ValueError("principal curvatures are required")

    value = (
        result.principal_1 * result.principal_1
        + result.principal_2 * result.principal_2
    ) / 2.0

    if _is_torch(value):
        return torch.sqrt(torch.clamp(value, min=0.0))
    return np.sqrt(np.maximum(value, 0.0))


def shape_index(
    result: CurvatureResult,
    *,
    eps: float = 1.0e-12,
) -> ArrayLike:
    """Compute Koenderink shape index in approximately [-1,1].

    The shape index describes local shape type independently of curvature
    magnitude. It is primarily diagnostic and not required by NSSR training.
    """
    if result.principal_1 is None or result.principal_2 is None:
        raise ValueError("principal curvatures are required")

    k1 = result.principal_1
    k2 = result.principal_2

    numerator = k1 + k2
    denominator = k1 - k2

    if _is_torch(k1):
        safe = torch.where(
            torch.abs(denominator) > eps,
            denominator,
            torch.ones_like(denominator),
        )
        raw = (2.0 / torch.pi) * torch.atan(numerator / safe)
        return torch.where(
            torch.abs(denominator) > eps,
            raw,
            torch.zeros_like(raw),
        )

    safe = np.where(
        np.abs(denominator) > eps,
        denominator,
        np.ones_like(denominator),
    )
    raw = (2.0 / np.pi) * np.arctan(numerator / safe)
    return np.where(
        np.abs(denominator) > eps,
        raw,
        np.zeros_like(raw),
    )


# ---------------------------------------------------------------------------
# Differentiable regularizers
# ---------------------------------------------------------------------------

def curvature_l2_loss(
    result: CurvatureResult,
    *,
    valid_mask: Optional[ArrayLike] = None,
    use_principal: bool = True,
) -> ArrayLike:
    """Quadratic curvature regularizer.

    By default this penalizes ``0.5*(k1^2+k2^2)``.  If principal curvatures
    are unavailable, or ``use_principal=False``, it penalizes ``H^2 + K^2``.

    This should normally be used with a small weight: the objective is to
    suppress pathological spikes, not flatten legitimate high-curvature
    geometry.
    """
    if (
        use_principal
        and result.principal_1 is not None
        and result.principal_2 is not None
    ):
        penalty = 0.5 * (
            result.principal_1 * result.principal_1
            + result.principal_2 * result.principal_2
        )
    else:
        penalty = (
            result.mean * result.mean
            + result.gaussian * result.gaussian
        )

    if valid_mask is None:
        return penalty.mean()

    if _is_torch(penalty):
        weights = valid_mask.to(dtype=penalty.dtype)
        return (penalty * weights).sum() / weights.sum().clamp_min(1.0)

    weights = np.asarray(valid_mask, dtype=penalty.dtype)
    return (penalty * weights).sum() / max(float(weights.sum()), 1.0)


def curvature_barrier_loss(
    result: CurvatureResult,
    *,
    max_abs_curvature: float,
    valid_mask: Optional[ArrayLike] = None,
    power: float = 2.0,
) -> ArrayLike:
    """Penalize only curvature magnitude above a chosen threshold.

    This is usually safer than globally minimizing curvature because normal
    object features are left unpenalized until the threshold is exceeded.
    """
    if max_abs_curvature <= 0:
        raise ValueError("max_abs_curvature must be positive")
    if power <= 0:
        raise ValueError("power must be positive")

    magnitude = curvature_magnitude(result)

    if _is_torch(magnitude):
        violation = torch.relu(magnitude - max_abs_curvature)
        penalty = violation.pow(power)
    else:
        violation = np.maximum(magnitude - max_abs_curvature, 0.0)
        penalty = violation ** power

    if valid_mask is None:
        return penalty.mean()

    if _is_torch(penalty):
        weights = valid_mask.to(dtype=penalty.dtype)
        return (penalty * weights).sum() / weights.sum().clamp_min(1.0)

    weights = np.asarray(valid_mask, dtype=penalty.dtype)
    return (penalty * weights).sum() / max(float(weights.sum()), 1.0)


__all__ = [
    "first_fundamental_form",
    "second_fundamental_form",
    "surface_normal",
    "curvatures_from_derivatives",
    "curvature_from_evaluation",
    "curvature_valid_mask",
    "curvature_magnitude",
    "curvedness",
    "shape_index",
    "curvature_l2_loss",
    "curvature_barrier_loss",
]
