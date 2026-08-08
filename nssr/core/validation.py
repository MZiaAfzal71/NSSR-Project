"""NSSR-V2 geometric validation and quality aggregation.

This module combines the independent diagnostics produced by:

- ``hermite.py``      -> SurfaceEvaluation
- ``jacobian.py``     -> JacobianEvaluation
- ``curvature.py``    -> CurvatureResult
- ``tangent.py``      -> tangent magnitudes / energies

into a single surface-quality decision.

The module intentionally does NOT repair geometry.  Validation should be
deterministic and side-effect free.  A later ``repair.py`` module can consume
the masks/metrics produced here and decide how to modify tangents or patches.

Design principle
----------------
Validation is layered:

1. numerical validity
2. parameterization validity
3. orientation validity
4. curvature validity
5. tangent-field validity
6. aggregate object/patch decision

This is preferable to a single opaque scalar threshold because it preserves
the reason a surface failed.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

from .curvature import (
    CurvatureResult,
    curvature_magnitude,
    curvature_valid_mask,
)
from .jacobian import JacobianEvaluation
from .tangent import tangent_energy, tangent_magnitude
from .types import SurfaceEvaluation, ValidationResult


ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _is_torch(x: ArrayLike) -> bool:
    return isinstance(x, torch.Tensor)


def _isfinite(x: ArrayLike) -> ArrayLike:
    return torch.isfinite(x) if _is_torch(x) else np.isfinite(x)


def _abs(x: ArrayLike) -> ArrayLike:
    return torch.abs(x) if _is_torch(x) else np.abs(x)


def _mean_bool(mask: ArrayLike) -> float:
    if _is_torch(mask):
        return float(mask.to(torch.float32).mean().detach().cpu().item())
    return float(np.asarray(mask, dtype=np.float64).mean())


def _sum_bool(mask: ArrayLike) -> int:
    if _is_torch(mask):
        return int(mask.sum().detach().cpu().item())
    return int(np.asarray(mask).sum())


def _max_value(x: ArrayLike) -> float:
    if _is_torch(x):
        return float(x.detach().amax().cpu().item())
    return float(np.max(x))


def _mean_value(x: ArrayLike) -> float:
    if _is_torch(x):
        return float(x.detach().mean().cpu().item())
    return float(np.mean(x))


def _all_finite_last(x: ArrayLike) -> ArrayLike:
    finite = _isfinite(x)
    if finite.ndim == 0:
        return finite
    if x.ndim > 0 and x.shape[-1] in (2, 3):
        return finite.all(dim=-1) if _is_torch(finite) else finite.all(axis=-1)
    return finite


def _logical_and_all(*masks: ArrayLike) -> ArrayLike:
    if not masks:
        raise ValueError("at least one mask is required")
    out = masks[0]
    for mask in masks[1:]:
        out = out & mask
    return out


# ---------------------------------------------------------------------------
# Per-component validation
# ---------------------------------------------------------------------------

def surface_finite_mask(evaluation: SurfaceEvaluation) -> ArrayLike:
    """Return samples where all surface quantities are finite."""
    masks = [
        _all_finite_last(evaluation.xyz),
        _all_finite_last(evaluation.Su),
        _all_finite_last(evaluation.Sv),
    ]

    if evaluation.Suu is not None:
        masks.append(_all_finite_last(evaluation.Suu))
    if evaluation.Suv is not None:
        masks.append(_all_finite_last(evaluation.Suv))
    if evaluation.Svv is not None:
        masks.append(_all_finite_last(evaluation.Svv))
    if evaluation.normals is not None:
        masks.append(_all_finite_last(evaluation.normals))

    return _logical_and_all(*masks)


def jacobian_failure_mask(
    jacobian: JacobianEvaluation,
    *,
    include_near_flip: bool = False,
) -> ArrayLike:
    """Return samples invalid because of fold/degeneracy."""
    failure = jacobian.degenerate_mask | jacobian.flipped_mask
    if include_near_flip:
        failure = failure | jacobian.near_flip_mask
    return failure


def curvature_failure_mask(
    curvature: CurvatureResult,
    evaluation: SurfaceEvaluation,
    *,
    max_abs_curvature: float,
    metric_eps: float = 1.0e-12,
) -> ArrayLike:
    """Return invalid curvature samples.

    A sample fails if the surface metric is degenerate, curvature is non-finite,
    or either principal curvature exceeds ``max_abs_curvature``.
    """
    if max_abs_curvature <= 0:
        raise ValueError("max_abs_curvature must be positive")

    metric_valid = curvature_valid_mask(
        evaluation.Su,
        evaluation.Sv,
        metric_eps=metric_eps,
    )

    magnitude = curvature_magnitude(curvature)
    finite = _isfinite(magnitude)

    return (~metric_valid) | (~finite) | (magnitude > max_abs_curvature)


def tangent_failure_mask(
    tangents: ArrayLike,
    *,
    max_magnitude: Optional[float] = None,
    max_energy: Optional[float] = None,
) -> ArrayLike:
    """Return per-tangent validity mask.

    The result has shape ``(...,N)`` for tangents shaped ``(...,N,D)``.
    """
    mag = tangent_magnitude(tangents)
    energy = tangent_energy(tangents)

    fail = ~_isfinite(mag) | ~_isfinite(energy)

    if max_magnitude is not None:
        if max_magnitude <= 0:
            raise ValueError("max_magnitude must be positive")
        fail = fail | (mag > max_magnitude)

    if max_energy is not None:
        if max_energy <= 0:
            raise ValueError("max_energy must be positive")
        fail = fail | (energy > max_energy)

    return fail


# ---------------------------------------------------------------------------
# Aggregate validation
# ---------------------------------------------------------------------------

def validate_surface(
    evaluation: SurfaceEvaluation,
    jacobian: JacobianEvaluation,
    curvature: Optional[CurvatureResult] = None,
    tangents: Optional[ArrayLike] = None,
    *,
    max_abs_curvature: float = 100.0,
    max_tangent_magnitude: Optional[float] = None,
    max_tangent_energy: Optional[float] = None,
    include_near_flip: bool = False,
    metric_eps: float = 1.0e-12,
    max_invalid_fraction: float = 0.0,
) -> ValidationResult:
    """Validate a reconstructed surface.

    Parameters
    ----------
    evaluation
        Hermite surface evaluation.
    jacobian
        Orientation and area-scaling diagnostics.
    curvature
        Optional curvature result. If omitted, curvature violations are zero.
    tangents
        Optional tangent field. If omitted, tangent violations are zero.
    max_abs_curvature
        Principal-curvature magnitude threshold.
    max_tangent_magnitude
        Optional maximum tangent norm.
    max_tangent_energy
        Optional maximum squared tangent norm.
    include_near_flip
        Treat near-zero signed Jacobians as failures.
    metric_eps
        Metric degeneracy threshold for curvature.
    max_invalid_fraction
        Surface can tolerate this fraction of invalid samples and still be
        marked globally valid. Default 0.0 means no invalid samples.

    Returns
    -------
    ValidationResult
        Uses the dataclass defined in ``nssr.core.types``.
    """
    if not (0.0 <= max_invalid_fraction <= 1.0):
        raise ValueError("max_invalid_fraction must lie in [0,1]")

    finite_mask = surface_finite_mask(evaluation)

    jac_fail = jacobian_failure_mask(
        jacobian,
        include_near_flip=include_near_flip,
    )

    # Jacobian/evaluation masks are expected to describe the same sampled grid.
    if finite_mask.shape != jac_fail.shape:
        raise ValueError(
            "SurfaceEvaluation and JacobianEvaluation sample shapes differ: "
            f"{finite_mask.shape} vs {jac_fail.shape}"
        )

    total_failure = (~finite_mask) | jac_fail

    negative_jacobian = _sum_bool(jacobian.flipped_mask)
    degenerate = _sum_bool(jacobian.degenerate_mask)
    fold_count = negative_jacobian

    curvature_violations = 0
    if curvature is not None:
        curv_fail = curvature_failure_mask(
            curvature,
            evaluation,
            max_abs_curvature=max_abs_curvature,
            metric_eps=metric_eps,
        )
        if curv_fail.shape != total_failure.shape:
            raise ValueError(
                "Curvature sample shape differs from surface sample shape: "
                f"{curv_fail.shape} vs {total_failure.shape}"
            )
        curvature_violations = _sum_bool(curv_fail)
        total_failure = total_failure | curv_fail

    tangent_violations = 0
    if tangents is not None:
        tan_fail = tangent_failure_mask(
            tangents,
            max_magnitude=max_tangent_magnitude,
            max_energy=max_tangent_energy,
        )
        tangent_violations = _sum_bool(tan_fail)

    invalid_fraction = _mean_bool(total_failure)
    valid = invalid_fraction <= max_invalid_fraction

    # Tangent failures are global field failures rather than surface-grid
    # samples, so they participate explicitly in the object-level decision.
    if tangent_violations > 0:
        valid = False

    return ValidationResult(
        valid=bool(valid),
        fold_count=fold_count,
        negative_jacobian=negative_jacobian,
        degenerate=degenerate,
        curvature_violations=curvature_violations,
        tangent_violations=tangent_violations,
    )


# ---------------------------------------------------------------------------
# Detailed metrics for logging / diagnostics
# ---------------------------------------------------------------------------

def validation_metrics(
    evaluation: SurfaceEvaluation,
    jacobian: JacobianEvaluation,
    curvature: Optional[CurvatureResult] = None,
    tangents: Optional[ArrayLike] = None,
) -> dict[str, float]:
    """Return scalar diagnostics suitable for logs and experiment tracking."""
    finite = surface_finite_mask(evaluation)

    metrics: dict[str, float] = {
        "finite_surface_fraction": _mean_bool(finite),
        "negative_jacobian_fraction": float(
            jacobian.negative_fraction.detach().cpu().item()
            if _is_torch(jacobian.negative_fraction)
            else jacobian.negative_fraction
        ),
        "degenerate_jacobian_fraction": float(
            jacobian.degenerate_fraction.detach().cpu().item()
            if _is_torch(jacobian.degenerate_fraction)
            else jacobian.degenerate_fraction
        ),
    }

    if curvature is not None:
        mag = curvature_magnitude(curvature)
        finite_curv = _isfinite(mag)

        if _is_torch(mag):
            finite_values = mag[finite_curv]
            if finite_values.numel() > 0:
                metrics["mean_abs_principal_curvature"] = float(
                    finite_values.mean().detach().cpu().item()
                )
                metrics["max_abs_principal_curvature"] = float(
                    finite_values.amax().detach().cpu().item()
                )
            else:
                metrics["mean_abs_principal_curvature"] = float("nan")
                metrics["max_abs_principal_curvature"] = float("nan")
        else:
            finite_values = mag[finite_curv]
            if finite_values.size > 0:
                metrics["mean_abs_principal_curvature"] = float(
                    finite_values.mean()
                )
                metrics["max_abs_principal_curvature"] = float(
                    finite_values.max()
                )
            else:
                metrics["mean_abs_principal_curvature"] = float("nan")
                metrics["max_abs_principal_curvature"] = float("nan")

    if tangents is not None:
        mag = tangent_magnitude(tangents)
        energy = tangent_energy(tangents)

        metrics["mean_tangent_magnitude"] = _mean_value(mag)
        metrics["max_tangent_magnitude"] = _max_value(mag)
        metrics["mean_tangent_energy"] = _mean_value(energy)
        metrics["max_tangent_energy"] = _max_value(energy)

    return metrics


def patch_validity_mask(
    evaluation: SurfaceEvaluation,
    jacobian: JacobianEvaluation,
    curvature: Optional[CurvatureResult] = None,
    *,
    max_abs_curvature: float = 100.0,
    include_near_flip: bool = False,
    metric_eps: float = 1.0e-12,
) -> ArrayLike:
    """Return a per-sample boolean validity mask.

    This is useful for visualization and later geometry repair.
    """
    finite = surface_finite_mask(evaluation)
    failure = jacobian_failure_mask(
        jacobian,
        include_near_flip=include_near_flip,
    )

    valid = finite & (~failure)

    if curvature is not None:
        curv_fail = curvature_failure_mask(
            curvature,
            evaluation,
            max_abs_curvature=max_abs_curvature,
            metric_eps=metric_eps,
        )
        valid = valid & (~curv_fail)

    return valid


__all__ = [
    "surface_finite_mask",
    "jacobian_failure_mask",
    "curvature_failure_mask",
    "tangent_failure_mask",
    "validate_surface",
    "validation_metrics",
    "patch_validity_mask",
]
