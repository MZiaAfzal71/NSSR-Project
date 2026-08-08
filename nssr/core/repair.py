"""NSSR-V2 conservative geometry repair.

This module repairs invalid Hermite geometry by modifying the *cause* of many
folds -- excessive tangent corrections -- rather than moving sampled surface
points after evaluation.

Primary strategy
----------------
Given:
- a learned/predicted tangent field,
- a safer reference tangent field (usually classical or monotone-limited),
- a callable that rebuilds/evaluates the surface from tangents,
- a callable that validates that rebuilt surface,

we search for the largest blend factor alpha in [0,1] such that

    T_repaired = T_reference + alpha * (T_predicted - T_reference)

remains geometrically valid.

alpha = 1 keeps the learned geometry unchanged.
alpha = 0 falls back completely to the reference geometry.

This preserves as much learned correction as possible while enforcing
geometric validity.

The module is deliberately policy-light.  It does not know how NSSR constructs
caps or how a neural network predicts tangent corrections.  Callers provide
the rebuild and validation functions.

The PyTorch path remains differentiable with respect to the final blended
tangent expression, but the binary-search decision itself is a discrete
inference-time operation and should normally be used outside backpropagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union, Any

import numpy as np
import torch

from .types import ValidationResult

ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass(slots=True)
class RepairResult:
    """Result of tangent-space geometry repair."""

    tangents: ArrayLike
    alpha: float
    repaired: bool
    initial_valid: bool
    final_valid: bool
    iterations: int
    initial_report: ValidationResult
    final_report: ValidationResult
    payload: Any = None


def _is_torch(x: ArrayLike) -> bool:
    return isinstance(x, torch.Tensor)


def _same_backend(a: ArrayLike, b: ArrayLike) -> bool:
    return _is_torch(a) == _is_torch(b)


def _blend(
    reference: ArrayLike,
    predicted: ArrayLike,
    alpha: float,
) -> ArrayLike:
    """Blend from reference (0) to predicted (1)."""
    return reference + alpha * (predicted - reference)


def _validate_tangent_inputs(
    predicted: ArrayLike,
    reference: ArrayLike,
) -> None:
    if not isinstance(predicted, (np.ndarray, torch.Tensor)):
        raise TypeError("predicted tangents must be NumPy or PyTorch")
    if not isinstance(reference, (np.ndarray, torch.Tensor)):
        raise TypeError("reference tangents must be NumPy or PyTorch")
    if not _same_backend(predicted, reference):
        raise TypeError("predicted and reference tangents must use same backend")
    if predicted.shape != reference.shape:
        raise ValueError(
            "predicted/reference tangent shapes differ: "
            f"{predicted.shape} vs {reference.shape}"
        )


def _report_valid(report: ValidationResult) -> bool:
    return bool(report.valid)


def repair_by_tangent_blend(
    predicted_tangents: ArrayLike,
    reference_tangents: ArrayLike,
    rebuild_fn: Callable[[ArrayLike], Any],
    validate_fn: Callable[[Any], ValidationResult],
    *,
    iterations: int = 20,
    min_alpha: float = 0.0,
    require_reference_valid: bool = True,
) -> RepairResult:
    """Project a tangent field onto a valid geometry region.

    Parameters
    ----------
    predicted_tangents
        Learned/current tangent field.
    reference_tangents
        Safe fallback tangent field, usually classical or monotone-limited.
    rebuild_fn
        Function accepting a tangent field and returning whatever object is
        needed by ``validate_fn`` (for example SurfaceEvaluation plus derived
        Jacobian/curvature data).
    validate_fn
        Function accepting the rebuild payload and returning ValidationResult.
    iterations
        Binary-search iterations.
    min_alpha
        Lower bound on retained learned correction. If the valid solution is
        below this threshold, the reference is returned instead.
    require_reference_valid
        If True, raise if alpha=0 is itself invalid.

    Returns
    -------
    RepairResult
        Largest valid blend found by binary search.

    Notes
    -----
    Validity as a function of alpha is assumed to be locally monotone enough
    for projection: reducing a problematic correction should not generally
    introduce a new fold.  This is the same practical assumption used by
    safety projections in many geometric optimization pipelines.
    """
    _validate_tangent_inputs(predicted_tangents, reference_tangents)

    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not (0.0 <= min_alpha <= 1.0):
        raise ValueError("min_alpha must lie in [0,1]")

    # Check learned/current geometry first.
    initial_payload = rebuild_fn(predicted_tangents)
    initial_report = validate_fn(initial_payload)

    if _report_valid(initial_report):
        return RepairResult(
            tangents=predicted_tangents,
            alpha=1.0,
            repaired=False,
            initial_valid=True,
            final_valid=True,
            iterations=0,
            initial_report=initial_report,
            final_report=initial_report,
            payload=initial_payload,
        )

    # Check reference geometry.
    reference_payload = rebuild_fn(reference_tangents)
    reference_report = validate_fn(reference_payload)

    if require_reference_valid and not _report_valid(reference_report):
        raise RuntimeError(
            "Reference tangent field is invalid; tangent blending cannot "
            "guarantee a safe repair. Fix the reference construction first."
        )

    if not _report_valid(reference_report):
        return RepairResult(
            tangents=reference_tangents,
            alpha=0.0,
            repaired=True,
            initial_valid=False,
            final_valid=False,
            iterations=0,
            initial_report=initial_report,
            final_report=reference_report,
            payload=reference_payload,
        )

    lo = 0.0
    hi = 1.0

    best_alpha = 0.0
    best_tangents = reference_tangents
    best_payload = reference_payload
    best_report = reference_report

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        candidate = _blend(reference_tangents, predicted_tangents, mid)
        payload = rebuild_fn(candidate)
        report = validate_fn(payload)

        if _report_valid(report):
            best_alpha = mid
            best_tangents = candidate
            best_payload = payload
            best_report = report
            lo = mid
        else:
            hi = mid

    if best_alpha < min_alpha:
        best_alpha = 0.0
        best_tangents = reference_tangents
        best_payload = reference_payload
        best_report = reference_report

    return RepairResult(
        tangents=best_tangents,
        alpha=float(best_alpha),
        repaired=True,
        initial_valid=False,
        final_valid=_report_valid(best_report),
        iterations=iterations,
        initial_report=initial_report,
        final_report=best_report,
        payload=best_payload,
    )


def damp_tangents(
    tangents: ArrayLike,
    factor: Union[float, ArrayLike],
) -> ArrayLike:
    """Apply scalar or per-node tangent damping.

    ``factor`` should normally lie in [0,1]. This helper clips values for
    safety rather than amplifying tangents accidentally.
    """
    if isinstance(factor, (float, int)):
        f = float(factor)
        f = max(0.0, min(1.0, f))
        return tangents * f

    if _is_torch(tangents):
        if not isinstance(factor, torch.Tensor):
            raise TypeError("factor must use same backend as tangents")
        f = torch.clamp(factor, 0.0, 1.0)
        while f.ndim < tangents.ndim:
            f = f.unsqueeze(-1)
        return tangents * f

    if not isinstance(factor, np.ndarray):
        raise TypeError("factor must use same backend as tangents")
    f = np.clip(factor, 0.0, 1.0)
    while f.ndim < tangents.ndim:
        f = np.expand_dims(f, axis=-1)
    return tangents * f


def violation_damping_mask(
    tangent_count: int,
    *,
    failing_indices: Optional[ArrayLike] = None,
    radius: int = 1,
    center_factor: float = 0.25,
    neighbor_factor: float = 0.6,
    backend_reference: Optional[ArrayLike] = None,
) -> ArrayLike:
    """Construct a 1-D local damping mask around problematic tangent nodes.

    This is useful when validation can map a failed patch back to nearby
    contour/tangent rows.  The center is strongly damped while neighboring
    nodes receive milder damping.

    Parameters
    ----------
    tangent_count
        Number of tangent nodes.
    failing_indices
        Integer indices of problematic nodes.
    radius
        Number of neighboring nodes on either side.
    center_factor
        Damping at the failing node.
    neighbor_factor
        Damping for immediate/near neighbors.
    backend_reference
        Optional tensor/array used to choose dtype/device/backend.

    Returns
    -------
    array
        Shape ``(tangent_count,)`` with values in [0,1].
    """
    if tangent_count <= 0:
        raise ValueError("tangent_count must be positive")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    center_factor = max(0.0, min(1.0, float(center_factor)))
    neighbor_factor = max(0.0, min(1.0, float(neighbor_factor)))

    if backend_reference is not None and _is_torch(backend_reference):
        mask = torch.ones(
            tangent_count,
            dtype=backend_reference.dtype,
            device=backend_reference.device,
        )
    else:
        dtype = (
            backend_reference.dtype
            if isinstance(backend_reference, np.ndarray)
            else np.float64
        )
        mask = np.ones(tangent_count, dtype=dtype)

    if failing_indices is None:
        return mask

    if isinstance(failing_indices, torch.Tensor):
        indices = [int(x) for x in failing_indices.detach().cpu().reshape(-1)]
    else:
        indices = [int(x) for x in np.asarray(failing_indices).reshape(-1)]

    for center in indices:
        if center < 0 or center >= tangent_count:
            continue

        mask[center] = min(float(mask[center]), center_factor)

        for d in range(1, radius + 1):
            # Linearly relax damping toward 1 farther from the center.
            t = d / (radius + 1)
            factor = neighbor_factor + t * (1.0 - neighbor_factor)

            left = center - d
            right = center + d

            if left >= 0:
                mask[left] = min(float(mask[left]), factor)
            if right < tangent_count:
                mask[right] = min(float(mask[right]), factor)

    return mask


def iterative_local_repair(
    tangents: ArrayLike,
    rebuild_fn: Callable[[ArrayLike], Any],
    validate_fn: Callable[[Any], ValidationResult],
    locate_failures_fn: Callable[[Any], ArrayLike],
    *,
    max_rounds: int = 6,
    radius: int = 1,
    center_factor: float = 0.25,
    neighbor_factor: float = 0.6,
) -> RepairResult:
    """Iteratively damp only tangent nodes associated with failed regions.

    ``locate_failures_fn(payload)`` must return indices in the tangent-node
    direction.  This keeps local repair independent of a specific NSSR patch
    indexing convention.

    Compared with global blending, this preserves more learned geometry when
    only a crown/shoulder region is problematic.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")

    current = tangents
    initial_payload = rebuild_fn(current)
    initial_report = validate_fn(initial_payload)

    if _report_valid(initial_report):
        return RepairResult(
            tangents=current,
            alpha=1.0,
            repaired=False,
            initial_valid=True,
            final_valid=True,
            iterations=0,
            initial_report=initial_report,
            final_report=initial_report,
            payload=initial_payload,
        )

    payload = initial_payload
    report = initial_report

    for round_index in range(1, max_rounds + 1):
        failing = locate_failures_fn(payload)

        mask = violation_damping_mask(
            current.shape[-2],
            failing_indices=failing,
            radius=radius,
            center_factor=center_factor,
            neighbor_factor=neighbor_factor,
            backend_reference=current,
        )

        current = damp_tangents(current, mask)
        payload = rebuild_fn(current)
        report = validate_fn(payload)

        if _report_valid(report):
            return RepairResult(
                tangents=current,
                alpha=float("nan"),
                repaired=True,
                initial_valid=False,
                final_valid=True,
                iterations=round_index,
                initial_report=initial_report,
                final_report=report,
                payload=payload,
            )

    return RepairResult(
        tangents=current,
        alpha=float("nan"),
        repaired=True,
        initial_valid=False,
        final_valid=_report_valid(report),
        iterations=max_rounds,
        initial_report=initial_report,
        final_report=report,
        payload=payload,
    )


__all__ = [
    "RepairResult",
    "repair_by_tangent_blend",
    "damp_tangents",
    "violation_damping_mask",
    "iterative_local_repair",
]
