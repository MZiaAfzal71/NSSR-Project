"""NSSR-V2 bicubic tensor-product Hermite surface evaluation.

This is the canonical Hermite interpolation kernel for NSSR-V2.

It evaluates bicubic Hermite patches, computes first/second derivatives,
and returns robust normals. Neural-network logic, tangent policy, cap
construction, Jacobian validation, curvature, and repair remain separate.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from .basis import (
    hermite_basis,
    hermite_basis_first,
    hermite_basis_second,
)
from .types import SurfaceEvaluation

ArrayLike = Union[np.ndarray, torch.Tensor]


def _is_torch(x: ArrayLike) -> bool:
    return isinstance(x, torch.Tensor)


def _contract(left: ArrayLike, grid: ArrayLike, right: ArrayLike) -> ArrayLike:
    if _is_torch(left):
        return torch.einsum("...i,...ijc,...j->...c", left, grid, right)
    return np.einsum("...i,...ijc,...j->...c", left, grid, right)


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


def _stack(values, *, axis: int) -> ArrayLike:
    first = values[0]
    if _is_torch(first):
        return torch.stack(tuple(values), dim=axis)
    return np.stack(tuple(values), axis=axis)


def _validate_backend(*arrays: ArrayLike) -> None:
    if not arrays:
        return
    first_torch = _is_torch(arrays[0])
    for array in arrays[1:]:
        if _is_torch(array) != first_torch:
            raise TypeError(
                "All Hermite inputs must use the same backend "
                "(all torch.Tensor or all numpy.ndarray)."
            )


def _validate_grid(grid: ArrayLike) -> None:
    if not isinstance(grid, (torch.Tensor, np.ndarray)):
        raise TypeError("grid must be torch.Tensor or numpy.ndarray")
    if grid.ndim < 3 or tuple(grid.shape[-3:]) != (4, 4, 3):
        raise ValueError(
            "Hermite grid must have shape (..., 4, 4, 3); "
            f"got {tuple(grid.shape)}"
        )


def hermite_control_grid(
    p00: ArrayLike,
    p01: ArrayLike,
    p10: ArrayLike,
    p11: ArrayLike,
    pu00: ArrayLike,
    pu01: ArrayLike,
    pu10: ArrayLike,
    pu11: ArrayLike,
    pv00: ArrayLike,
    pv01: ArrayLike,
    pv10: ArrayLike,
    pv11: ArrayLike,
    puv00: ArrayLike,
    puv01: ArrayLike,
    puv10: ArrayLike,
    puv11: ArrayLike,
) -> ArrayLike:
    """Assemble the canonical ``(...,4,4,3)`` Hermite control grid."""
    controls = (
        p00, p01, p10, p11,
        pu00, pu01, pu10, pu11,
        pv00, pv01, pv10, pv11,
        puv00, puv01, puv10, puv11,
    )
    _validate_backend(*controls)

    reference_shape = tuple(p00.shape)
    if not reference_shape or reference_shape[-1] != 3:
        raise ValueError(
            f"control points must have shape (...,3), got {reference_shape}"
        )

    for value in controls[1:]:
        if tuple(value.shape) != reference_shape:
            raise ValueError(
                "All Hermite geometric controls must have identical shapes; "
                f"expected {reference_shape}, got {tuple(value.shape)}"
            )

    rows = (
        (p00, p01, pv00, pv01),
        (p10, p11, pv10, pv11),
        (pu00, pu01, puv00, puv01),
        (pu10, pu11, puv10, puv11),
    )

    return _stack(
        [_stack(row, axis=-2) for row in rows],
        axis=-3,
    )


def evaluate_patch(
    grid: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    *,
    normalize_normals: bool = True,
    normal_eps: float = 1.0e-12,
) -> SurfaceEvaluation:
    """Evaluate a bicubic Hermite patch and all first/second derivatives."""
    _validate_grid(grid)
    _validate_backend(grid, u, v)

    Hu = hermite_basis(u)
    Hv = hermite_basis(v)
    dHu = hermite_basis_first(u)
    dHv = hermite_basis_first(v)
    ddHu = hermite_basis_second(u)
    ddHv = hermite_basis_second(v)

    xyz = _contract(Hu, grid, Hv)
    Su = _contract(dHu, grid, Hv)
    Sv = _contract(Hu, grid, dHv)
    Suu = _contract(ddHu, grid, Hv)
    Suv = _contract(dHu, grid, dHv)
    Svv = _contract(Hu, grid, ddHv)

    normals = _cross(Su, Sv)
    if normalize_normals:
        normals = _normalize(normals, eps=normal_eps)

    return SurfaceEvaluation(
        xyz=xyz,
        Su=Su,
        Sv=Sv,
        Suu=Suu,
        Suv=Suv,
        Svv=Svv,
        normals=normals,
    )


def evaluate_position(
    grid: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
) -> ArrayLike:
    """Evaluate only surface position for derivative-free hot paths."""
    _validate_grid(grid)
    _validate_backend(grid, u, v)
    return _contract(hermite_basis(u), grid, hermite_basis(v))


def evaluate_patch_from_derivatives(
    p00: ArrayLike,
    p01: ArrayLike,
    p10: ArrayLike,
    p11: ArrayLike,
    pu00: ArrayLike,
    pu01: ArrayLike,
    pu10: ArrayLike,
    pu11: ArrayLike,
    pv00: ArrayLike,
    pv01: ArrayLike,
    pv10: ArrayLike,
    pv11: ArrayLike,
    puv00: ArrayLike,
    puv01: ArrayLike,
    puv10: ArrayLike,
    puv11: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    *,
    normalize_normals: bool = True,
    normal_eps: float = 1.0e-12,
) -> SurfaceEvaluation:
    """Assemble and evaluate one Hermite patch from explicit derivatives."""
    grid = hermite_control_grid(
        p00, p01, p10, p11,
        pu00, pu01, pu10, pu11,
        pv00, pv01, pv10, pv11,
        puv00, puv01, puv10, puv11,
    )
    return evaluate_patch(
        grid,
        u,
        v,
        normalize_normals=normalize_normals,
        normal_eps=normal_eps,
    )


def evaluate_patch_grid(
    grid: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    *,
    normalize_normals: bool = True,
    normal_eps: float = 1.0e-12,
) -> SurfaceEvaluation:
    """Evaluate a patch on the tensor product of 1-D u and v coordinates."""
    if u.ndim != 1 or v.ndim != 1:
        raise ValueError(
            "evaluate_patch_grid expects one-dimensional u and v vectors"
        )

    uu = u.reshape(-1, 1)
    vv = v.reshape(1, -1)

    return evaluate_patch(
        grid,
        uu,
        vv,
        normalize_normals=normalize_normals,
        normal_eps=normal_eps,
    )


def evaluate_u_boundary(
    grid: ArrayLike,
    u_value: float,
    v: ArrayLike,
    *,
    normalize_normals: bool = True,
    normal_eps: float = 1.0e-12,
) -> SurfaceEvaluation:
    """Evaluate an entire u=constant boundary for C0/C1 tests."""
    if _is_torch(grid):
        u = torch.as_tensor(u_value, dtype=grid.dtype, device=grid.device)
    else:
        u = np.asarray(u_value, dtype=grid.dtype)

    return evaluate_patch(
        grid,
        u,
        v,
        normalize_normals=normalize_normals,
        normal_eps=normal_eps,
    )


def evaluate_v_boundary(
    grid: ArrayLike,
    v_value: float,
    u: ArrayLike,
    *,
    normalize_normals: bool = True,
    normal_eps: float = 1.0e-12,
) -> SurfaceEvaluation:
    """Evaluate an entire v=constant boundary for C0/C1 tests."""
    if _is_torch(grid):
        v = torch.as_tensor(v_value, dtype=grid.dtype, device=grid.device)
    else:
        v = np.asarray(v_value, dtype=grid.dtype)

    return evaluate_patch(
        grid,
        u,
        v,
        normalize_normals=normalize_normals,
        normal_eps=normal_eps,
    )


__all__ = [
    "hermite_control_grid",
    "evaluate_patch",
    "evaluate_position",
    "evaluate_patch_from_derivatives",
    "evaluate_patch_grid",
    "evaluate_u_boundary",
    "evaluate_v_boundary",
]
