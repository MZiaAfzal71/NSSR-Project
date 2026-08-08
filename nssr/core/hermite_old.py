"""NSSR-V2 bicubic tensor-product Hermite surface kernel.

This module centralizes patch evaluation for both NumPy and PyTorch.  It
returns the surface and first/second parameter derivatives needed by later
Jacobian and curvature modules.

The basis convention matches the existing NSSR implementation:
L0=1-3u^2+2u^3, L1=3u^2-2u^3,
H0=u-2u^2+u^3, H1=-u^2+u^3.

A control grid has shape (...,4,4,3):
[[P00,P01,Pv00,Pv01],
 [P10,P11,Pv10,Pv11],
 [Pu00,Pu01,Puv00,Puv01],
 [Pu10,Pu11,Puv10,Puv11]]
where u,v are dimensionless patch coordinates.
"""

from __future__ import annotations
from typing import NamedTuple, Union
import numpy as np
import torch
from .basis import hermite_basis, hermite_basis_first, hermite_basis_second

ArrayLike = Union[np.ndarray, torch.Tensor]


class SurfaceEvaluation(NamedTuple):
    """Local differential description of a Hermite patch."""
    xyz: ArrayLike
    Su: ArrayLike
    Sv: ArrayLike
    Suu: ArrayLike
    Suv: ArrayLike
    Svv: ArrayLike
    normal: ArrayLike


def _basis(u, order):
    if order == 0:
        return hermite_basis(u)
    if order == 1:
        return hermite_basis_first(u)
    if order == 2:
        return hermite_basis_second(u)
    raise ValueError("order must be 0, 1, or 2")


def _contract(left, grid, right):
    if isinstance(left, torch.Tensor):
        return torch.einsum("...i,...ijc,...j->...c", left, grid, right)
    return np.einsum("...i,...ijc,...j->...c", left, grid, right)


def _cross(a, b):
    if isinstance(a, torch.Tensor):
        return torch.linalg.cross(a, b, dim=-1)
    return np.cross(a, b, axis=-1)


def _normalize(v, eps=1e-12):
    if isinstance(v, torch.Tensor):
        n = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        return v / torch.clamp(n, min=eps)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def evaluate_patch(grid, u, v, *, return_derivatives=True,
                   normalize_normal=True, normal_eps=1e-12):
    """Evaluate a bicubic Hermite patch.

    Parameters
    ----------
    grid : ndarray or Tensor
        Shape (...,4,4,3).
    u, v : scalar or array-like
        Dimensionless patch coordinates. Values outside [0,1] are allowed
        for diagnostics/extrapolation and are not silently clipped.
    return_derivatives : bool
        If False, return only xyz.
    normalize_normal : bool
        Return unit normals when True.

    Returns
    -------
    SurfaceEvaluation or array
        xyz, Su, Sv, Suu, Suv, Svv and normal.
    """
    shape = tuple(grid.shape)
    if len(shape) < 3 or shape[-3:] != (4, 4, 3):
        raise ValueError(f"grid must have shape (...,4,4,3), got {shape}")

    H, dH, ddH = _basis(u, 0), _basis(u, 1), _basis(u, 2)
    K, dK, ddK = _basis(v, 0), _basis(v, 1), _basis(v, 2)

    xyz = _contract(H, grid, K)
    if not return_derivatives:
        return xyz

    Su = _contract(dH, grid, K)
    Sv = _contract(H, grid, dK)
    Suu = _contract(ddH, grid, K)
    Suv = _contract(dH, grid, dK)
    Svv = _contract(H, grid, ddK)

    n = _cross(Su, Sv)
    if normalize_normal:
        n = _normalize(n, normal_eps)

    return SurfaceEvaluation(xyz, Su, Sv, Suu, Suv, Svv, n)


def evaluate_patch_from_derivatives(
    p00, p01, p10, p11,
    pu00, pu01, pu10, pu11,
    pv00, pv01, pv10, pv11,
    puv00, puv01, puv10, puv11,
    u, v, *, return_derivatives=True, normalize_normal=True,
    normal_eps=1e-12):
    """Evaluate a patch from its 16 Hermite geometric quantities."""
    rows = (
        (p00, p01, pv00, pv01),
        (p10, p11, pv10, pv11),
        (pu00, pu01, puv00, puv01),
        (pu10, pu11, puv10, puv11),
    )
    if isinstance(p00, torch.Tensor):
        grid = torch.stack([torch.stack(row, dim=-2) for row in rows], dim=-3)
    elif isinstance(p00, np.ndarray):
        grid = np.stack([np.stack(row, axis=-2) for row in rows], axis=-3)
    else:
        raise TypeError("control data must be NumPy arrays or PyTorch tensors")
    return evaluate_patch(
        grid, u, v, return_derivatives=return_derivatives,
        normalize_normal=normalize_normal, normal_eps=normal_eps)


def evaluate_patch_grid(grid, u, v, *, normalize_normal=True,
                        normal_eps=1e-12):
    """Evaluate one patch on the tensor-product grid defined by u and v."""
    uu = u.reshape(-1, 1)
    vv = v.reshape(1, -1)
    return evaluate_patch(
        grid, uu, vv, return_derivatives=True,
        normalize_normal=normalize_normal, normal_eps=normal_eps)


__all__ = [
    "SurfaceEvaluation",
    "evaluate_patch",
    "evaluate_patch_from_derivatives",
    "evaluate_patch_grid",
]
