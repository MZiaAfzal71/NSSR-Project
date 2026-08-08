"""
===============================================================================

NSSR-V2
--------

Hermite basis functions.

This module provides a single canonical implementation of the cubic Hermite
basis used throughout NSSR-V2.

Every surface evaluation, Jacobian computation, curvature computation,
training loss and validation routine must obtain Hermite basis functions from
this module.

This avoids mathematical duplication between geometry.py,
geometry_np.py and the reference implementation.

Author
------
NSSR-V2 Development

===============================================================================
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

ArrayLike = Union[torch.Tensor, np.ndarray]


# =============================================================================
# Backend helpers
# =============================================================================

def _stack(values, ref: ArrayLike) -> ArrayLike:
    """
    Stack values preserving backend.

    Parameters
    ----------
    values
        Sequence of arrays.

    ref
        Reference array used to determine backend.

    Returns
    -------
    torch.Tensor or np.ndarray
    """
    if isinstance(ref, torch.Tensor):
        return torch.stack(values, dim=-1)

    return np.stack(values, axis=-1)


# =============================================================================
# Hermite basis
# =============================================================================

def hermite_basis(u: ArrayLike) -> ArrayLike:
    """
    Cubic Hermite basis.

    Returns

        [h00 h01 h10 h11]

    Shape

        (...,4)
    """

    u2 = u * u
    u3 = u2 * u

    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h01 = -2.0 * u3 + 3.0 * u2
    h10 = u3 - 2.0 * u2 + u
    h11 = u3 - u2

    return _stack((h00, h01, h10, h11), u)


# =============================================================================
# First derivatives
# =============================================================================

def hermite_basis_first(u: ArrayLike) -> ArrayLike:
    """
    First derivative of Hermite basis.
    """

    u2 = u * u

    h00 = 6.0 * u2 - 6.0 * u
    h01 = -6.0 * u2 + 6.0 * u
    h10 = 3.0 * u2 - 4.0 * u + 1.0
    h11 = 3.0 * u2 - 2.0 * u

    return _stack((h00, h01, h10, h11), u)


# =============================================================================
# Second derivatives
# =============================================================================

def hermite_basis_second(u: ArrayLike) -> ArrayLike:
    """
    Second derivative of Hermite basis.
    """

    h00 = 12.0 * u - 6.0
    h01 = -12.0 * u + 6.0
    h10 = 6.0 * u - 4.0
    h11 = 6.0 * u - 2.0

    return _stack((h00, h01, h10, h11), u)


# =============================================================================
# Utilities
# =============================================================================

def basis_all(u: ArrayLike):
    """
    Compute basis and its derivatives.

    Returns
    -------
    H
        Hermite basis.

    dH
        First derivative.

    ddH
        Second derivative.
    """

    H = hermite_basis(u)
    dH = hermite_basis_first(u)
    ddH = hermite_basis_second(u)

    return H, dH, ddH