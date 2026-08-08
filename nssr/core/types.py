"""
===============================================================================

NSSR-V2

Strongly typed geometry data structures.

The original NSSR implementation passes dictionaries of tensors between
modules. While compact, this makes the code difficult to understand,
debug and extend.

NSSR-V2 replaces these dictionaries with typed dataclasses.

===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


Tensor = torch.Tensor


# ============================================================================
# Network Parameters
# ============================================================================

@dataclass(slots=True)
class GeometryParameters:
    """
    Network predicted geometric parameters.

    All tensors are expected to have batch dimension first.
    """

    alpha: Tensor
    beta: Tensor
    tau: Tensor

    base_scale: Tensor
    crown_scale: Tensor

    base_height: Tensor
    crown_height: Tensor


# ============================================================================
# Tangent Field
# ============================================================================

@dataclass(slots=True)
class TangentField:
    """
    Tangent vectors for every contour.
    """

    radial: Tensor
    axial: Tensor

    magnitude: Optional[Tensor] = None

    energy: Optional[Tensor] = None


# ============================================================================
# Surface Patch
# ============================================================================

@dataclass(slots=True)
class SurfacePatch:
    """
    Sampled Hermite surface patch.
    """

    xyz: Tensor

    normals: Optional[Tensor] = None

    uv: Optional[Tensor] = None

    valid_mask: Optional[Tensor] = None


# ============================================================================
# Surface Derivatives
# ============================================================================

@dataclass(slots=True)
class SurfaceDerivatives:
    """
    First and second derivatives of a surface.
    """

    Su: Tensor

    Sv: Tensor

    Suu: Optional[Tensor] = None

    Suv: Optional[Tensor] = None

    Svv: Optional[Tensor] = None


# ============================================================================
# Jacobian
# ============================================================================

@dataclass(slots=True)
class JacobianResult:
    """
    Jacobian statistics.
    """

    determinant: Tensor

    orientation: Tensor

    valid_mask: Tensor

    negative_fraction: Tensor


# ============================================================================
# Curvature
# ============================================================================

@dataclass(slots=True)
class CurvatureResult:
    """
    Differential geometry measures.
    """

    mean: Tensor

    gaussian: Tensor

    principal_1: Optional[Tensor] = None

    principal_2: Optional[Tensor] = None


# ============================================================================
# Validation
# ============================================================================

@dataclass(slots=True)
class ValidationResult:
    """
    Surface quality report.
    """

    valid: bool

    fold_count: int

    negative_jacobian: int

    degenerate: int

    curvature_violations: int

    tangent_violations: int


# ============================================================================
# Geometry Statistics
# ============================================================================

@dataclass(slots=True)
class GeometryStatistics:
    """
    Global statistics for one reconstructed object.
    """

    mean_curvature: float

    max_curvature: float

    tangent_energy: float

    negative_jacobian_ratio: float

    valid_surface_ratio: float

    confidence: float


# ============================================================================
# Complete Geometry Output
# ============================================================================

@dataclass(slots=True)
class GeometryOutput:
    """
    Output of the geometry engine.

    This is the object passed to the loss function,
    reconstruction pipeline and diagnostics.
    """

    surface: SurfacePatch

    tangents: TangentField

    derivatives: SurfaceDerivatives

    jacobian: Optional[JacobianResult] = None

    curvature: Optional[CurvatureResult] = None

    validation: Optional[ValidationResult] = None

    statistics: Optional[GeometryStatistics] = None


@dataclass(slots=True)
class SurfaceEvaluation:

    xyz: Tensor

    Su: Tensor

    Sv: Tensor

    Suu: Tensor

    Suv: Tensor

    Svv: Tensor

    normals: Tensor
    