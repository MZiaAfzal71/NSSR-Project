"""
===============================================================================

NSSR-V2
--------

Core mathematical constants used throughout the geometry engine.

This module centralizes all numerical tolerances, Hermite basis matrices,
default geometric limits, and configuration values so that the entire
geometry engine behaves consistently.

Author:
    NSSR-V2 Development

===============================================================================
"""

from __future__ import annotations

import torch

###############################################################################
# Numerical tolerances
###############################################################################

EPS: float = 1.0e-8

FLOAT_DTYPE = torch.float32

###############################################################################
# Hermite Basis Matrix
#
# Cubic Hermite polynomial basis
#
# H =
#
# |  2 -2  1  1 |
# | -3  3 -2 -1 |
# |  0  0  1  0 |
# |  1  0  0  0 |
#
###############################################################################

HERMITE_MATRIX = torch.tensor(
    [
        [2.0, -2.0, 1.0, 1.0],
        [-3.0, 3.0, -2.0, -1.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ],
    dtype=FLOAT_DTYPE,
)

###############################################################################
# Adaptive Tangent Parameters
###############################################################################

DEFAULT_TANGENT_LIMIT = 2.5

DEFAULT_MIN_SLICE_SPACING = 1e-3

DEFAULT_MAX_SCALING = 3.0

DEFAULT_MIN_SCALING = 0.25

###############################################################################
# Curvature Limits
###############################################################################

DEFAULT_MAX_MEAN_CURVATURE = 100.0

DEFAULT_MAX_GAUSSIAN_CURVATURE = 250.0

###############################################################################
# Jacobian Validation
###############################################################################

DEFAULT_MIN_JACOBIAN = 1.0e-6

DEFAULT_NEGATIVE_JACOBIAN_TOL = -1.0e-8

###############################################################################
# Surface Validation
###############################################################################

DEFAULT_MAX_NORMAL_CHANGE = 35.0

DEFAULT_MAX_PATCH_ASPECT = 8.0

DEFAULT_MAX_TANGENT_ENERGY = 10.0

###############################################################################
# Training Defaults
###############################################################################

DEFAULT_LAMBDA_JACOBIAN = 0.10

DEFAULT_LAMBDA_CURVATURE = 0.01

DEFAULT_LAMBDA_TANGENT = 0.02

DEFAULT_LAMBDA_BOUNDARY = 0.05

DEFAULT_LAMBDA_CAP = 0.05

###############################################################################
# Diagnostics
###############################################################################

DEFAULT_VERBOSE = False

DEFAULT_ENABLE_VALIDATION = True

DEFAULT_ENABLE_DIAGNOSTICS = True

###############################################################################
# Sampling
###############################################################################

DEFAULT_PATCH_SAMPLES_U = 32

DEFAULT_PATCH_SAMPLES_V = 32

DEFAULT_CURVATURE_STENCIL = 3

###############################################################################
# Miscellaneous
###############################################################################

PI = torch.pi

TWOPI = 2.0 * torch.pi