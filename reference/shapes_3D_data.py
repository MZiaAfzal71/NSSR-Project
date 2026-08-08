"""
Return predefined 3D shape datasets as stacked 2D contours.

Each dataset consists of multiple planar (x, y) contours placed at
z-heights, representing cross-sections of a 3D object.

Parameters
----------
ds : str, default='banana'
    Shape type: 'banana', 'apple', or 'vase' (default).

dtype : numpy.dtype or torch.dtype
    Controls numerical precision and backend (NumPy or PyTorch).

device : str or torch.device, default='cpu'
    Device for PyTorch tensors (ignored for NumPy).

Returns
-------
I : list
    List of (M, 2) contours representing (x, y) coordinates.

Z : array-like, shape (N,)
    z-heights corresponding to each contour.

Null_Hts : array-like, shape (2,)
    Lower and upper height bounds with no geometry.
"""

import numpy as np
import torch
from typing import Literal


def _shape_geometry(ds: str):
    ds = ds.lower()

    if ds == "banana":
        P = [
            [[0, 0, -3], [0.5, 0.75, -3], [0, 1.5, -3], [-0.5, 0.75, -3]],
            [[0, 1, -2], [0.6, 2, -2], [0, 3, -2], [-0.6, 2, -2]],
            [[0, 1.9, -1], [0.6, 2.9, -1], [0, 3.9, -1], [-0.6, 2.9, -1]],
            [[0, 2.6, 0], [0.6, 3.4, 0], [0, 4.2, 0], [-0.6, 3.4, 0]],
            [[0, 3, 1], [0.5, 3.65, 1], [0, 4.3, 1], [-0.5, 3.65, 1]],
            [[0, 3.2, 2], [0.15, 3.35, 2], [0, 3.5, 2], [-0.15, 3.35, 2]],
            [[0, 3, 2.3], [0.2, 3.3, 2.3], [0, 3.6, 2.3], [-0.2, 3.3, 2.3]]
        ]
        null_hts = [-3.5, 2.8]

    elif ds == "apple":
        P = [
            [[0.2, 0, 2.3], [0, 0.2, 2.3], [-0.2, 0, 2.3], [0, -0.2, 2.3]],
            [[0.7, 0, 0.7], [0, 0.7, 0.7], [-0.7, 0, 0.7], [0, -0.7, 0.7]],
            [[3, 0, 0], [0, 3, 0], [-3, 0, 0], [0, -3, 0]],
            [[6, 0, 2], [0, 6, 2], [-6, 0, 2], [0, -6, 2]],
            [[7.5, 0, 7], [0, 7.5, 7], [-7.5, 0, 7], [0, -7.5, 7]],
            [[7, 0, 9], [0, 7, 9], [-7, 0, 9], [0, -7, 9]],
            [[4, 0, 11], [0, 4, 11], [-4, 0, 11], [0, -4, 11]],
            [[1.3, 0, 10.5], [0, 1.3, 10.5], [-1.3, 0, 10.5], [0, -1.3, 10.5]],
            [[0.7, 0, 9.5], [0, 0.7, 9.5], [-0.7, 0, 9.5], [0, -0.7, 9.5]]
        ]
        null_hts = [3.5, 7.3]

    elif ds == "vase":
        P = [
            [[3, 0, 0.75], [0, 3, 0.75], [-3, 0, 0.75], [0, -3, 0.75]],
            [[3.75, 0, 0], [0, 3.75, 0], [-3.75, 0, 0], [0, -3.75, 0]],
            [[3.5, 0, 0.75], [0, 3.5, 0.75], [-3.5, 0, 0.75], [0, -3.5, 0.75]],
            [[3.25, 0, 1.5], [0, 3.25, 1.5], [-3.25, 0, 1.5], [0, -3.25, 1.5]],
            [[6, 0, 5], [0, 6, 5], [-6, 0, 5], [0, -6, 5]],
            [[3.5, 0, 8], [0, 3.5, 8], [-3.5, 0, 8], [0, -3.5, 8]],
            [[1.75, 0, 15], [0, 1.75, 15], [-1.75, 0, 15], [0, -1.75, 15]],
            [[3, 0, 21.5], [0, 3, 21.5], [-3, 0, 21.5], [0, -3, 21.5]],
            [[4, 0, 21], [0, 4, 21], [-4, 0, 21], [0, -4, 21]]
        ]
        null_hts = [1.7, 20.0]

    else:
        raise ValueError(f"Unknown dataset '{ds}'")

    return P, null_hts


def data_3d_shape(
    ds: str = "banana",
    backend: Literal["numpy", "torch"] = "numpy",
    dtype=None,
    device="cpu",
):
    """
    Unified 3D shape data loader supporting NumPy and PyTorch.
    """

    P, null_hts = _shape_geometry(ds)

    if backend == "numpy":
        dtype = dtype or np.float64
        P = np.asarray(P, dtype=dtype)
        I = P[:, :, :2]         # (n, 4, 2)
        Z = P[:, 0, 2]          # (n,)
        Null_Hts = np.asarray(null_hts, dtype=dtype)

    elif backend == "torch":
        dtype = dtype or torch.float64
        P = torch.tensor(P, dtype=dtype, device=device)
        I = P[:, :, :2]         # (n, 4, 2)
        Z = P[:, 0, 2]          # (n,)
        Null_Hts = torch.tensor(null_hts, dtype=dtype, device=device)

    else:
        raise ValueError("backend must be 'numpy' or 'torch'")

    return I, Z, Null_Hts

