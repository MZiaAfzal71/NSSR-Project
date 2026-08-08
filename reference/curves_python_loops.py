# curves_python_loops.py
# ----------------------
# Reference implementation of 2D curve smoothing using Goodman’s method.
# This version is written in pure NumPy with explicit Python loops and
# is intended for correctness, clarity, and baseline comparison.
#
# ⚠️ This file does NOT use Numba, GPU acceleration, or PyTorch.
# Optimized implementations (Numba / Numba+GPU / PyTorch) are provided
# in separate files within the capsule.
#
# Used by Code Ocean reproducible runs as the CPU-only, loop-based variant.

import numpy as np


def cross2d(a, b):
    """2D cross product scalar for arrays of 2-vectors: a,b shape (...,2) -> (...,)"""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def curve_goodman(I, seg_pts, dtype=None):
    """
    Smooth curve through given data points using Goodman’s method.

    Parameters
    ----------
    I : (n,2) ndarray
        Input polyline points.
    seg_pts : int
        Number of interpolation points per segment.

    Returns
    -------
    r : (n*seg_pts,2) ndarray
        Smoothed curve points.
    """
    if dtype is None:
        dtype = I.dtype

    no_pts = I.shape[0]

    # constants
    m, n = 0.5, 0.5
    r_coef, s_coef = 0.25, 0.25
    t_vals = np.linspace(0, 1, seg_pts + 1, dtype=dtype)[:-1] # ignore end-point

    # storage
    lineleft = np.zeros((no_pts,), dtype=np.bool_)
    lineright = np.zeros((no_pts,), dtype=np.bool_)
    kleft = np.zeros(no_pts, dtype=dtype)
    kright = np.zeros(no_pts, dtype=dtype)

    # --- curvature flags + curvature values
    for i in range(no_pts):
        # indices with wrap-around
        im1, ip1, ip2, im2 = (i-1) % no_pts, (i+1) % no_pts, (i+2) % no_pts, (i-2) % no_pts

        # collinearity check
        if abs((I[i,1]-I[im1,1])*(I[ip1,0]-I[im1,0]) - (I[ip1,1]-I[im1,1])*(I[i,0]-I[im1,0])) < 1e-9:
            lineleft[i] = True; lineright[i] = True
        elif abs((I[i,1]-I[ip1,1])*(I[ip2,0]-I[ip1,0]) - (I[ip2,1]-I[ip1,1])*(I[i,0]-I[ip1,0])) < 1e-9:
            lineright[i] = True
        elif abs((I[i,1]-I[im1,1])*(I[im2,0]-I[im1,0]) - (I[im2,1]-I[im1,1])*(I[i,0]-I[im1,0])) < 1e-9:
            lineleft[i] = True

        if not lineleft[i]:
            num = 2*cross2d(I[i]-I[im1], I[ip1]-I[i])
            den = np.linalg.norm(I[i]-I[im1])*np.linalg.norm(I[ip1]-I[i])*np.linalg.norm(I[ip1]-I[im1])
            kleft[i] = num/den if den != 0 else 0
        if not lineright[i]:
            num = 2*cross2d(I[i]-I[im1], I[ip1]-I[i])
            den = np.linalg.norm(I[i]-I[im1])*np.linalg.norm(I[ip1]-I[i])*np.linalg.norm(I[ip1]-I[im1])
            kright[i] = num/den if den != 0 else 0

    # --- tangents
    T = np.zeros((no_pts,2), dtype=dtype)
    Tu = np.zeros((no_pts,2), dtype=dtype)
    for i in range(no_pts):
        im1, ip1 = (i-1) % no_pts, (i+1) % no_pts
        if kleft[i] != 0 or kright[i] != 0:
            a = abs(kleft[ip1]) * np.linalg.norm(I[ip1]-I[i])**2
            b = abs(kright[im1]) * np.linalg.norm(I[i]-I[im1])**2
            T[i] = a*(I[i]-I[im1]) + b*(I[ip1]-I[i])
            normT = np.linalg.norm(T[i])
            Tu[i] = T[i]/normT if normT != 0 else np.zeros(2)

    # --- Bezier control points
    A, B, C, D = np.zeros_like(I), np.zeros_like(I), np.zeros_like(I), np.zeros_like(I)
    lengthab, lengthcd = np.zeros(no_pts, dtype=dtype), np.zeros(no_pts, dtype=dtype)

    for i in range(no_pts):
        ip1 = (i+1) % no_pts
        if kright[i]*kleft[ip1] > 0:  # convex
            sina = cross2d(T[i], I[ip1]-I[i]) / (np.linalg.norm(T[i])*np.linalg.norm(I[ip1]-I[i]))
            sinb = cross2d(I[ip1]-I[i], T[ip1]) / (np.linalg.norm(T[ip1])*np.linalg.norm(I[ip1]-I[i]))
            sinab = cross2d(T[i], T[ip1]) / (np.linalg.norm(T[i])*np.linalg.norm(T[ip1]))
            L = np.linalg.norm(I[ip1]-I[i])
            p = 2*abs(sinb)/(2*m*abs(sinb)+(1-m)*L*abs(kleft[ip1])+2*abs(sinab))
            q = 2*abs(sina)/(2*n*abs(sina)+(1-n)*L*abs(kright[i])+2*abs(sinab))
            lengthab[i], lengthcd[i] = p*L, q*L
            A[i], D[i] = I[i], I[ip1]
            B[i], C[i] = A[i]+lengthab[i]*Tu[i], D[i]-lengthcd[i]*Tu[ip1]

        elif kright[i]*kleft[ip1] < 0:  # inflection
            L = np.linalg.norm(I[ip1]-I[i])
            lengthab[i], lengthcd[i] = r_coef*L, s_coef*L
            A[i], D[i] = I[i], I[ip1]
            B[i], C[i] = A[i]+lengthab[i]*Tu[i], D[i]-lengthcd[i]*Tu[ip1]

        else:  # straight
            A[i], D[i] = I[i], I[ip1]

    # --- weights
    alpha, beta = np.zeros(no_pts, dtype=dtype), np.zeros(no_pts, dtype=dtype)
    for i in range(no_pts):
        ip1 = (i+1) % no_pts
        if kright[i] != 0:
            denom1 = 2*cross2d(B[i]-A[i], C[i]-B[i])
            if denom1 != 0:
                alpha[i] = kright[i]*lengthab[i]**3/denom1
            denom2 = 2*cross2d(C[i]-B[i], D[i]-C[i])
            if denom2 != 0:
                beta[i] = kleft[ip1]*lengthcd[i]**3/denom2

    # --- final curve points
    r = np.zeros((no_pts*seg_pts,2), dtype=dtype)
    for i in range(no_pts):
        if kright[i] != 0:  # nonlinear
            for j, tj in enumerate(t_vals):
                num = (A[i]*alpha[i]*(1-tj)**3 +
                       B[i]*tj*(1-tj)**2 +
                       C[i]*tj**2*(1-tj) +
                       D[i]*beta[i]*tj**3)
                den = alpha[i]*(1-tj)**3 + tj*(1-tj)**2 + tj**2*(1-tj) + beta[i]*tj**3
                r[i*seg_pts+j] = num/den
        else:  # straight segment
            for j, tj in enumerate(t_vals):
                r[i*seg_pts+j] = (1-tj)*A[i] + tj*D[i]

    return r