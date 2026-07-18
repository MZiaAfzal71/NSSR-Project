"""NumPy preprocessing (non-learnable stages, Eqs. 5-17).

Two entry points:

  preprocess_object(raw_contours, Z, ...)
      Generic path for dense contours (mesh slices, synthetic data):
      arc-length resampling to fixed m + full cyclic-shift alignment,
      then base/crown points and circle-fit heights ported EXACTLY from
      reference/surfaces_python_loops.py (base_crown_pt / base_crown_ht).

  preprocess_designer(ds, n1, ...)
      Designer path for the CiSE datasets (banana / apple / vase):
      runs the vendored reference pipeline verbatim (curve_goodman
      resampling, coarse M-shift matching), so these objects flow through
      NSSR with zero drift from the published results.  tests/parity_check
      guarantees the downstream geometry then matches the reference too.

Both return the same dict consumed by nssr.train.to_torch:
  R (N,m,2), Z (N,), RB, RC (2,), Bh, Th, base_circular, crown_circular,
  norm {center_xy, zmid, scale}.
"""
from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------
# Resampling and alignment (generic path)
# ----------------------------------------------------------------------
def resample_contour(P: np.ndarray, m: int) -> np.ndarray:
    """Arc-length uniform resampling of a closed polygon (K,2) -> (m,2).
    For designer data with few control points use preprocess_designer
    (rational-cubic-spline resampling) instead."""
    P = np.asarray(P, dtype=np.float64)
    Pc = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(Pc, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], m, endpoint=False)
    return np.stack([np.interp(t, s, Pc[:, 0]),
                     np.interp(t, s, Pc[:, 1])], axis=1)


def align_contours(contours: list[np.ndarray], M: int = 60) -> np.ndarray:
    """Cyclic alignment (Eqs. 7-8): full shift search minimizing sampled
    squared distance to the previous contour (a refinement of the
    reference's coarse M-shift search).  Also fixes orientation (CCW)."""
    def ccw(C):
        area = 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1)
                            - np.roll(C[:, 0], -1) * C[:, 1])
        return C if area > 0 else C[::-1].copy()

    out = [ccw(contours[0])]
    m = out[0].shape[0]
    delta = max(m // M, 1)
    idx = np.arange(0, m, delta)
    for C in contours[1:]:
        C = ccw(C)
        prev = out[-1][idx]
        d = [np.sum((prev - C[(idx + j) % m]) ** 2) for j in range(m)]
        out.append(np.roll(C, -int(np.argmin(d)), axis=0))
    return np.stack(out, axis=0)


# ----------------------------------------------------------------------
# Base / crown POINTS — exact port of reference base_crown_pt
# ----------------------------------------------------------------------
def base_crown_points(R: np.ndarray, n_pairs: int | None = None):
    """Curvature-weighted end-contour centers (Eqs. 10-11), reference
    weighting: (gamma*A + alpha*B + beta*(A+B)) / (alpha + 2 beta + gamma)
    with alpha=||A-C||, beta=||A-B||, gamma=||B-D||, averaged over
    diametric pairs.  A,B on the end contour; C,D on the adjacent one."""
    N, m, _ = R.shape
    n_pairs = n_pairs or min(24, m // 2)
    j = (np.linspace(0, m // 2 - 1, n_pairs)).astype(int)
    k = (j + m // 2) % m

    def wcenter(end, adj):
        A, B = end[j], end[k]
        C, D = adj[j], adj[k]
        alpha = np.linalg.norm(A - C, axis=1)
        beta = np.linalg.norm(A - B, axis=1)
        gamma = np.linalg.norm(B - D, axis=1)
        denom = (alpha + 2 * beta + gamma)[:, None]
        contrib = (gamma[:, None] * A + alpha[:, None] * B
                   + beta[:, None] * (A + B)) / denom
        return contrib.mean(axis=0)

    return wcenter(R[0], R[1]), wcenter(R[-1], R[-2])


# ----------------------------------------------------------------------
# Base / crown HEIGHTS — exact port of reference base_crown_ht
# ----------------------------------------------------------------------
def _det3(M):
    return np.linalg.det(np.asarray(M, dtype=np.float64))


def _height_one(p_end_j, p_end_k, p_adj_j, z_end, z_adj, null_ht,
                reverse: bool):
    x1, y1, z1 = p_end_j[0], p_end_j[1], z_end
    x2, y2, z2 = p_end_k[0], p_end_k[1], z_end
    x3, y3, z3 = p_adj_j[0], p_adj_j[1], z_adj
    x4 = -_det3([[y1, z1, 1], [y2, z2, 1], [y3, z3, 1]])
    y4 = _det3([[x1, z1, 1], [x2, z2, 1], [x3, z3, 1]])
    z4 = -_det3([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])
    b4 = _det3([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
    try:
        X = np.linalg.solve(
            [[2*x1, 2*y1, 2*z1, 1], [2*x2, 2*y2, 2*z2, 1],
             [2*x3, 2*y3, 2*z3, 1], [x4, y4, z4, 0]],
            [-(x1**2 + y1**2 + z1**2), -(x2**2 + y2**2 + z2**2),
             -(x3**2 + y3**2 + z3**2), b4])
    except np.linalg.LinAlgError:
        return null_ht
    u, v, w, d = X
    mdx1, mdy1, mdz1 = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2
    a = (-u - mdx1) / (-w - mdz1)
    b = (-v - mdy1) / (-w - mdz1)
    p1 = a**2 + b**2 + 1
    p2 = (2*a*(mdx1 - a*mdz1) + 2*b*(mdy1 - b*mdz1) - 2*u*a + 2*v*b + 2*w)
    p3 = ((mdx1 - a*mdz1)**2 + (mdy1 - b*mdz1)**2
          + 2*u*(mdx1 - a*mdz1) + 2*v*(mdy1 - b*mdz1) + d)
    disc = max(p2**2 - 4*p1*p3, 0.0)
    r1 = (-p2 + np.sqrt(disc)) / (2*p1)
    r2 = (-p2 - np.sqrt(disc)) / (2*p1)
    if reverse:                                        # crown
        if null_ht > z_end:
            return min(null_ht, max(r1, r2))
        return max(null_ht, min(r1, r2))
    if null_ht < z_end:                                # base
        return max(null_ht, min(r1, r2))
    return min(null_ht, max(r1, r2))


def cap_heights(R: np.ndarray, Z: np.ndarray, null_hts,
                n_pairs: int | None = None):
    """Mean circle-fit heights (Eqs. 12-17), reference semantics."""
    N, m, _ = R.shape
    n_pairs = n_pairs or min(24, m // 2)
    js = (np.linspace(0, m - 1, n_pairs)).astype(int)
    zb, zc = [], []
    for j in js:
        k = (j + m // 2) % m
        zb.append(_height_one(R[0, j], R[0, k], R[1, j],
                              float(Z[0]), float(Z[1]), float(null_hts[0]),
                              reverse=False))
        zc.append(_height_one(R[-1, j], R[-1, k], R[-2, j],
                              float(Z[-1]), float(Z[-2]), float(null_hts[1]),
                              reverse=True))
    return float(np.mean(zb)), float(np.mean(zc))


def default_null_hts(Z: np.ndarray):
    """When no physical prior is given (mesh / synthetic data): allow the
    caps to extend up to one inter-slice gap beyond the end contours."""
    return (float(Z[0] - abs(Z[1] - Z[0])),
            float(Z[-1] + abs(Z[-1] - Z[-2])))


# ----------------------------------------------------------------------
def _normalize(R, Z, RB, RC, Bh, Th):
    all_pts = np.concatenate([R.reshape(-1, 2), RB[None], RC[None]])
    center_xy = all_pts.mean(0)
    zmid = 0.5 * (Bh + Th)
    scale = max(np.abs(all_pts - center_xy).max(),
                abs(Th - zmid), abs(Bh - zmid), 1e-9)
    return {"R": (R - center_xy) / scale, "Z": (Z - zmid) / scale,
            "RB": (RB - center_xy) / scale, "RC": (RC - center_xy) / scale,
            "Bh": (Bh - zmid) / scale, "Th": (Th - zmid) / scale,
            "norm": {"center_xy": center_xy, "zmid": zmid, "scale": scale}}


def preprocess_object(raw_contours, Z, m: int = 256, null_hts=None,
                      base_circular=True, crown_circular=True,
                      use_null_hts_directly=False) -> dict:
    """Generic path: raw slice contours -> everything geometry.py needs."""
    Z = np.asarray(Z, dtype=np.float64)
    contours = [resample_contour(C, m) for C in raw_contours]
    R = align_contours(contours)
    RB, RC = base_crown_points(R)
    null_hts = null_hts if null_hts is not None else default_null_hts(Z)
    if use_null_hts_directly:                # e.g. the apple in the paper
        Bh, Th = float(null_hts[0]), float(null_hts[1])
    else:
        Bh, Th = cap_heights(R, Z, null_hts)
    out = _normalize(R, Z, RB, RC, Bh, Th)
    out["base_circular"] = base_circular
    out["crown_circular"] = crown_circular
    return out


def preprocess_designer(ds: str = "banana", n1: int = 25) -> dict:
    """Designer path: run the vendored reference pipeline verbatim on the
    CiSE datasets.  Drops the duplicated closing point so the (N, m, 2)
    layout matches the generic path (m = tot_pts)."""
    import sys, types
    try:
        import torch  # noqa: F401
    except ImportError:                       # reference module imports torch
        sys.modules.setdefault("torch", types.ModuleType("torch"))
    from reference.shapes_3D_data import data_3d_shape
    from reference.curves_python_loops import curve_goodman
    from reference.surfaces_python_loops import (
        t_no_pts, match_parameters, base_crown_pt, base_crown_ht)

    I, Z, Null_Hts = data_3d_shape(ds, dtype=np.float64)
    tot_pts, seg_pts = t_no_pts(I, n1)
    N = len(seg_pts); M = 4; step = tot_pts // M
    r = np.stack([curve_goodman(I[k], seg_pts[k]) for k in range(len(I))])
    Rfull = match_parameters(r, N, tot_pts, M)          # (N, tot_pts+1, 2)
    RB, RC = base_crown_pt(Rfull, N, tot_pts, M, step)
    if ds == "apple":
        Bh, Th = float(Null_Hts[0]), float(Null_Hts[1])
        circ = False
    else:
        Bh, Th = base_crown_ht(Rfull, N, tot_pts, M, step, Z, Null_Hts,
                               dtype=np.float64)
        circ = True
    out = _normalize(Rfull[:, :-1].astype(np.float64),
                     np.asarray(Z, dtype=np.float64),
                     np.asarray(RB, dtype=np.float64),
                     np.asarray(RC, dtype=np.float64),
                     float(Bh), float(Th))
    out["base_circular"] = circ
    out["crown_circular"] = circ
    out["closed_top"] = (ds != "vase")        # vase top is open
    return out
