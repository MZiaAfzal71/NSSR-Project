"""NumPy preprocessing (non-learnable stages, Eqs. 5-17).

For v1 we use arc-length resampling of each raw contour to a common m.
If you want bit-exact continuity with the CiSE pipeline, swap
`resample_contour` for your rational-cubic-spline resampler (Eqs. 1-4) from
the reference repository -- the interface is identical:
(K_i, 2) raw points -> (m, 2) resampled points.

Base/crown POINTS use the curvature-weighted average (Eqs. 10-11 spirit);
base/crown HEIGHTS use the three-point circle fit (Eqs. 12-17).
"""
from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------
def resample_contour(P: np.ndarray, m: int) -> np.ndarray:
    """Arc-length uniform resampling of a closed polygon P (K, 2) -> (m, 2)."""
    P = np.asarray(P, dtype=np.float64)
    Pc = np.vstack([P, P[:1]])                       # close
    seg = np.linalg.norm(np.diff(Pc, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    t = np.linspace(0.0, L, m, endpoint=False)
    x = np.interp(t, s, Pc[:, 0])
    y = np.interp(t, s, Pc[:, 1])
    return np.stack([x, y], axis=1)


def align_contours(contours: list[np.ndarray], M: int = 60) -> np.ndarray:
    """Cyclic alignment (Eqs. 7-8): shift each contour to minimize squared
    distance to the previous one.  contours: list of (m, 2).  Returns (N, m, 2).
    Also fixes orientation (all counter-clockwise)."""
    def ccw(C):
        area = 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1)
                            - np.roll(C[:, 0], -1) * C[:, 1])
        return C if area > 0 else C[::-1].copy()

    out = [ccw(contours[0])]
    m = out[0].shape[0]
    delta = max(m // M, 1)
    for C in contours[1:]:
        C = ccw(C)
        best_j, best_d = 0, np.inf
        idx = np.arange(0, m, delta)
        prev = out[-1][idx]
        for j in range(m):                            # full search; cheap
            d = np.sum((prev - C[(idx + j) % m]) ** 2)
            if d < best_d:
                best_d, best_j = d, j
        out.append(np.roll(C, -best_j, axis=0))
    return np.stack(out, axis=0)


# ----------------------------------------------------------------------
def base_crown_points(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted centers of the bottom and top contours (Eqs. 10-11 spirit).
    Uses diametric triangle weights; degenerates gracefully to the centroid."""
    def wcenter(C):
        m = C.shape[0]
        h = m // 2
        A, B = C, C[(np.arange(m) + h) % m]
        w = np.linalg.norm(A - B, axis=1)             # diameter weights
        w = np.where(w < 1e-12, 1.0, w)
        mid = 0.5 * (A + B)
        return (mid * w[:, None]).sum(0) / w.sum()
    return wcenter(R[0]), wcenter(R[-1])


def _cap_height(C0: np.ndarray, C1: np.ndarray, z0: float, z1: float,
                below: bool) -> float:
    """Circle-fit cap height (Eqs. 12-17): for pairs of diametrically
    opposite points p1, p2 on contour z0 and the corresponding p3 on z1,
    fit the circle through (p1,p2,p3) in 3-D and intersect its axis-plane
    construction with the object axis.  Averaged over M pairs."""
    m = C0.shape[0]
    M = min(24, m // 2)
    idx = np.linspace(0, m // 2 - 1, M).astype(int)
    zs = []
    for j in idx:
        k = (j + m // 2) % m
        p1 = np.array([C0[j, 0], C0[j, 1], z0])
        p2 = np.array([C0[k, 0], C0[k, 1], z0])
        p3 = np.array([C1[j, 0], C1[j, 1], z1])
        z = _circle_apex(p1, p2, p3, z0, below)
        if z is not None:
            zs.append(z)
    if not zs:                                        # flat fallback
        return z0 - 0.05 * abs(z1 - z0) if below else z0 + 0.05 * abs(z1 - z0)
    return float(np.mean(zs))


def _circle_apex(p1, p2, p3, z_ref, below):
    """Apex height of the circle through p1,p2,p3: the point on the circle
    farthest below (or above) z_ref, found via center + radius."""
    # circle center in 3-D: intersection of bisector planes within the
    # plane of the three points.
    v1, v2 = p2 - p1, p3 - p1
    n = np.cross(v1, v2)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return None
    n = n / nn
    A = np.stack([v1, v2, n])
    b = np.array([v1 @ (p1 + p2) / 2.0, v2 @ (p1 + p3) / 2.0, n @ p1])
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    r = np.linalg.norm(p1 - c)
    # extremal z on the circle: c_z -+ r * sqrt(1 - n_z^2)
    dz = r * np.sqrt(max(0.0, 1.0 - n[2] ** 2))
    z = c[2] - dz if below else c[2] + dz
    # guard: must lie beyond the reference slice, not absurdly far
    span = max(abs(p3[2] - z_ref), 1e-9)
    if below and not (z_ref - 2 * span <= z < z_ref):
        return None
    if (not below) and not (z_ref < z <= z_ref + 2 * span):
        return None
    return z


def cap_heights(R: np.ndarray, Z: np.ndarray) -> tuple[float, float]:
    Bh = _cap_height(R[0], R[1], float(Z[0]), float(Z[1]), below=True)
    Th = _cap_height(R[-1], R[-2], float(Z[-1]), float(Z[-2]), below=False)
    return Bh, Th


# ----------------------------------------------------------------------
def preprocess_object(raw_contours: list[np.ndarray], Z: np.ndarray,
                      m: int = 256) -> dict:
    """Raw slice contours -> everything geometry.py needs (all NumPy)."""
    contours = [resample_contour(C, m) for C in raw_contours]
    R = align_contours(contours)
    RB, RC = base_crown_points(R)
    Bh, Th = cap_heights(R, np.asarray(Z, dtype=np.float64))
    # normalize to unit scale (store the transform for inverse mapping)
    all_pts = np.concatenate([R.reshape(-1, 2),
                              np.array([[RB[0], RB[1]], [RC[0], RC[1]]])])
    center_xy = all_pts.mean(0)
    zmid = 0.5 * (Bh + Th)
    scale = max(np.abs(all_pts - center_xy).max(), abs(Th - zmid), 1e-9)
    R = (R - center_xy) / scale
    RB = (RB - center_xy) / scale
    RC = (RC - center_xy) / scale
    Z = (np.asarray(Z, dtype=np.float64) - zmid) / scale
    Bh = (Bh - zmid) / scale
    Th = (Th - zmid) / scale
    return {"R": R, "Z": Z, "RB": RB, "RC": RC, "Bh": Bh, "Th": Th,
            "norm": {"center_xy": center_xy, "zmid": zmid, "scale": scale}}
