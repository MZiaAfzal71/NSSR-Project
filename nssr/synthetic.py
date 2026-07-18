"""Synthetic training data: random generalized cylinders / solids of
revolution with harmonic cross-sections and closed (or open) caps.

Surface (analytic, dense GT):
    S(theta, t) = ( x0(t) + r(theta, t) cos(theta),
                    y0(t) + r(theta, t) sin(theta),
                    z(t) ),  t in [0, 1], theta in [0, 2*pi)
    r(theta, t) = r0(t) * (1 + sum_k eps_k(t) * cos(k*theta + phi_k))

r0(t) is a random smooth positive profile that pinches toward the caps;
eps_k(t) varies smoothly with height (so the cross-section SHAPE changes
between slices — this is what makes sparse-slice interpolation hard and
what the learned tangents must capture).  An optional bent axis
(x0(t), y0(t)) produces banana-like objects.

Each sample provides:
  - sparse input: N slice contours (raw, variable point counts)
  - dense GT point cloud + analytic normals
"""
from __future__ import annotations
import numpy as np


def _smooth_profile(rng, n_ctrl=6, lo=0.25, hi=1.0):
    """Random smooth positive function of t in [0,1] via cosine-interp of
    control values; endpoints reduced to create cap curvature."""
    v = rng.uniform(lo, hi, n_ctrl)
    v[0] *= rng.uniform(0.3, 0.7)
    v[-1] *= rng.uniform(0.3, 0.7)
    tc = np.linspace(0, 1, n_ctrl)

    def f(t):
        t = np.asarray(t)
        i = np.clip(np.searchsorted(tc, t, side="right") - 1, 0, n_ctrl - 2)
        w = (t - tc[i]) / (tc[i + 1] - tc[i])
        w = 0.5 - 0.5 * np.cos(np.pi * w)          # smooth-step
        return v[i] * (1 - w) + v[i + 1] * w
    return f


def sample_object(rng: np.random.Generator, max_harmonic=5, bend=True):
    """Return a dict describing one random object (functions of t, theta)."""
    r0 = _smooth_profile(rng)
    K = rng.integers(0, max_harmonic + 1)
    harmonics = []
    for k in rng.choice(np.arange(2, 7), size=K, replace=False):
        amp = _smooth_profile(rng, lo=0.0, hi=0.18 / np.sqrt(k))
        phi = rng.uniform(0, 2 * np.pi)
        harmonics.append((int(k), amp, phi))
    if bend and rng.random() < 0.4:
        bx = rng.uniform(-0.5, 0.5); by = rng.uniform(-0.5, 0.5)
        axis = lambda t: (bx * np.sin(np.pi * t), by * np.sin(np.pi * t))
    else:
        axis = lambda t: (np.zeros_like(t), np.zeros_like(t))
    height = rng.uniform(1.0, 2.5)
    return {"r0": r0, "harmonics": harmonics, "axis": axis, "height": height}


def radius(obj, theta, t):
    r = obj["r0"](t) * np.ones_like(theta * t)
    base = np.ones_like(theta * t)
    for k, amp, phi in obj["harmonics"]:
        base = base + amp(t) * np.cos(k * theta + phi)
    return obj["r0"](t) * np.clip(base, 0.2, None)


def surface_point(obj, theta, t):
    r = radius(obj, theta, t)
    x0, y0 = obj["axis"](t)
    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)
    z = obj["height"] * t
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def dense_ground_truth(obj, n_theta=256, n_t=256):
    """Dense GT points and normals via finite differences."""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    t = np.linspace(0.005, 0.995, n_t)
    TH, T = np.meshgrid(th, t, indexing="ij")
    P = surface_point(obj, TH, T)                      # (n_theta, n_t, 3)
    dth = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    dt = np.gradient(P, axis=1)
    n = np.cross(dth, dt)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)
    return P.reshape(-1, 3), n.reshape(-1, 3)


def slice_object(obj, N=7, n_pts_range=(60, 200), rng=None,
                 t_lo=0.06, t_hi=0.94, jitter=0.15):
    """Sparse input: N contours at (slightly jittered) heights."""
    rng = rng or np.random.default_rng()
    t = np.linspace(t_lo, t_hi, N)
    if N > 2:
        dt = (t_hi - t_lo) / (N - 1)
        t[1:-1] += rng.uniform(-jitter * dt, jitter * dt, N - 2)
    contours, Z = [], []
    for ti in t:
        npts = int(rng.integers(*n_pts_range))
        th = np.linspace(0, 2 * np.pi, npts, endpoint=False)
        C = surface_point(obj, th, np.full_like(th, ti))
        contours.append(C[:, :2])
        Z.append(C[0, 2])
    return contours, np.array(Z)


def make_sample(seed: int, N: int = 7):
    """One (input, ground-truth) pair, fully reproducible from the seed."""
    rng = np.random.default_rng(seed)
    obj = sample_object(rng)
    contours, Z = slice_object(obj, N=N, rng=rng)
    gt_pts, gt_normals = dense_ground_truth(obj)
    return {"contours": contours, "Z": Z,
            "gt_pts": gt_pts, "gt_normals": gt_normals, "seed": seed}
