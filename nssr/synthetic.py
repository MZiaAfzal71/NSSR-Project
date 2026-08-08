"""Synthetic training data: random generalized cylinders / solids of
revolution with harmonic cross-sections and closed (or open) caps.

Surface (analytic, dense GT):
    S(theta, t) = ( x0(t) + r(theta, t) cos(theta),
                    y0(t) + r(theta, t) sin(theta),
                    z(t) ),  t in [0, 1], theta in [0, 2*pi)
    r(theta, t) = r0(t) * (1 + sum_k eps_k(t) * cos(k*theta + phi_k))

Each object belongs to one of three FAMILIES, chosen so the training
distribution genuinely covers the geometric stress-cases the three CiSE
designer shapes represent (banana, apple, vase) rather than only smooth
convex blobs:

  "standard" -- monotonic height z(t) = height*t, both caps present.
                A bent variant (curved axis, narrow tapered ends) covers
                the BANANA-like case.
  "dimpled"  -- z(t) is deliberately NON-MONOTONIC near one or both poles
                (a smooth local "fold-back", via _pole_dip), so adjacent
                INTERIOR slices can have inverted height order -- the same
                stress case the paper's APPLE requires (see abstract:
                "non-monotone cross sections... inward-dipping profiles of
                an apple"). The true first/last slices stay monotonic
                w.r.t. their immediate neighbour so default_null_hts still
                behaves sensibly; only interior adjacency is folded.
  "open_top" -- no crown contour at all (closed_top=False), wide un-tapered
                rim -- the case a genuinely open real-world scan (a bowl,
                an open vase) would present, as opposed to the paper's
                vase, which DOES have crown data (see nssr/preprocess.py).

Independently of family, base_circular / crown_circular are randomized
(mostly True, sometimes False) so the network also sees the apple's
non-circular-cap convention on other shapes.

Each sample provides:
  - sparse input: N slice contours (raw, variable point counts)
  - dense GT point cloud + analytic normals
  - base_circular, crown_circular, closed_top flags (mirrors what
    preprocess_object / preprocess_designer attach to real data)
"""
from __future__ import annotations
import numpy as np

FAMILY_PROBS = {"standard": 0.5, "dimpled": 0.25, "open_top": 0.25}


def _smooth_profile(rng, n_ctrl=6, lo=0.25, hi=1.0, taper_lo=True, taper_hi=True):
    """Random smooth positive function of t in [0,1] via cosine-interp of
    control values. taper_lo/hi shrink the corresponding endpoint value to
    create cap curvature; set False to keep a wide, un-tapered rim (used
    for the open_top family's open end)."""
    v = rng.uniform(lo, hi, n_ctrl)
    if taper_lo:
        v[0] *= rng.uniform(0.3, 0.7)
    if taper_hi:
        v[-1] *= rng.uniform(0.3, 0.7)
    tc = np.linspace(0, 1, n_ctrl)

    def f(t):
        t = np.asarray(t)
        i = np.clip(np.searchsorted(tc, t, side="right") - 1, 0, n_ctrl - 2)
        w = (t - tc[i]) / (tc[i + 1] - tc[i])
        w = 0.5 - 0.5 * np.cos(np.pi * w)          # smooth-step
        return v[i] * (1 - w) + v[i + 1] * w
    return f


def _pole_dip(t, t0, width, depth):
    """Smooth compactly-supported bump peaked at t0 (raised-cosine), used
    to fold the height map z(t) back on itself near a pole -- the same
    kind of local non-monotonicity the real apple's Z sequence has between
    adjacent slices (see module docstring)."""
    t = np.asarray(t, dtype=np.float64)
    u = np.clip((t - t0) / width, -1.0, 1.0)
    bump = 0.5 * (1.0 + np.cos(np.pi * u))
    return depth * bump * (np.abs(t - t0) < width)


def sample_object(rng: np.random.Generator, max_harmonic=5, bend=True,
                  family: str | None = None):
    """Return a dict describing one random object (functions of t, theta)
    plus metadata (family, closed_top, base_circular, crown_circular)."""
    if family is None:
        names, probs = zip(*FAMILY_PROBS.items())
        family = rng.choice(names, p=probs)

    open_top = (family == "open_top")
    r0 = _smooth_profile(rng, taper_hi=not open_top)

    K = rng.integers(0, max_harmonic + 1)
    harmonics = []
    for k in rng.choice(np.arange(2, 7), size=K, replace=False):
        amp = _smooth_profile(rng, lo=0.0, hi=0.18 / np.sqrt(k))
        phi = rng.uniform(0, 2 * np.pi)
        harmonics.append((int(k), amp, phi))

    # bent axis: BANANA-like case. More likely / more pronounced for
    # "standard" family so that family carries the elongated-curved stress.
    bend_prob = 0.55 if family == "standard" else 0.25
    if bend and rng.random() < bend_prob:
        mag = rng.uniform(0.3, 0.9) if family == "standard" else rng.uniform(0.2, 0.5)
        bx = rng.uniform(-mag, mag); by = rng.uniform(-mag, mag)
        axis = lambda t: (bx * np.sin(np.pi * t), by * np.sin(np.pi * t))
    else:
        axis = lambda t: (np.zeros_like(t), np.zeros_like(t))

    height = rng.uniform(1.0, 2.5)

    if family == "dimpled":
        # fold near one or both poles, kept well inside the interior so
        # the TRUE first/last slice (t_lo/t_hi in slice_object) stays
        # monotonic w.r.t. its own immediate neighbour. Parameters
        # calibrated empirically (see docs/RESULTS_AND_TUNING.md): zero
        # endpoint-monotonicity violations across 3000 randomized trials
        # (including slice jitter), ~40% of dimpled samples still get a
        # genuine interior Z reversal between adjacent slices.
        dips = []
        if rng.random() < 0.85:
            dips.append((rng.uniform(0.22, 0.30), rng.uniform(0.035, 0.05),
                        height * rng.uniform(0.10, 0.18)))
        if rng.random() < 0.85:
            dips.append((rng.uniform(0.70, 0.78), rng.uniform(0.035, 0.05),
                        height * rng.uniform(0.10, 0.18)))

        def z_map(t, _dips=dips, _h=height):
            t = np.asarray(t, dtype=np.float64)
            z = _h * t
            for t0, width, depth in _dips:
                z = z - _pole_dip(t, t0, width, depth)
            return z
    else:
        def z_map(t, _h=height):
            return _h * np.asarray(t, dtype=np.float64)

    base_circular = bool(rng.random() < 0.8)
    crown_circular = bool(rng.random() < 0.8) if not open_top else True

    return {"r0": r0, "harmonics": harmonics, "axis": axis, "height": height,
            "z_map": z_map, "family": family, "closed_top": not open_top,
            "base_circular": base_circular, "crown_circular": crown_circular}


def radius(obj, theta, t):
    base = np.ones_like(theta * t)
    for k, amp, phi in obj["harmonics"]:
        base = base + amp(t) * np.cos(k * theta + phi)
    return obj["r0"](t) * np.clip(base, 0.2, None)


def surface_point(obj, theta, t):
    r = radius(obj, theta, t)
    x0, y0 = obj["axis"](t)
    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)
    z = obj["z_map"](t)
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def dense_ground_truth(obj, n_theta=256, n_t=256):
    """Dense GT points and normals via finite differences.  For open_top
    objects the parameter range stops at a wide-open rim (no pole, no cap
    surface), matching what an open real-world scan would provide."""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    t_hi = 0.90 if not obj["closed_top"] else 0.995
    t = np.linspace(0.005, t_hi, n_t)
    TH, T = np.meshgrid(th, t, indexing="ij")
    P = surface_point(obj, TH, T)                      # (n_theta, n_t, 3)
    dth = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    dt = np.gradient(P, axis=1)
    n = np.cross(dth, dt)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)
    return P.reshape(-1, 3), n.reshape(-1, 3)


def slice_object(obj, N=7, n_pts_range=(60, 200), rng=None,
                 t_lo=0.06, t_hi=0.94, jitter=0.15):
    """Sparse input: N contours at (slightly jittered) heights.  For
    open_top objects the last slice sits at the open rim instead."""
    rng = rng or np.random.default_rng()
    hi = 0.85 if not obj["closed_top"] else t_hi
    t = np.linspace(t_lo, hi, N)
    if N > 2:
        dt = (hi - t_lo) / (N - 1)
        t[1:-1] += rng.uniform(-jitter * dt, jitter * dt, N - 2)
    contours, Z = [], []
    for ti in t:
        npts = int(rng.integers(*n_pts_range))
        th = np.linspace(0, 2 * np.pi, npts, endpoint=False)
        C = surface_point(obj, th, np.full_like(th, ti))
        contours.append(C[:, :2])
        Z.append(C[0, 2])
    return contours, np.array(Z)


def make_sample(seed: int, N: int = 7, family: str | None = None):
    """One (input, ground-truth) pair, fully reproducible from the seed.
    Pass family='standard'|'dimpled'|'open_top' to force a family instead
    of sampling one (used by the dataset generator's balancing option)."""
    rng = np.random.default_rng(seed)
    obj = sample_object(rng, family=family)
    contours, Z = slice_object(obj, N=N, rng=rng)
    gt_pts, gt_normals = dense_ground_truth(obj)
    return {"contours": contours, "Z": Z,
            "gt_pts": gt_pts, "gt_normals": gt_normals, "seed": seed,
            "family": obj["family"], "closed_top": obj["closed_top"],
            "base_circular": obj["base_circular"],
            "crown_circular": obj["crown_circular"]}
