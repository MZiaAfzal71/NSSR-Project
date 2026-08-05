"""Slice real watertight meshes into cross-sectional contours (Phase 1b).

Sources to populate data/meshes/ :
  - Thingi10k (https://ten-thousand-models.appspot.com) -- filter watertight
  - ShapeNetCore categories: bottle, vase, jar, can, mug
  - Your own scans of fruit / pottery

Selection policy for v1: keep only meshes for which EVERY slice yields a
single closed loop with NO holes (genus-0, star-shaped-ish objects -- the
same object class the classical pipeline assumes).  Multi-loop or holed
slices are a documented limitation / future-work item.

Correctness notes (all three were real bugs found by review; this module
cannot be exercised in an environment without trimesh, so it is written
defensively and `scripts/check_mesh_pipeline.py` validates it on your box):

 1. AXIS ALIGNMENT MUST BE A ROTATION, NOT A REFLECTION.  Building the
    permutation matrix naively gives det = -1 for 3 of the 6 orderings,
    which MIRRORS the mesh and inverts its normals -- silently corrupting
    the normal-consistency metric.  We now force det = +1.
 2. SECTION -> 2-D MUST USE AN EXPLICIT TRANSFORM.  trimesh's
    `Path3D.to_planar()` picks its own in-plane frame, which can differ
    from slice to slice; contours from different heights would then live in
    inconsistent, mutually rotated frames, destroying cross-slice
    correspondence (and with it the whole reconstruction).  Because our
    cutting planes are always z = const with normal +z, we pass an explicit
    to_2D transform so the planar coordinates are exactly world (x, y).
 3. POLYGONS WITH HOLES MUST BE REJECTED, not just multi-polygon slices.
"""
from __future__ import annotations
import numpy as np

try:
    import trimesh
except ImportError:                                   # pragma: no cover
    trimesh = None


def _require_trimesh():
    if trimesh is None:
        raise ImportError(
            "trimesh is required for mesh slicing: "
            "pip install trimesh shapely --break-system-packages")


def _rotation_for_order(order) -> np.ndarray:
    """3x3 proper rotation (det=+1) that permutes axes into `order`
    (columns), i.e. mesh axis order[-1] becomes the new z. Never a
    reflection: if the naive permutation has det=-1, flip one column."""
    M = np.eye(3)[:, order]
    if np.linalg.det(M) < 0:
        M[:, 0] = -M[:, 0]
    return M


def _apply_axis_order(mesh, order):
    """Return a COPY of mesh with axis `order[-1]` (in the current frame)
    rotated up onto +z. `order` is any permutation of (0,1,2)."""
    m2 = mesh.copy()
    M = _rotation_for_order(list(order))
    T = np.eye(4)
    T[:3, :3] = M
    m2.apply_transform(T)
    return m2


def _elongation(contour: np.ndarray) -> float:
    """PCA aspect ratio of a single closed 2-D contour: ratio of the larger
    to the smaller in-plane singular value of the (centered) point spread.
    ~1.0 for a round/star-shaped cross-section, large for a thin,
    plank-like one. Used to score candidate slicing axes: the axis giving
    round cross-sections is the one the classical/NSSR pipeline (built for
    star-shaped-ish contours) is designed for."""
    C = np.asarray(contour, dtype=np.float64)
    Cc = C - C.mean(axis=0, keepdims=True)
    # singular values of the centered point cloud == sqrt(eigenvalues of
    # the 2x2 covariance) up to a constant factor; ratio is all that matters
    s = np.linalg.svd(Cc, compute_uv=False)
    lo = max(s[-1], 1e-12)
    return float(s[0] / lo)


def _best_slicing_axis(mesh, N_probe: int = 9, margin: float = 0.06):
    """Try each of the 3 axes as the slicing (z) axis, slice a small probe
    set of contours for each, and keep whichever gives the roundest
    (lowest mean elongation) cross-sections.

    Rationale: `mesh.extents`-based "longest axis -> z" is the right
    heuristic for elongated objects (poles, bottles, bananas) but actively
    wrong for flattish disk/washer/lens-shaped objects, where the longest
    extent is a DIAMETER, not the thickness -- standing it up along z
    slices parallel to the flat faces and produces long, thin, near-
    degenerate contours (exactly the `thingi_59126` failure mode). Scoring
    candidates directly on the thing the downstream pipeline actually cares
    about (round, star-shaped-ish slices) is more robust than any single
    extents-based rule.

    Falls back to the longest-extent order if every candidate axis fails to
    slice cleanly (e.g. too few closed loops), so this never raises where
    the old heuristic would have succeeded.
    """
    candidates = [
        [1, 2, 0],    # x -> z
        [0, 2, 1],    # y -> z
        [0, 1, 2],    # z -> z (identity, i.e. "no change")
    ]
    best_order, best_score = None, np.inf
    for order in candidates:
        m2 = _apply_axis_order(mesh, order)
        sl, reason = slice_mesh(m2, N=N_probe, margin=margin)
        if sl is None:
            continue
        contours, _ = sl
        score = float(np.mean([_elongation(c) for c in contours]))
        if score < best_score:
            best_score, best_order = score, order
    if best_order is None:                            # nothing sliced cleanly
        return list(np.argsort(mesh.extents)), None
    return best_order, best_score


def load_and_normalize(path: str, align_axis: bool = True,
                       axis_select: str = "longest", N_probe: int = 9):
    """Load, center, scale to unit max-extent, and choose a slicing axis
    (a proper rotation, never a reflection).

    axis_select:
      "longest" (default, original behaviour) -- stand the longest bounding-
          box extent up along +z. Right for elongated objects.
      "search" -- try all 3 axes, slice a probe set with `_best_slicing_axis`,
          keep whichever gives the roundest cross-sections (lowest mean
          `_elongation`). Right for flattish disk/washer/lens-shaped objects
          where "longest extent" is a diameter, not the thickness.
    """
    _require_trimesh()
    mesh = trimesh.load(path, force="mesh")
    if not mesh.is_watertight:
        raise ValueError(f"{path}: not watertight")
    mesh.apply_translation(-mesh.bounding_box.centroid)
    ext = mesh.extents
    mesh.apply_scale(1.0 / max(ext))
    if align_axis:
        if axis_select == "search":
            order, _ = _best_slicing_axis(mesh, N_probe=N_probe)
        else:
            order = list(np.argsort(mesh.extents))     # shortest .. longest
        if list(order) != [0, 1, 2]:
            M = _rotation_for_order(order)
            T = np.eye(4)
            T[:3, :3] = M
            mesh.apply_transform(T)
    return mesh


def slice_mesh(mesh, N=7, margin=0.06, min_pts=16):
    """Return (contours [(K_i, 2)], Z (N,)) or None (with a reason string)
    if the mesh is unsuitable.  Returns (result, reason)."""
    _require_trimesh()
    zmin, zmax = mesh.bounds[0][2], mesh.bounds[1][2]
    span = zmax - zmin
    if span <= 0:
        return None, "degenerate z extent"
    Z = np.linspace(zmin + margin * span, zmax - margin * span, N)
    contours = []
    for z in Z:
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None:
            return None, f"empty section at z={z:.3f}"
        # Explicit in-plane frame: translate by -z so planar == world (x, y).
        T = np.eye(4)
        T[2, 3] = -z
        try:
            planar, _ = sec.to_planar(to_2D=T, check=False)
        except Exception as e:                        # noqa: BLE001
            return None, f"to_planar failed at z={z:.3f}: {type(e).__name__}"
        polys = planar.polygons_full
        if len(polys) != 1:
            return None, f"{len(polys)} loops at z={z:.3f} (need exactly 1)"
        poly = polys[0]
        if len(poly.interiors) > 0:
            return None, f"slice has hole(s) at z={z:.3f}"
        C = np.asarray(poly.exterior.coords)[:-1]     # drop repeated point
        if C.shape[0] < min_pts:
            return None, f"only {C.shape[0]} points at z={z:.3f}"
        contours.append(C)
    return (contours, Z), "ok"


def ground_truth_sample(mesh, n=60000):
    _require_trimesh()
    pts, face_idx = trimesh.sample.sample_surface(mesh, n)
    normals = mesh.face_normals[face_idx]
    return np.asarray(pts), np.asarray(normals)


def make_sample_from_mesh(path: str, N=7, base_circular=True,
                          crown_circular=True, n_gt=60000,
                          axis_select: str = "longest"):
    """Real-mesh sample in the SAME dict schema as the synthetic generator.
    Returns (sample, reason); sample is None if the mesh was rejected.

    closed_top=True: both ends come from margin-offset slices of a
    watertight mesh, so a real crown ring always exists (unlike the
    synthetic open_top family, which has no crown data at all).
    base_circular/crown_circular default to the circular-cap formula; there
    is no way to infer the right convention for real data a priori, so it
    is exposed as a script option.

    axis_select="search" tries all 3 candidate slicing axes and keeps the
    one giving the roundest cross-sections (see `_best_slicing_axis`);
    fixes disk/washer/lens-shaped objects that the default "longest extent
    -> z" heuristic slices the wrong way (thin, plank-like contours).
    """
    mesh = load_and_normalize(path, axis_select=axis_select, N_probe=N)
    sl, reason = slice_mesh(mesh, N=N)
    if sl is None:
        return None, reason
    contours, Z = sl
    gt_pts, gt_normals = ground_truth_sample(mesh, n=n_gt)
    return {"contours": contours, "Z": Z,
            "gt_pts": gt_pts, "gt_normals": gt_normals, "path": path,
            "family": "mesh", "closed_top": True,
            "base_circular": base_circular,
            "crown_circular": crown_circular}, "ok"
