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


def load_and_normalize(path: str, align_axis: bool = True):
    """Load, center, scale to unit max-extent, and stand the object's
    longest axis up along +z (a proper rotation, never a reflection)."""
    _require_trimesh()
    mesh = trimesh.load(path, force="mesh")
    if not mesh.is_watertight:
        raise ValueError(f"{path}: not watertight")
    mesh.apply_translation(-mesh.bounding_box.centroid)
    ext = mesh.extents
    mesh.apply_scale(1.0 / max(ext))
    if align_axis:
        order = np.argsort(mesh.extents)              # shortest .. longest
        if order[-1] != 2:
            M = np.eye(3)[:, order]                   # columns permuted
            if np.linalg.det(M) < 0:                  # fix reflection
                M[:, 0] = -M[:, 0]
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
                          crown_circular=True, n_gt=60000):
    """Real-mesh sample in the SAME dict schema as the synthetic generator.
    Returns (sample, reason); sample is None if the mesh was rejected.

    closed_top=True: both ends come from margin-offset slices of a
    watertight mesh, so a real crown ring always exists (unlike the
    synthetic open_top family, which has no crown data at all).
    base_circular/crown_circular default to the circular-cap formula; there
    is no way to infer the right convention for real data a priori, so it
    is exposed as a script option.
    """
    mesh = load_and_normalize(path)
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
