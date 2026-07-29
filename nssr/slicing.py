"""Slice real watertight meshes into cross-sectional contours (Phase 1b).

Sources to populate data/meshes/ :
  - Thingi10k (https://ten-thousand-models.appspot.com) — filter watertight
  - ShapeNetCore categories: bottle, vase, jar, can, mug
  - COSEG vases; your own scans of fruit / pottery

Selection policy for v1: keep only meshes for which EVERY slice yields a
single closed loop (genus-0, star-shaped-ish objects — same object class
as the classical pipeline assumes).  Multi-loop slices are a documented
limitation / future-work item, exactly as in the classical literature.
"""
from __future__ import annotations
import numpy as np

try:
    import trimesh
except ImportError:                                   # pragma: no cover
    trimesh = None


def load_and_normalize(path: str):
    mesh = trimesh.load(path, force="mesh")
    if not mesh.is_watertight:
        raise ValueError(f"{path}: not watertight")
    mesh.apply_translation(-mesh.bounding_box.centroid)
    mesh.apply_scale(1.0 / max(mesh.extents))
    # align longest axis with z
    order = np.argsort(mesh.extents)
    if order[-1] != 2:
        axes = np.eye(3)
        Tm = np.eye(4)
        Tm[:3, :3] = np.stack([axes[order[0]], axes[order[1]],
                               axes[order[-1]]], axis=1)
        mesh.apply_transform(Tm)
    return mesh


def slice_mesh(mesh, N=7, margin=0.06):
    """Return (contours [(K_i, 2)], Z (N,)) or None if any slice is not a
    single closed loop."""
    zmin, zmax = mesh.bounds[0][2], mesh.bounds[1][2]
    span = zmax - zmin
    Z = np.linspace(zmin + margin * span, zmax - margin * span, N)
    contours = []
    for z in Z:
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None:
            return None
        planar, _ = sec.to_2D()
        polys = planar.polygons_full
        if len(polys) != 1:
            return None
        C = np.array(polys[0].exterior.coords)[:-1]   # drop repeated point
        contours.append(C)
    return contours, Z


def ground_truth_sample(mesh, n=60000):
    pts, face_idx = trimesh.sample.sample_surface(mesh, n)
    normals = mesh.face_normals[face_idx]
    return np.asarray(pts), np.asarray(normals)


def make_sample_from_mesh(path: str, N=7, base_circular=True,
                          crown_circular=True):
    """Real-mesh sample, closed_top=True (both ends come from margin-offset
    slices of a watertight mesh, so a real crown ring always exists --
    unlike the synthetic open_top family, which has no crown data at all).
    base_circular/crown_circular default to the more expressive circular-
    cap formula; pass False to match the apple's non-circular convention
    if that fits your data better (no way to know a priori for real data,
    so this is exposed as a script option, not inferred)."""
    mesh = load_and_normalize(path)
    sl = slice_mesh(mesh, N=N)
    if sl is None:
        return None
    contours, Z = sl
    gt_pts, gt_normals = ground_truth_sample(mesh)
    return {"contours": contours, "Z": Z,
            "gt_pts": gt_pts, "gt_normals": gt_normals, "path": path,
            "family": "mesh", "closed_top": True,
            "base_circular": base_circular, "crown_circular": crown_circular}
