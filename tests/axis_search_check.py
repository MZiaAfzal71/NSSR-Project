"""Regression check for nssr.slicing._best_slicing_axis / _elongation.

Two claims to verify:
  1. On elongated objects (sphere, vase, apple, banana -- the existing
     test corpus), axis_select="search" agrees with the original
     axis_select="longest" heuristic: no behaviour change where the old
     rule already worked.
  2. On a disk/washer-shaped object (longest extent = a diameter, not the
     thickness -- the `thingi_59126` failure mode from the real-data run),
     "search" finds a materially rounder slicing axis than "longest".

Requires trimesh (skips with a clear message if unavailable, same
convention as the rest of the real-mesh test path).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None


def _make_disk(radius=1.0, thickness_frac=0.15, subdivisions=3):
    """Flattened sphere: coin/washer-like, longest extent is a diameter."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh.vertices[:, 2] *= thickness_frac
    mesh.fix_normals()
    return mesh


def main():
    if trimesh is None:
        print("axis_search_check: SKIPPED (trimesh not installed)")
        return 0

    from nssr.slicing import load_and_normalize, slice_mesh, _elongation

    # 1. elongated objects: search should not change the slicing axis choice
    #    (same axis -> same order -> ~identical elongation to `longest`)
    import scripts.make_test_meshes as mtm
    ok = True
    for name in mtm.PROFILES:
        V, F = mtm.build_mesh(name)
        path = f"/tmp/_axis_check_{name}.obj"
        mtm.write_obj(path, V, F)
        elong = {}
        for mode in ("longest", "search"):
            mesh = load_and_normalize(path, axis_select=mode, N_probe=9)
            sl, reason = slice_mesh(mesh, N=9)
            assert sl is not None, f"{name}/{mode}: {reason}"
            contours, _ = sl
            elong[mode] = float(np.mean([_elongation(c) for c in contours]))
        agree = abs(elong["longest"] - elong["search"]) < 0.05
        ok &= agree
        print(f"  {name:11s} longest={elong['longest']:.3f} "
              f"search={elong['search']:.3f}  "
              f"{'OK (unchanged)' if agree else '** DIVERGED **'}")

    # 2. disk-shaped object: search must find a materially rounder axis
    disk = _make_disk()
    disk_path = "/tmp/_axis_check_disk.obj"
    disk.export(disk_path)
    elong = {}
    for mode in ("longest", "search"):
        mesh = load_and_normalize(disk_path, axis_select=mode, N_probe=9)
        sl, reason = slice_mesh(mesh, N=9)
        assert sl is not None, f"disk/{mode}: {reason}"
        contours, _ = sl
        elong[mode] = float(np.mean([_elongation(c) for c in contours]))
    fixed = elong["search"] < 2.0 and elong["longest"] > 4.0
    ok &= fixed
    print(f"  disk        longest={elong['longest']:.3f} "
          f"search={elong['search']:.3f}  "
          f"{'OK (fixed)' if fixed else '** NOT FIXED **'}")

    print(f"\nAXIS SEARCH CHECK: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
