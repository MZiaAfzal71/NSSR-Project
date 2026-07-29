"""Validate the real-mesh pipeline on your machine (needs trimesh).

For each mesh: load -> normalize/align -> slice -> preprocess -> classical
reconstruct, reporting exactly where anything fails and how good the
classical baseline is. Run this BEFORE scraping a large mesh corpus, and
again on the first 10-20 real meshes you download, so you learn the
rejection rate early.

Usage:
    python scripts/make_test_meshes.py --out data/meshes_test
    python scripts/check_mesh_pipeline.py --meshes data/meshes_test --N 9
"""
import sys, os, argparse, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

MESH_EXTS = (".obj", ".ply", ".stl", ".off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", required=True)
    ap.add_argument("--N", type=int, default=9)
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    try:
        import trimesh  # noqa: F401
    except ImportError:
        print("ERROR: needs trimesh -> "
              "pip install trimesh shapely --break-system-packages")
        return 1

    from nssr.slicing import load_and_normalize, slice_mesh, make_sample_from_mesh
    from nssr.preprocess import preprocess_object
    from nssr.geometry_np import hermite_surface_np, zero_params_np

    paths = []
    for ext in MESH_EXTS:
        paths.extend(glob.glob(os.path.join(a.meshes, "**", f"*{ext}"),
                               recursive=True))
    paths = sorted(paths)
    if a.limit:
        paths = paths[:a.limit]
    if not paths:
        print(f"no meshes found under {a.meshes}")
        return 1
    print(f"checking {len(paths)} mesh(es) at N={a.N}\n")

    kept, reasons = 0, {}
    for p in paths:
        name = os.path.basename(p)
        try:
            mesh = load_and_normalize(p)
        except Exception as e:                        # noqa: BLE001
            print(f"[FAIL] {name:16s} load/normalize: {e}")
            reasons.setdefault(str(e)[:60], 0)
            reasons[str(e)[:60]] += 1
            continue

        # axis alignment sanity: transform must be a rotation, not a mirror
        vol_sign = "ok" if mesh.volume > 0 else "** NEGATIVE VOLUME (mirrored) **"
        sl, reason = slice_mesh(mesh, N=a.N)
        if sl is None:
            print(f"[skip] {name:16s} {reason}")
            reasons.setdefault(reason.split(" at ")[0], 0)
            reasons[reason.split(" at ")[0]] += 1
            continue
        contours, Z = sl
        npts = [len(c) for c in contours]

        pre = preprocess_object(contours, Z, m=a.m)
        p0 = zero_params_np(pre["R"].shape[0], pre["R"].shape[1])
        S = hermite_surface_np(pre["R"], pre["Z"], pre["RB"], pre["RC"],
                               pre["Bh"], pre["Th"], params=p0, n_u=16,
                               closed_top=pre["closed_top"],
                               base_circular=pre["base_circular"],
                               crown_circular=pre["crown_circular"])
        finite = np.isfinite(S).all()

        # classical accuracy vs the mesh's own surface samples
        from nssr.slicing import ground_truth_sample
        gt, _ = ground_truth_sample(mesh, n=20000)
        nrm = pre["norm"]
        gtn = (gt - np.array([*nrm["center_xy"], nrm["zmid"]])) / nrm["scale"]
        pred = S.reshape(-1, 3)
        sub = pred[np.random.default_rng(0).choice(len(pred),
                                                   min(4000, len(pred)),
                                                   replace=False)]
        d = np.linalg.norm(sub[:, None, :] - gtn[None, ::4, :], axis=-1).min(1)
        kept += 1
        print(f"[ OK ] {name:16s} volume {vol_sign} | slice pts "
              f"{min(npts)}-{max(npts)} | Bh={pre['Bh']:+.3f} "
              f"Th={pre['Th']:+.3f} | finite={finite} | "
              f"classical mean surface err={d.mean():.4f}")

    print(f"\nkept {kept}/{len(paths)}")
    if reasons:
        print("rejection reasons:")
        for r, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {c:4d}  {r}")
    print("\nIf the kept rate looks reasonable, build the dataset with:")
    print(f"  python scripts/make_mesh_dataset.py --meshes {a.meshes} "
          f"--N {a.N} --out data/real")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
