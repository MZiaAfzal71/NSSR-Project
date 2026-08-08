"""Validate the real-mesh pipeline on your machine (V2-compatible).

For each mesh:
load -> normalize/align -> slice -> preprocess -> classical reconstruct,
reporting where failures happen and how good the classical baseline is.

This V2 port uses nssr.geometry (Torch) instead of the legacy geometry_np path.
"""
import sys, os, argparse, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

MESH_EXTS = (".obj", ".ply", ".stl", ".off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", required=True)
    ap.add_argument("--N", type=int, default=9)
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--axis_select",
        choices=["longest", "search"],
        default="longest",
    )
    ap.add_argument("--fp64", action="store_true")
    a = ap.parse_args()

    try:
        import trimesh  # noqa: F401
    except ImportError:
        print("ERROR: needs trimesh -> pip install trimesh shapely")
        return 1

    from nssr.slicing import load_and_normalize, slice_mesh, _elongation, ground_truth_sample
    from nssr.preprocess import preprocess_object
    from nssr.geometry import hermite_surface, zero_params

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float64 if a.fp64 else torch.float32

    paths = []
    for ext in MESH_EXTS:
        paths.extend(glob.glob(os.path.join(a.meshes, "**", f"*{ext}"), recursive=True))
    paths = sorted(paths)
    if a.limit:
        paths = paths[:a.limit]
    if not paths:
        print(f"no meshes found under {a.meshes}")
        return 1

    print(f"checking {len(paths)} mesh(es) at N={a.N}\\n")
    kept, reasons = 0, {}

    for p in paths:
        name = os.path.basename(p)
        try:
            mesh = load_and_normalize(p, axis_select=a.axis_select, N_probe=a.N)
        except Exception as e:
            print(f"[FAIL] {name:16s} load/normalize: {e}")
            reasons.setdefault(str(e)[:60], 0)
            reasons[str(e)[:60]] += 1
            continue

        vol_sign = "ok" if mesh.volume > 0 else "** NEGATIVE VOLUME (mirrored) **"

        sl, reason = slice_mesh(mesh, N=a.N)
        if sl is None:
            print(f"[skip] {name:16s} {reason}")
            key = reason.split(" at ")[0]
            reasons.setdefault(key, 0)
            reasons[key] += 1
            continue

        contours, Z = sl
        npts = [len(c) for c in contours]
        elong = np.mean([_elongation(c) for c in contours])
        elong_flag = (
            "  ** elongated cross-sections -- try --axis_select search **"
            if elong > 2.0 else ""
        )

        pre = preprocess_object(contours, Z, m=a.m)

        T = lambda x: torch.as_tensor(np.asarray(x), device=dev, dtype=dt)
        R = T(pre["R"])
        Zt = T(pre["Z"])
        RB = T(pre["RB"])
        RC = T(pre["RC"])
        Bh = T(pre["Bh"])
        Th = T(pre["Th"])

        p0 = zero_params(R.shape[0], R.shape[1], device=dev, dtype=dt)

        with torch.no_grad():
            S = hermite_surface(
                R, Zt, RB, RC, Bh, Th, p0,
                n_u=16,
                closed_top=pre["closed_top"],
                base_circular=pre["base_circular"],
                crown_circular=pre["crown_circular"],
            )

        Sn = S.detach().cpu().numpy()
        finite = np.isfinite(Sn).all()

        gt, _ = ground_truth_sample(mesh, n=20000)
        nrm = pre["norm"]
        gtn = (
            gt - np.array([*nrm["center_xy"], nrm["zmid"]])
        ) / nrm["scale"]

        pred = Sn.reshape(-1, 3)
        sub = pred[
            np.random.default_rng(0).choice(
                len(pred),
                min(4000, len(pred)),
                replace=False,
            )
        ]
        d = np.linalg.norm(
            sub[:, None, :] - gtn[None, ::4, :],
            axis=-1,
        ).min(1)

        kept += 1

        print(
            f"[ OK ] {name:16s} volume {vol_sign} | slice pts "
            f"{min(npts)}-{max(npts)} | Bh={pre['Bh']:+.3f} "
            f"Th={pre['Th']:+.3f} | finite={finite} | "
            f"mean elongation={elong:.2f} | "
            f"classical mean surface err={d.mean():.4f}{elong_flag}"
        )

    print(f"\\nkept {kept}/{len(paths)}")
    if reasons:
        print("rejection reasons:")
        for r, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {c:4d}  {r}")

    print("\\nIf the kept rate looks reasonable, build the dataset with:")
    print(
        f"  python scripts/make_mesh_dataset.py --meshes {a.meshes} "
        f"--N {a.N} --out data/real"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
