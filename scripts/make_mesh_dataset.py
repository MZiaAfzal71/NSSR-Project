"""Build train/val/test pickles from a directory of real meshes.

Mirrors scripts/make_synthetic_dataset.py's output format exactly (same
sample dict schema, via nssr.slicing.make_sample_from_mesh), so
scripts/train_model.py and scripts/evaluate.py work on real data with NO
changes.

Candidate sources (see docs/RESULTS_AND_TUNING.md "Real-world data" section
for the full guide):
  - Thingi10k (https://ten-thousand-models.appspot.com) -- filter watertight
  - ShapeNetCore: bottle / vase / jar / can / mug categories
  - Your own photogrammetry / structured-light scans of fruit or pottery,
    exported as a watertight OBJ/PLY/STL

Selection policy (matches nssr/slicing.py): a mesh is KEPT only if it is
watertight AND every one of the N slices yields a single closed loop
(genus-0, star-shaped-ish -- the same object class the classical pipeline
assumes). Rejected meshes are counted and reported by reason, which is
worth quoting directly in the paper's dataset-construction paragraph.

Usage:
    python scripts/make_mesh_dataset.py --meshes data/meshes --N 7 \
        --out data/real --val_frac 0.15 --test_frac 0.15
"""
import sys, os, argparse, pickle, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from nssr.slicing import make_sample_from_mesh

MESH_EXTS = (".obj", ".ply", ".stl", ".off")


def find_meshes(root):
    paths = []
    for ext in MESH_EXTS:
        paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"),
                               recursive=True))
    return sorted(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", required=True,
                    help="directory to scan recursively for mesh files")
    ap.add_argument("--N", type=int, nargs="+", default=[7],
                    help="slice count(s) to generate, e.g. --N 5 7 9 15")
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--base_circular", action="store_true", default=True)
    ap.add_argument("--no_base_circular", dest="base_circular",
                    action="store_false")
    ap.add_argument("--crown_circular", action="store_true", default=True)
    ap.add_argument("--no_crown_circular", dest="crown_circular",
                    action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--axis_select", choices=["longest", "search"],
                    default="longest",
                    help="'longest': stand the longest bbox extent up "
                         "along z (original heuristic, right for elongated "
                         "objects). 'search': try all 3 axes and keep the "
                         "one giving the roundest cross-sections (fixes "
                         "disk/washer/lens-shaped objects -- see "
                         "nssr.slicing._best_slicing_axis).")
    a = ap.parse_args()

    try:
        import trimesh  # noqa: F401
    except ImportError:
        print("ERROR: trimesh is required for this script "
              "(pip install trimesh shapely --break-system-packages)")
        return 1

    paths = find_meshes(a.meshes)
    if not paths:
        print(f"No mesh files found under {a.meshes} "
              f"(looked for {MESH_EXTS})")
        return 1
    print(f"found {len(paths)} candidate mesh files")

    os.makedirs(a.out, exist_ok=True)
    for N in a.N:
        kept, rejected = [], {}
        for p in paths:
            try:
                s, reason = make_sample_from_mesh(
                    p, N=N, base_circular=a.base_circular,
                    crown_circular=a.crown_circular,
                    axis_select=a.axis_select)
            except Exception as e:                     # noqa: BLE001
                rejected.setdefault(f"{type(e).__name__}: {e}"[:70],
                                    []).append(p)
                continue
            if s is None:
                # normalize "... at z=0.123" variants into one bucket
                rejected.setdefault(reason.split(" at ")[0], []).append(p)
                continue
            kept.append(s)

        print(f"\n--- N={N} ---")
        print(f"kept: {len(kept)} / {len(paths)}")
        for reason, plist in rejected.items():
            print(f"  rejected ({reason}): {len(plist)}")
            for p in plist[:3]:
                print(f"    e.g. {p}")

        if len(kept) < 3:
            print(f"Too few usable meshes at N={N} to split into "
                  f"train/val/test ({len(kept)} kept) -- skipping.")
            continue

        rng = np.random.default_rng(a.seed)
        idx = rng.permutation(len(kept))
        n_val = max(1, int(len(kept) * a.val_frac))
        n_test = max(1, int(len(kept) * a.test_frac))
        n_train = len(kept) - n_val - n_test
        if n_train < 1:
            print(f"val_frac + test_frac too large for {len(kept)} "
                  f"kept meshes at N={N} -- skipping.")
            continue

        splits = {
            "train": [kept[i] for i in idx[:n_train]],
            "val": [kept[i] for i in idx[n_train:n_train + n_val]],
            "test": [kept[i] for i in idx[n_train + n_val:]],
        }
        for split, samples in splits.items():
            path = os.path.join(a.out, f"{split}_N{N}.pkl")
            with open(path, "wb") as f:
                pickle.dump(samples, f)
            print(f"wrote {path}  ({len(samples)} samples)")

    print("\nUse these exactly like the synthetic ones, e.g.:")
    print(f"  python scripts/train_model.py --data {a.out} --N 7 ...")
    print(f"  python scripts/evaluate.py --data {a.out} --N 7 ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
