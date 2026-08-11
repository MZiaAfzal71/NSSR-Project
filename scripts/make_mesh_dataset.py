"""Build paper-grade real-mesh train/val/test datasets.

Key difference from the earlier builder:
  split membership is assigned deterministically from the mesh path, so an
  object that survives at multiple slice counts N stays in the SAME split
  across N.  This makes cross-N comparisons much cleaner.

A CSV manifest records every mesh/N outcome and rejection reason.
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, os, pickle, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path

from nssr.slicing import make_sample_from_mesh

MESH_EXTS = (".obj", ".ply", ".stl", ".off")


def find_meshes(root):
    paths = []
    for ext in MESH_EXTS:
        paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    return sorted(paths)


def stable_u01(path, seed):
    key = f"{seed}|{os.path.abspath(path)}".encode("utf-8")
    h = hashlib.sha256(key).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def split_for(path, seed, val_frac, test_frac):
    u = stable_u01(path, seed)
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--meshes", required=True)
    ap.add_argument("--N", type=int, nargs="+", default=[15])
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--axis_select", choices=["longest", "search"], default="search")
    ap.add_argument("--base_circular", action="store_true", default=True)
    ap.add_argument("--no_base_circular", dest="base_circular", action="store_false")
    ap.add_argument("--crown_circular", action="store_true", default=True)
    ap.add_argument("--no_crown_circular", dest="crown_circular", action="store_false")
    a = ap.parse_args()

    if a.val_frac < 0 or a.test_frac < 0 or a.val_frac + a.test_frac >= 1:
        raise SystemExit("require val_frac>=0, test_frac>=0, val_frac+test_frac<1")

    paths = find_meshes(a.meshes)
    if not paths:
        raise SystemExit(f"no meshes found under {a.meshes}")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    manifest = []

    for N in sorted(set(a.N)):
        splits = {"train": [], "val": [], "test": []}
        rejected = {}

        for p in paths:
            assigned = split_for(p, a.seed, a.val_frac, a.test_frac)
            try:
                s, reason = make_sample_from_mesh(
                    p, N=N,
                    base_circular=a.base_circular,
                    crown_circular=a.crown_circular,
                    axis_select=a.axis_select,
                )
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"[:120]
                s = None

            if s is None:
                reason = (reason or "unknown rejection").split(" at ")[0]
                rejected[reason] = rejected.get(reason, 0) + 1
                manifest.append({
                    "N": N, "path": p, "assigned_split": assigned,
                    "kept": 0, "reason": reason,
                })
                continue

            splits[assigned].append(s)
            manifest.append({
                "N": N, "path": p, "assigned_split": assigned,
                "kept": 1, "reason": "",
            })

        print(f"\n--- N={N} ---")
        print(f"kept {sum(map(len,splits.values()))}/{len(paths)}")
        for k in ("train","val","test"):
            print(f"  {k:5s}: {len(splits[k])}")
        for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
            print(f"  rejected {count:4d}: {reason}")

        if any(len(splits[k]) == 0 for k in splits):
            print("  WARNING: at least one split is empty; increase mesh corpus size")

        for split, samples in splits.items():
            outp = os.path.join(a.out, f"{split}_N{N}.pkl")
            with open(outp, "wb") as f:
                pickle.dump(samples, f)
            print("  wrote", outp)

    manifest_path = os.path.join(a.out, "mesh_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["N","path","assigned_split","kept","reason"]
        )
        w.writeheader()
        w.writerows(manifest)
    print("\nwrote", manifest_path)


if __name__ == "__main__":
    main()
