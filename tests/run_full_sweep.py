"""Run the standard NSSR-V2 synthetic experiment sweep (max 100 epochs).

This script deliberately uses subprocess calls to the public CLI scripts so
the experiment is reproducible from the same commands used manually.

Default sweep:
- N = 5, 7, 9, 15
- legacy-compatible baseline model
- V2 Jacobian+curvature model
- evaluate + validate each
- cap ablation at N=9
- global-constants ablation at N=7

Usage:
    python scripts/run_full_sweep.py --epochs 100
"""

import argparse
import os
import subprocess
import sys


def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--Ns", type=int, nargs="+", default=[5, 7, 9, 15])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--lam_jacobian", type=float, default=1e-3)
    ap.add_argument("--lam_curvature", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=20)
    a = ap.parse_args()

    if a.epochs > 100:
        raise ValueError("This sweep is intentionally capped at 100 epochs.")

    os.makedirs(a.runs, exist_ok=True)
    os.makedirs(a.results, exist_ok=True)

    py = sys.executable

    for N in a.Ns:
        base = os.path.join(a.runs, f"N{N}_baseline")
        geom = os.path.join(a.runs, f"N{N}_geom")

        run([
            py, "scripts/train_model.py",
            "--data", a.data,
            "--N", str(N),
            "--epochs", str(a.epochs),
            "--out", base,
            "--val_every", "5",
            "--patience", str(a.patience),
            "--surf_sub", "8000",
            "--gt_sub", "8000",
        ])

        run([
            py, "scripts/evaluate.py",
            "--data", a.data,
            "--N", str(N),
            "--ckpt", os.path.join(base, "best.pt"),
            "--out", os.path.join(a.results, f"eval_N{N}_baseline.csv"),
        ])

        run([
            py, "scripts/validate.py",
            "--data", a.data,
            "--split", "test",
            "--N", str(N),
            "--ckpt", os.path.join(base, "best.pt"),
            "--out", os.path.join(a.results, f"validate_N{N}_baseline.csv"),
        ])

        run([
            py, "scripts/train_model.py",
            "--data", a.data,
            "--N", str(N),
            "--epochs", str(a.epochs),
            "--out", geom,
            "--val_every", "5",
            "--patience", str(a.patience),
            "--surf_sub", "8000",
            "--gt_sub", "8000",
            "--lam_jacobian", str(a.lam_jacobian),
            "--lam_curvature", str(a.lam_curvature),
        ])

        run([
            py, "scripts/evaluate.py",
            "--data", a.data,
            "--N", str(N),
            "--ckpt", os.path.join(geom, "best.pt"),
            "--out", os.path.join(a.results, f"eval_N{N}_geom.csv"),
        ])

        run([
            py, "scripts/validate.py",
            "--data", a.data,
            "--split", "test",
            "--N", str(N),
            "--ckpt", os.path.join(geom, "best.pt"),
            "--out", os.path.join(a.results, f"validate_N{N}_geom.csv"),
        ])

    if 9 in a.Ns:
        run([
            py, "scripts/ablation_caps.py",
            "--data", a.data,
            "--N", "9",
            "--epochs", str(a.epochs),
            "--runs_dir", os.path.join(a.runs, "caps_ablation"),
            "--out", os.path.join(a.results, "ablation_caps_N9.csv"),
        ])

    if 7 in a.Ns:
        run([
            py, "scripts/ablation_global_constants.py",
            "--data", a.data,
            "--N", "7",
            "--ckpt", os.path.join(a.runs, "N7_geom", "best.pt"),
            "--out", os.path.join(a.results, "ablation_global_N7.csv"),
        ])

    print("\nFULL SWEEP COMPLETE")


if __name__ == "__main__":
    main()
