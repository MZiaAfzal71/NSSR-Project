"""Paper-oriented real/mixed-domain NSSR-V2 experiment pipeline.

Stages
------
real:
  - optional mesh pipeline check
  - optional real pickle construction
  - real-only 100-epoch sweep
transfer:
  - evaluate synthetic-trained checkpoints on untouched real test sets
mixed:
  - build domain-balanced synthetic+real train/val
  - train mixed checkpoints once
  - evaluate each mixed checkpoint separately on real and synthetic test sets
figures:
  - create publication/supplementary real-object figures
collect:
  - write a compact domain-comparison CSV

This script assumes the final shared scripts are installed:
  scripts/run_full_sweep.py
  scripts/validate.py
  scripts/evaluate_projection.py
  scripts/visualize_real.py
"""
from __future__ import annotations
import argparse, csv, glob, os, re, shlex, subprocess, sys
from pathlib import Path
import numpy as np

N_RE = re.compile(r"(?:train|test)_N(\d+)\.pkl$")


def run(cmd, dry=False):
    print("\n>>>", shlex.join([str(x) for x in cmd]), flush=True)
    if not dry:
        subprocess.run([str(x) for x in cmd], check=True)


def discover_Ns(data):
    Ns = []
    for p in glob.glob(os.path.join(data, "train_N*.pkl")):
        m = N_RE.search(os.path.basename(p))
        if m:
            N = int(m.group(1))
            if all(os.path.exists(os.path.join(data, f"{s}_N{N}.pkl"))
                   for s in ("train", "val", "test")):
                Ns.append(N)
    return sorted(set(Ns))


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean(rows, key):
    vals = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else float("nan")


def eval_ckpt(py, data, N, ckpt, result_prefix, a):
    val = result_prefix + "_validate.csv"
    proj = result_prefix + "_projection.csv"
    run([
        py, "scripts/validate.py",
        "--data", data, "--split", "test", "--N", N,
        "--ckpt", ckpt, "--m", a.m, "--n_u", a.eval_n_u,
        "--c_bound", a.c_bound, "--max_cap_fold", a.max_cap_fold,
        "--out", val,
    ], a.dry_run)
    run([
        py, "scripts/evaluate_projection.py",
        "--data", data, "--split", "test", "--N", N,
        "--ckpt", ckpt, "--m", a.m, "--n_u", a.eval_n_u,
        "--c_bound", a.c_bound, "--max_cap_fold", a.max_cap_fold,
        "--projection_mode", "staged", "--out", proj,
    ], a.dry_run)
    return val, proj


def collect_domain(rowspecs, out):
    rows_out = []
    for domain, train_domain, N, val_path, proj_path in rowspecs:
        if not os.path.exists(val_path) or not os.path.exists(proj_path):
            continue
        vr, pr = read_csv(val_path), read_csv(proj_path)
        row = {
            "train_domain": train_domain,
            "test_domain": domain,
            "N": N,
            "classical_chamfer_l2": mean(vr, "classical_chamfer_l2"),
            "learned_chamfer_l2": mean(vr, "learned_chamfer_l2"),
            "raw_j_valid_rate": mean(vr, "learned_geom_jacobian_valid"),
            "raw_cap_safe_rate": mean(vr, "learned_geom_cap_safe"),
            "raw_safe_rate": mean(vr, "learned_geom_safe"),
            "post_j_valid_rate": mean(pr, "post_j_valid"),
            "post_cap_safe_rate": mean(pr, "post_cap_safe"),
            "post_safe_rate": mean(pr, "post_safe"),
            "projection_activation_rate": mean(pr, "projection_activated"),
            "post_chamfer_l2": mean(pr, "post_chamfer_l2"),
            "raw_chamfer_l2_projection_eval": mean(pr, "raw_chamfer_l2"),
            "mean_projection_delta": mean(pr, "delta_chamfer_l2"),
        }
        for stage in ("cap_all", "tangent", "all"):
            sr = [r for r in pr if r.get("projection_stage") == stage]
            row[f"{stage}_count"] = len(sr)
            row[f"{stage}_alpha_mean"] = mean(sr, "alpha") if sr else float("nan")
        rows_out.append(row)

    if not rows_out:
        return
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows_out[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--meshes", default="data/meshes")
    ap.add_argument("--real", default="data/real")
    ap.add_argument("--synthetic", default="data/synthetic")
    ap.add_argument("--mixed", default="data/mixed")
    ap.add_argument("--Ns", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--train_n_u", type=int, default=12)
    ap.add_argument("--eval_n_u", type=int, default=16)
    ap.add_argument("--max_cap_fold", type=float, default=1e-3)
    ap.add_argument("--c_bound", type=float, default=1.0)
    ap.add_argument("--axis_select", choices=["longest", "search"],
                    default="search")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--real_runs", default="runs/paper_real_100ep")
    ap.add_argument("--mixed_runs", default="runs/paper_mixed_100ep")
    ap.add_argument("--results", default="results/paper_real")
    ap.add_argument("--synthetic_runs", default="runs/paper_full_100ep")
    ap.add_argument("--build_real", action="store_true")
    ap.add_argument("--check_meshes", action="store_true")
    ap.add_argument("--skip_real_train", action="store_true")
    ap.add_argument("--skip_transfer", action="store_true")
    ap.add_argument("--skip_mixed", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--figure_n", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    if not (1 <= a.epochs <= 100):
        raise SystemExit("--epochs must be 1..100")

    py = sys.executable
    Path(a.results).mkdir(parents=True, exist_ok=True)

    # Build or discover real datasets.
    Ns = sorted(set(a.Ns or discover_Ns(a.real)))
    if a.build_real:
        if not a.Ns:
            raise SystemExit("--build_real requires explicit --Ns")
        if a.check_meshes:
            for N in a.Ns:
                run([
                    py, "scripts/check_mesh_pipeline.py",
                    "--meshes", a.meshes, "--N", N, "--m", a.m,
                    "--axis_select", a.axis_select,
                ], a.dry_run)
        run([
            py, "scripts/make_mesh_dataset.py",
            "--meshes", a.meshes, "--N", *a.Ns,
            "--out", a.real,
            "--val_frac", a.val_frac, "--test_frac", a.test_frac,
            "--axis_select", a.axis_select, "--seed", 0,
        ], a.dry_run)
        Ns = sorted(set(a.Ns))

    if not Ns:
        raise SystemExit(
            "No complete real datasets found. Use --build_real --Ns ..."
        )

    print("Real paper Ns:", Ns)

    # 1) Real-only training/evaluation.
    if not a.skip_real_train:
        run([
            py, "scripts/run_full_sweep.py",
            "--data", a.real,
            "--Ns", *Ns,
            "--epochs", a.epochs,
            "--val_every", a.val_every,
            "--val_subset", 1000000,  # all available validation objects
            "--m", a.m,
            "--train_n_u", a.train_n_u,
            "--eval_n_u", a.eval_n_u,
            "--surf_sub", 2000,
            "--gt_sub", 2000,
            "--lam_jacobian", 0.01,
            "--jacobian_margin", 0.05,
            "--jacobian_power", 2,
            "--lam_cap_fold", 0.005,
            "--cap_fold_margin", 1e-3,
            "--cap_fold_power", 2,
            "--geometry_topk", 0.05,
            "--max_cap_fold", a.max_cap_fold,
            "--c_bound", a.c_bound,
            "--runs", a.real_runs,
            "--results", os.path.join(a.results, "real_only"),
        ], a.dry_run)

    rowspecs = []

    # Register real-only outputs.
    for N in Ns:
        rowspecs.append((
            "real", "real", N,
            os.path.join(a.results, "real_only", f"N{N}_validate.csv"),
            os.path.join(a.results, "real_only", f"N{N}_projection.csv"),
        ))

    # 2) Synthetic-trained -> real zero-shot transfer.
    if not a.skip_transfer:
        transfer_dir = os.path.join(a.results, "synthetic_to_real")
        Path(transfer_dir).mkdir(parents=True, exist_ok=True)
        for N in Ns:
            ckpt = os.path.join(a.synthetic_runs, f"N{N}", "best.pt")
            if not a.dry_run and not os.path.isfile(ckpt):
                print(f"[skip transfer N={N}] missing {ckpt}")
                continue
            prefix = os.path.join(transfer_dir, f"N{N}")
            v, p = eval_ckpt(py, a.real, N, ckpt, prefix, a)
            rowspecs.append(("real", "synthetic", N, v, p))

    # 3) Domain-balanced mixed training.
    if not a.skip_mixed:
        run([
            py, "scripts/make_mixed_dataset.py",
            "--synthetic", a.synthetic,
            "--real", a.real,
            "--out", a.mixed,
            "--Ns", *Ns,
            "--seed", 0,
        ], a.dry_run)

        run([
            py, "scripts/run_full_sweep.py",
            "--data", a.mixed,
            "--Ns", *Ns,
            "--epochs", a.epochs,
            "--val_every", a.val_every,
            "--val_subset", 1000000,
            "--m", a.m,
            "--train_n_u", a.train_n_u,
            "--eval_n_u", a.eval_n_u,
            "--surf_sub", 2000,
            "--gt_sub", 2000,
            "--lam_jacobian", 0.01,
            "--jacobian_margin", 0.05,
            "--jacobian_power", 2,
            "--lam_cap_fold", 0.005,
            "--cap_fold_margin", 1e-3,
            "--cap_fold_power", 2,
            "--geometry_topk", 0.05,
            "--max_cap_fold", a.max_cap_fold,
            "--c_bound", a.c_bound,
            "--runs", a.mixed_runs,
            "--results", os.path.join(a.results, "mixed_diagnostic"),
        ], a.dry_run)

        for N in Ns:
            ckpt = os.path.join(a.mixed_runs, f"N{N}", "best.pt")
            for test_domain, data in (
                ("real", a.real), ("synthetic", a.synthetic)
            ):
                d = os.path.join(a.results, f"mixed_to_{test_domain}")
                Path(d).mkdir(parents=True, exist_ok=True)
                prefix = os.path.join(d, f"N{N}")
                v, p = eval_ckpt(py, data, N, ckpt, prefix, a)
                rowspecs.append((test_domain, "mixed", N, v, p))

    # 4) Real figures from real-only checkpoint.
    if a.figures:
        fig_dir = os.path.join(a.results, "figures")
        for N in Ns:
            ckpt = os.path.join(a.real_runs, f"N{N}", "best.pt")
            if not a.dry_run and not os.path.isfile(ckpt):
                continue
            run([
                py, "scripts/visualize_real.py",
                "--data", a.real, "--N", N, "--m", a.m,
                "--n_u", a.eval_n_u, "--ckpt", ckpt,
                "--n", a.figure_n,
                "--out", os.path.join(fig_dir, f"N{N}"),
                "--project_safe", "--pole_zoom", "--surface_render",
            ], a.dry_run)

    if not a.dry_run:
        collect_domain(
            rowspecs,
            os.path.join(a.results, "domain_comparison.csv")
        )

    print("\nREAL/MIXED PAPER PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
