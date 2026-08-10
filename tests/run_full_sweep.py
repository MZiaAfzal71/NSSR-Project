"""Run the frozen NSSR-V2 multi-N training/safety/projection sweep.

By default, slice counts are auto-discovered from ``train_N*.pkl`` files in
``--data``.  For each N the pipeline:

1. trains the frozen safety-aware model (unless a checkpoint is reused),
2. validates raw classical/learned accuracy and sampled safety,
3. evaluates the failure-aware shared projection,
4. collects all N values into one final CSV table.

The frozen training configuration matches the finalized N=15 experiment:
    lambda_J          = 0.01
    orientation margin= 0.05
    lambda_cap        = 0.005
    cap margin/max    = 1e-3
    geometry top-k    = 0.05
    c_bound           = 1.0
    n_u train         = 12
    m                 = 128
    safety checkpoint = best.pt

Training epochs are capped at 100.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


MAX_EPOCHS = 100
TRAIN_RE = re.compile(r"train_N(\d+)\.pkl$")


def discover_Ns(data_dir: str):
    found = []
    for path in glob.glob(os.path.join(data_dir, "train_N*.pkl")):
        m = TRAIN_RE.search(os.path.basename(path))
        if m:
            N = int(m.group(1))
            # Require all three splits before including this N.
            if all(
                os.path.isfile(os.path.join(data_dir, f"{s}_N{N}.pkl"))
                for s in ("train", "val", "test")
            ):
                found.append(N)
    return sorted(set(found))


def parse_ckpt_overrides(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(
                f"invalid --ckpt {item!r}; expected N=PATH, e.g. "
                "--ckpt 15=runs/safe_select_J01_N15/best.pt"
            )
        left, path = item.split("=", 1)
        try:
            N = int(left)
        except ValueError as exc:
            raise SystemExit(f"invalid N in --ckpt {item!r}") from exc
        if not path:
            raise SystemExit(f"empty path in --ckpt {item!r}")
        out[N] = path
    return out


def run(cmd, *, dry_run=False):
    print("\n>>>", shlex.join([str(x) for x in cmd]), flush=True)
    if not dry_run:
        subprocess.run([str(x) for x in cmd], check=True)


def maybe_run(cmd, output, *, force=False, dry_run=False):
    if output and os.path.exists(output) and not force:
        print(f"\n[skip] exists: {output}")
        return
    run(cmd, dry_run=dry_run)


def build_parser():
    ap = argparse.ArgumentParser(
        description="Run frozen NSSR-V2 train/validate/projection sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument(
        "--Ns", type=int, nargs="*", default=None,
        help="slice counts; omit to auto-discover train_N*.pkl",
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--runs", default="runs/final_sweep")
    ap.add_argument("--results", default="results/final_sweep")
    ap.add_argument("--seed", type=int, default=0)

    # Frozen geometry/training settings.
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--train_n_u", type=int, default=12)
    ap.add_argument("--eval_n_u", type=int, default=16)
    ap.add_argument("--surf_sub", type=int, default=2000)
    ap.add_argument("--gt_sub", type=int, default=2000)
    ap.add_argument("--val_subset", type=int, default=20)
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--lam_jacobian", type=float, default=0.01)
    ap.add_argument("--jacobian_margin", type=float, default=0.05)
    ap.add_argument("--jacobian_power", type=float, default=2.0)
    ap.add_argument("--lam_cap_fold", type=float, default=0.005)
    ap.add_argument("--cap_fold_margin", type=float, default=1e-3)
    ap.add_argument("--cap_fold_power", type=float, default=2.0)
    ap.add_argument("--geometry_topk", type=float, default=0.05)
    ap.add_argument("--max_cap_fold", type=float, default=1e-3)
    ap.add_argument("--c_bound", type=float, default=1.0)

    ap.add_argument(
        "--ckpt", action="append", default=[], metavar="N=PATH",
        help="reuse an existing best.pt for a particular N; repeatable",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="rerun stages even when their expected output already exists",
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument(
        "--no_collect", action="store_true",
        help="do not invoke collect_results.py at the end",
    )
    return ap


def main():
    a = build_parser().parse_args()

    if not (1 <= a.epochs <= MAX_EPOCHS):
        raise SystemExit(f"--epochs must be in [1, {MAX_EPOCHS}]")
    if a.train_n_u < 3 or a.eval_n_u < 3:
        raise SystemExit("--train_n_u and --eval_n_u must be >= 3")
    if a.m < 4:
        raise SystemExit("--m must be >= 4")

    discovered = discover_Ns(a.data)
    if a.Ns is None or len(a.Ns) == 0:
        Ns = discovered
    else:
        Ns = sorted(set(a.Ns))
        missing = [N for N in Ns if N not in discovered]
        if missing:
            raise SystemExit(
                "missing complete train/val/test datasets for N="
                + ",".join(map(str, missing))
            )

    if not Ns:
        raise SystemExit(
            f"no complete train_N*/val_N*/test_N* datasets found in {a.data}"
        )

    ckpt_overrides = parse_ckpt_overrides(a.ckpt)
    unknown = sorted(set(ckpt_overrides) - set(Ns))
    if unknown:
        raise SystemExit(
            "--ckpt supplied for N not in sweep: "
            + ",".join(map(str, unknown))
        )

    Path(a.runs).mkdir(parents=True, exist_ok=True)
    Path(a.results).mkdir(parents=True, exist_ok=True)

    print("NSSR-V2 final multi-N sweep")
    print("  Ns                 :", ", ".join(map(str, Ns)))
    print("  epochs             :", a.epochs)
    print("  train n_u          :", a.train_n_u)
    print("  evaluation n_u     :", a.eval_n_u)
    print("  lambda Jacobian    :", a.lam_jacobian)
    print("  lambda cap fold    :", a.lam_cap_fold)
    print("  cap threshold      :", a.max_cap_fold)
    print("  geometry top-k     :", a.geometry_topk)
    print("  c_bound            :", a.c_bound)
    print("  projection         : failure-aware shared")

    py = sys.executable

    for N in Ns:
        print("\n" + "=" * 72)
        print(f"N={N}")
        print("=" * 72)

        run_dir = os.path.join(a.runs, f"N{N}")
        default_ckpt = os.path.join(run_dir, "best.pt")
        ckpt = ckpt_overrides.get(N, default_ckpt)

        if N not in ckpt_overrides:
            train_cmd = [
                py, "scripts/train_model.py",
                "--data", a.data,
                "--N", str(N),
                "--epochs", str(a.epochs),
                "--seed", str(a.seed),
                "--m", str(a.m),
                "--n_u", str(a.train_n_u),
                "--eval_n_u", "0",  # preserve finalized N=15 training setup
                "--surf_sub", str(a.surf_sub),
                "--gt_sub", str(a.gt_sub),
                "--val_every", str(a.val_every),
                "--val_subset", str(a.val_subset),
                "--lam_jacobian", str(a.lam_jacobian),
                "--jacobian_margin", str(a.jacobian_margin),
                "--jacobian_power", str(a.jacobian_power),
                "--lam_cap_fold", str(a.lam_cap_fold),
                "--cap_fold_margin", str(a.cap_fold_margin),
                "--cap_fold_power", str(a.cap_fold_power),
                "--max_cap_fold", str(a.max_cap_fold),
                "--geometry_topk", str(a.geometry_topk),
                "--c_bound", str(a.c_bound),
                "--out", run_dir,
            ]
            maybe_run(
                train_cmd,
                default_ckpt,
                force=a.force,
                dry_run=a.dry_run,
            )
        else:
            print(f"[reuse] N={N} checkpoint: {ckpt}")

        if not a.dry_run and not os.path.isfile(ckpt):
            raise RuntimeError(f"checkpoint not found after training: {ckpt}")

        validate_out = os.path.join(a.results, f"N{N}_validate.csv")
        validate_cmd = [
            py, "scripts/validate.py",
            "--data", a.data,
            "--split", "test",
            "--N", str(N),
            "--ckpt", ckpt,
            "--m", str(a.m),
            "--n_u", str(a.eval_n_u),
            "--c_bound", str(a.c_bound),
            "--max_cap_fold", str(a.max_cap_fold),
            "--out", validate_out,
        ]
        maybe_run(
            validate_cmd,
            validate_out,
            force=a.force,
            dry_run=a.dry_run,
        )

        projection_out = os.path.join(a.results, f"N{N}_projection.csv")
        projection_cmd = [
            py, "scripts/evaluate_projection.py",
            "--data", a.data,
            "--split", "test",
            "--N", str(N),
            "--ckpt", ckpt,
            "--m", str(a.m),
            "--n_u", str(a.eval_n_u),
            "--c_bound", str(a.c_bound),
            "--max_cap_fold", str(a.max_cap_fold),
            "--projection_mode", "staged",
            "--out", projection_out,
        ]
        maybe_run(
            projection_cmd,
            projection_out,
            force=a.force,
            dry_run=a.dry_run,
        )

    if not a.no_collect:
        summary_out = os.path.join(a.results, "summary.csv")
        collect_cmd = [
            py, "scripts/collect_results.py",
            "--results", a.results,
            "--out", summary_out,
        ]
        run(collect_cmd, dry_run=a.dry_run)

    print("\nFINAL SWEEP COMPLETE")


if __name__ == "__main__":
    main()
