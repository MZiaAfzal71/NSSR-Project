"""Command-line entry point for NSSR-V2 training.

Active training objective
-------------------------
Accuracy:
  - Chamfer distance
  - optional normal loss
  - parameter magnitude regularization
  - circumferential parameter smoothness

Geometry safety:
  - top-k signed-Jacobian barrier
  - top-k cap turn-back barrier

Curvature is intentionally not part of the active training objective.

Example (N=15 safety smoke test):
    python scripts/train_model.py \
        --data data/synthetic \
        --N 15 \
        --epochs 5 \
        --m 128 \
        --n_u 12 \
        --surf_sub 2000 \
        --gt_sub 2000 \
        --val_every 1 \
        --val_subset 10 \
        --lam_jacobian 0.01 \
        --lam_cap_fold 0.1 \
        --cap_fold_margin 1e-3 \
        --geometry_topk 0.05 \
        --out runs/smoke_safe_topk_N15
"""

from __future__ import annotations

import argparse
import inspect
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nssr.train import train


MAX_EPOCHS = 100


def _bounded_fraction(value: str) -> float:
    x = float(value)
    if not (0.0 < x <= 1.0):
        raise argparse.ArgumentTypeError("must lie in (0, 1]")
    return x


def _nonnegative(value: str) -> float:
    x = float(value)
    if x < 0.0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return x


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train NSSR-V2 with Jacobian + cap-turnback safety.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset / experiment.
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--N", type=int, default=7)
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)

    # Optimizer / representation.
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n_u", type=int, default=24)
    ap.add_argument(
        "--eval_n_u",
        type=int,
        default=0,
        help="validation n_u; 0 uses training n_u",
    )
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--fp64", action="store_true")

    # Accuracy / classical NSSR regularizers.
    ap.add_argument("--lam_n", type=float, default=0.1)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--lam_s", type=float, default=1e-3)
    ap.add_argument(
        "--cap_weight",
        type=float,
        default=1.0,
        help="Chamfer weight for actual cap-patch points",
    )

    # Geometry-safety objective.
    ap.add_argument(
        "--lam_jacobian",
        type=_nonnegative,
        default=0.0,
        help="weight of the hard-sample signed-Jacobian barrier",
    )
    ap.add_argument(
        "--jacobian_margin",
        type=_nonnegative,
        default=1e-4,
        help="desired positive signed-Jacobian margin",
    )
    ap.add_argument(
        "--jacobian_power",
        type=float,
        default=2.0,
        help="power applied to Jacobian margin violations",
    )
    ap.add_argument(
        "--lam_cap_fold",
        type=_nonnegative,
        default=0.0,
        help="weight of cap radial/meridional turn-back barrier",
    )
    ap.add_argument(
        "--cap_fold_margin",
        type=_nonnegative,
        default=1e-3,
        help="allowed normalized cap turn-back; matches validation threshold",
    )
    ap.add_argument(
        "--cap_fold_power",
        type=float,
        default=2.0,
        help="power applied to cap turn-back excess",
    )
    ap.add_argument(
        "--geometry_topk",
        type=_bounded_fraction,
        default=0.05,
        help="fraction of worst geometry samples used by both safety barriers",
    )

    # Model options.
    ap.add_argument(
        "--c_bound",
        type=float,
        default=1.0,
        help="ParamNet tanh output bound",
    )
    ap.add_argument(
        "--no_learn_heights",
        action="store_true",
        help="disable learned base/crown height corrections",
    )
    ap.add_argument(
        "--init_ckpt",
        default="",
        help="initialize/fine-tune from an existing checkpoint",
    )

    # Training memory / validation.
    ap.add_argument("--surf_sub", type=int, default=8000)
    ap.add_argument("--gt_sub", type=int, default=8000)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument(
        "--val_subset",
        type=int,
        default=0,
        help="validate on first K objects; 0 means all",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=0,
        help="early-stop patience in epochs; 0 disables",
    )

    return ap


def _check_args(a: argparse.Namespace) -> None:
    if a.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if a.epochs > MAX_EPOCHS:
        raise SystemExit(
            f"--epochs={a.epochs} exceeds the project maximum of "
            f"{MAX_EPOCHS}. Use <= {MAX_EPOCHS}."
        )
    if a.N < 2:
        raise SystemExit("--N must be >= 2")
    if a.m < 4:
        raise SystemExit("--m must be >= 4")
    if a.n_u < 3:
        raise SystemExit("--n_u must be >= 3")
    if a.eval_n_u < 0:
        raise SystemExit("--eval_n_u must be >= 0")
    if a.accum < 1:
        raise SystemExit("--accum must be >= 1")
    if a.val_every < 1:
        raise SystemExit("--val_every must be >= 1")
    if a.surf_sub < 1 or a.gt_sub < 1:
        raise SystemExit("--surf_sub and --gt_sub must be >= 1")
    if a.jacobian_power <= 0 or a.cap_fold_power <= 0:
        raise SystemExit("--jacobian_power and --cap_fold_power must be > 0")
    if not (0.0 < a.cap_weight):
        raise SystemExit("--cap_weight must be > 0")


def _load_split(data_dir: str, split: str, N: int):
    path = os.path.join(data_dir, f"{split}_N{N}.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"missing {split} split for N={N}: {path}"
        )
    with open(path, "rb") as f:
        samples = pickle.load(f)
    if not samples:
        raise RuntimeError(f"empty dataset split: {path}")
    return samples, path


def _verify_train_api() -> None:
    """Fail early with a precise message if nssr/train.py is stale."""
    required = {
        "lam_jacobian",
        "jacobian_margin",
        "jacobian_power",
        "lam_cap_fold",
        "cap_fold_margin",
        "cap_fold_power",
        "geometry_topk_fraction",
    }
    available = set(inspect.signature(train).parameters)
    missing = sorted(required - available)

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "scripts/train_model.py is newer than nssr/train.py.\n"
            f"Your train() function is missing: {joined}\n\n"
            "Update nssr/train.py so train(...) accepts these arguments and "
            "forwards them to total_loss(...)."
        )


def main() -> None:
    ap = build_parser()
    a = ap.parse_args()
    _check_args(a)
    _verify_train_api()

    train_samples, train_path = _load_split(a.data, "train", a.N)
    val_samples, val_path = _load_split(a.data, "val", a.N)

    dtype = torch.float64 if a.fp64 else torch.float32
    geometry_on = (a.lam_jacobian > 0.0 or a.lam_cap_fold > 0.0)

    print("NSSR-V2 training configuration")
    print(f"  train data          : {train_path}")
    print(f"  validation data     : {val_path}")
    print(f"  train objects       : {len(train_samples)}")
    print(f"  validation objects  : {len(val_samples)}")
    print(f"  slices N            : {a.N}")
    print(f"  contour samples m   : {a.m}")
    print(f"  patch samples n_u   : {a.n_u}")
    print(f"  epochs              : {a.epochs}")
    print(f"  dtype               : {dtype}")
    print(f"  geometry losses     : {'ON' if geometry_on else 'OFF'}")
    print(f"  lambda Jacobian     : {a.lam_jacobian:g}")
    print(f"  Jacobian margin     : {a.jacobian_margin:g}")
    print(f"  Jacobian power      : {a.jacobian_power:g}")
    print(f"  lambda cap fold     : {a.lam_cap_fold:g}")
    print(f"  cap fold margin     : {a.cap_fold_margin:g}")
    print(f"  cap fold power      : {a.cap_fold_power:g}")
    print(f"  geometry top-k      : {100.0 * a.geometry_topk:.1f}%")
    print(f"  output              : {a.out}")

    os.makedirs(a.out, exist_ok=True)

    train(
        train_samples,
        val_samples,
        out_dir=a.out,
        epochs=a.epochs,
        lr=a.lr,
        m=a.m,
        n_u=a.n_u,
        dtype=dtype,
        lam_n=a.lam_n,
        lam_r=a.reg,
        lam_s=a.lam_s,
        lam_jacobian=a.lam_jacobian,
        jacobian_margin=a.jacobian_margin,
        jacobian_power=a.jacobian_power,
        lam_cap_fold=a.lam_cap_fold,
        cap_fold_margin=a.cap_fold_margin,
        cap_fold_power=a.cap_fold_power,
        geometry_topk_fraction=a.geometry_topk,
        accum=a.accum,
        seed=a.seed,
        surf_sub=a.surf_sub,
        gt_sub=a.gt_sub,
        val_every=a.val_every,
        patience=a.patience,
        val_subset=a.val_subset,
        eval_n_u=(a.eval_n_u or None),
        init_ckpt=(a.init_ckpt or None),
        cap_weight=a.cap_weight,
        learn_heights=not a.no_learn_heights,
        c_bound=a.c_bound,
    )


if __name__ == "__main__":
    main()
