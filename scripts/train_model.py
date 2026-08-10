"""CLI for NSSR-V2 safety-aware training."""

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


def _positive(value: str) -> float:
    x = float(value)
    if x <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return x


def _nonnegative(value: str) -> float:
    x = float(value)
    if x < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return x


def build_parser():
    ap = argparse.ArgumentParser(
        description=(
            "Train NSSR-V2 with normalized orientation/cap safety and "
            "safety-aware checkpoint selection."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--N", type=int, default=7)
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n_u", type=int, default=24)
    ap.add_argument(
        "--eval_n_u", type=int, default=0,
        help="validation n_u; 0 uses training n_u"
    )
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--fp64", action="store_true")

    ap.add_argument("--lam_n", type=float, default=0.1)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--lam_s", type=float, default=1e-3)
    ap.add_argument("--cap_weight", type=_positive, default=1.0)

    ap.add_argument("--lam_jacobian", type=_nonnegative, default=0.0)
    ap.add_argument(
        "--jacobian_margin",
        type=_nonnegative,
        default=0.05,
        help="minimum normalized orientation signed_J / area_scale",
    )
    ap.add_argument("--jacobian_power", type=_positive, default=2.0)

    ap.add_argument("--lam_cap_fold", type=_nonnegative, default=0.0)
    ap.add_argument(
        "--cap_fold_margin",
        type=_positive,
        default=1e-3,
        help="training cap turn-back threshold",
    )
    ap.add_argument("--cap_fold_power", type=_positive, default=2.0)
    ap.add_argument(
        "--geometry_topk",
        type=_bounded_fraction,
        default=0.05,
        help="worst geometry fraction used by training barriers",
    )

    ap.add_argument(
        "--max_cap_fold",
        type=_positive,
        default=1e-3,
        help=(
            "validation cap safety threshold; should normally equal "
            "--cap_fold_margin"
        ),
    )

    ap.add_argument("--c_bound", type=float, default=1.0)
    ap.add_argument("--no_learn_heights", action="store_true")
    ap.add_argument("--init_ckpt", default="")

    ap.add_argument("--surf_sub", type=int, default=8000)
    ap.add_argument("--gt_sub", type=int, default=8000)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--val_subset", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0)

    return ap


def _check_args(a):
    if not (1 <= a.epochs <= MAX_EPOCHS):
        raise SystemExit(
            f"--epochs must be in [1, {MAX_EPOCHS}]"
        )
    if a.N < 2:
        raise SystemExit("--N must be >= 2")
    if a.m < 4:
        raise SystemExit("--m must be >= 4")
    if a.n_u < 3:
        raise SystemExit("--n_u must be >= 3")
    if a.accum < 1:
        raise SystemExit("--accum must be >= 1")
    if a.val_every < 1:
        raise SystemExit("--val_every must be >= 1")
    if a.eval_n_u < 0:
        raise SystemExit("--eval_n_u must be >= 0")
    if a.surf_sub < 1 or a.gt_sub < 1:
        raise SystemExit("--surf_sub and --gt_sub must be >= 1")


def _load_split(data_dir, split, N):
    path = os.path.join(data_dir, f"{split}_N{N}.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        samples = pickle.load(f)
    if not samples:
        raise RuntimeError(f"empty split: {path}")
    return samples, path


def _verify_train_api():
    required = {
        "lam_jacobian",
        "jacobian_margin",
        "jacobian_power",
        "lam_cap_fold",
        "cap_fold_margin",
        "cap_fold_power",
        "geometry_topk_fraction",
        "max_cap_fold",
    }
    available = set(inspect.signature(train).parameters)
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(
            "scripts/train_model.py and nssr/train.py are out of sync. "
            "train() is missing: " + ", ".join(missing)
        )


def main():
    a = build_parser().parse_args()
    _check_args(a)
    _verify_train_api()

    tr, tr_path = _load_split(a.data, "train", a.N)
    va, va_path = _load_split(a.data, "val", a.N)

    dtype = torch.float64 if a.fp64 else torch.float32
    geometry_on = (a.lam_jacobian > 0 or a.lam_cap_fold > 0)

    print("NSSR-V2 training configuration")
    print(f"  train data          : {tr_path}")
    print(f"  validation data     : {va_path}")
    print(f"  train objects       : {len(tr)}")
    print(f"  validation objects  : {len(va)}")
    print(f"  slices N            : {a.N}")
    print(f"  contour samples m   : {a.m}")
    print(f"  patch samples n_u   : {a.n_u}")
    print(f"  epochs              : {a.epochs}")
    print(f"  dtype               : {dtype}")
    print(f"  geometry losses     : {'ON' if geometry_on else 'OFF'}")
    print(f"  lambda Jacobian     : {a.lam_jacobian:g}")
    print(f"  orientation margin  : {a.jacobian_margin:g}")
    print(f"  Jacobian power      : {a.jacobian_power:g}")
    print(f"  lambda cap fold     : {a.lam_cap_fold:g}")
    print(f"  cap fold margin     : {a.cap_fold_margin:g}")
    print(f"  cap fold power      : {a.cap_fold_power:g}")
    print(f"  geometry top-k      : {100*a.geometry_topk:.1f}%")
    print(f"  validation cap max  : {a.max_cap_fold:g}")
    print("  checkpoint policy   : safety-first (best.pt)")
    print(f"  output              : {a.out}")

    if abs(a.max_cap_fold - a.cap_fold_margin) > 1e-12:
        print(
            "WARNING: training cap margin and validation cap threshold differ: "
            f"{a.cap_fold_margin:g} vs {a.max_cap_fold:g}"
        )

    os.makedirs(a.out, exist_ok=True)

    train(
        tr,
        va,
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
        max_cap_fold=a.max_cap_fold,
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
