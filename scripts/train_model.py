"""Train NSSR-V2.

This CLI preserves the existing NSSR training options and adds the V2
geometry-aware regularization controls introduced in ``nssr.train``.

Examples
--------
Legacy-compatible training:

    python scripts/train_model.py \
        --data data/synthetic \
        --N 7 \
        --epochs 200 \
        --out runs/baseline

V2 geometry-aware training:

    python scripts/train_model.py \
        --data data/synthetic \
        --N 7 \
        --epochs 200 \
        --out runs/v2_geom \
        --lam_jacobian 1e-3 \
        --lam_curvature 1e-5 \
        --jacobian_margin 1e-4 \
        --max_abs_curvature 100
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import torch


# Allow execution directly from the repository root without package install.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
    ),
)

from nssr.train import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train NSSR-V2 surface reconstruction model."
    )

    # ------------------------------------------------------------------
    # Dataset / output
    # ------------------------------------------------------------------
    parser.add_argument(
        "--data",
        default="data/synthetic",
        help="Directory containing train_N*.pkl and val_N*.pkl.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=7,
        help="Number of input contour slices used by the dataset split.",
    )
    parser.add_argument(
        "--out",
        default="runs/exp1",
        help="Output directory for logs and checkpoints.",
    )

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--accum",
        type=int,
        default=8,
        help="Gradient accumulation object count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    # ------------------------------------------------------------------
    # Surface sampling / preprocessing
    # ------------------------------------------------------------------
    parser.add_argument(
        "--m",
        type=int,
        default=256,
        help="Circumferential contour resampling resolution.",
    )
    parser.add_argument(
        "--n_u",
        type=int,
        default=24,
        help="Hermite samples per patch during training.",
    )
    parser.add_argument(
        "--eval_n_u",
        type=int,
        default=0,
        help="Validation patch resolution; 0 uses --n_u.",
    )
    parser.add_argument(
        "--surf_sub",
        type=int,
        default=8000,
        help="Maximum predicted points used in Chamfer training loss.",
    )
    parser.add_argument(
        "--gt_sub",
        type=int,
        default=8000,
        help="Maximum ground-truth points used in training.",
    )

    # ------------------------------------------------------------------
    # Legacy NSSR losses
    # ------------------------------------------------------------------
    parser.add_argument(
        "--lam_n",
        type=float,
        default=0.1,
        help="Normal-loss weight.",
    )
    parser.add_argument(
        "--reg",
        "--lam_r",
        dest="lam_r",
        type=float,
        default=1e-3,
        help="Parameter proximity/L2 weight.",
    )
    parser.add_argument(
        "--lam_s",
        type=float,
        default=1e-3,
        help="Circumferential parameter-smoothness weight.",
    )

    # ------------------------------------------------------------------
    # NSSR-V2 geometry-aware losses
    # ------------------------------------------------------------------
    parser.add_argument(
        "--lam_jacobian",
        type=float,
        default=0.0,
        help=(
            "Signed-Jacobian barrier weight. Default 0 preserves the "
            "legacy training path."
        ),
    )
    parser.add_argument(
        "--jacobian_margin",
        type=float,
        default=1e-4,
        help="Desired positive signed-Jacobian margin.",
    )
    parser.add_argument(
        "--lam_cap_fold",
        type=float,
        default=0.0,
        help="Weight for cap radial turn-back / loop prevention.",
    )
    parser.add_argument(
        "--cap_fold_power",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--cap_fold_margin",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--geometry_topk",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--jacobian_power",
        type=float,
        default=2.0,
    )

    # ------------------------------------------------------------------
    # Model constraints / cap handling
    # ------------------------------------------------------------------
    parser.add_argument(
        "--c_bound",
        type=float,
        default=1.0,
        help=(
            "Bound on learned log-multipliers. Must match checkpoint "
            "configuration when loading an existing model."
        ),
    )
    parser.add_argument(
        "--no_learn_heights",
        action="store_true",
        help=(
            "Disable learned base/crown height multipliers s_bh/s_th."
        ),
    )
    parser.add_argument(
        "--cap_weight",
        type=float,
        default=1.0,
        help=(
            "Relative Chamfer weight assigned to cap-patch points. "
            "1.0 disables cap reweighting."
        ),
    )

    # ------------------------------------------------------------------
    # Validation / checkpointing
    # ------------------------------------------------------------------
    parser.add_argument(
        "--val_every",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early-stop patience in epochs; 0 disables early stopping.",
    )
    parser.add_argument(
        "--val_subset",
        type=int,
        default=0,
        help="Validate on first K objects only; 0 uses all.",
    )
    parser.add_argument(
        "--init_ckpt",
        default="",
        help="Optional checkpoint used to initialize/fine-tune the model.",
    )

    # ------------------------------------------------------------------
    # Numeric precision
    # ------------------------------------------------------------------
    parser.add_argument(
        "--fp64",
        action="store_true",
        help="Train using torch.float64 instead of torch.float32.",
    )

    return parser


def load_split(data_dir: str, split: str, n_slices: int):
    path = os.path.join(
        data_dir,
        f"{split}_N{n_slices}.pkl",
    )

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset split not found: {path}"
        )

    with open(path, "rb") as handle:
        return pickle.load(handle)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.N < 2:
        parser.error("--N must be >= 2")
    if args.epochs < 0:
        parser.error("--epochs must be >= 0")
    if args.m < 3:
        parser.error("--m must be >= 3")
    if args.n_u < 2:
        parser.error("--n_u must be >= 2")
    if args.accum < 1:
        parser.error("--accum must be >= 1")
    if args.lam_jacobian < 0:
        parser.error("--lam_jacobian must be >= 0")
    if args.lam_cap_fold < 0:
        parser.error("--lam_cap_fold must be >= 0")
    if args.cap_fold_power <= 0:
        parser.error("--cap_fold_power must be > 0")
    if args.cap_weight <= 0:
        parser.error("--cap_weight must be > 0")

    train_samples = load_split(
        args.data,
        "train",
        args.N,
    )
    val_samples = load_split(
        args.data,
        "val",
        args.N,
    )

    dtype = (
        torch.float64
        if args.fp64
        else torch.float32
    )

    geometry_enabled = (
        args.lam_jacobian != 0.0
        or args.lam_cap_fold != 0.0
    )

    print("NSSR-V2 training configuration")
    print(f"  train objects       : {len(train_samples)}")
    print(f"  validation objects  : {len(val_samples)}")
    print(f"  contour samples m   : {args.m}")
    print(f"  patch samples n_u   : {args.n_u}")
    print(f"  dtype               : {dtype}")
    print(f"  geometry losses     : {'ON' if geometry_enabled else 'OFF'}")
    print(f"  cap fold margin     : {args.cap_fold_margin:g}")
    print(f"  geometry top-k      : {100 * args.geometry_topk:.1f}%")

    if geometry_enabled:
        print(f"  lambda Jacobian     : {args.lam_jacobian:g}")
        print(f"  lambda cap fold    : {args.lam_cap_fold:g}")
        print(f"  Jacobian margin     : {args.jacobian_margin:g}")

    train(
        train_samples,
        val_samples,
        out_dir=args.out,
        epochs=args.epochs,
        lr=args.lr,
        m=args.m,
        n_u=args.n_u,
        dtype=dtype,
        lam_n=args.lam_n,
        lam_r=args.lam_r,
        lam_s=args.lam_s,
        cap_fold_margin=args.cap_fold_margin,
        geometry_topk_fraction=args.geometry_topk,
        jacobian_power=args.jacobian_power,
        lam_jacobian=args.lam_jacobian,
        jacobian_margin=args.jacobian_margin,
        lam_cap_fold=args.lam_cap_fold,
        cap_fold_power=args.cap_fold_power,
        accum=args.accum,
        seed=args.seed,
        surf_sub=args.surf_sub,
        gt_sub=args.gt_sub,
        val_every=args.val_every,
        patience=args.patience,
        val_subset=args.val_subset,
        eval_n_u=(args.eval_n_u or None),
        init_ckpt=(args.init_ckpt or None),
        cap_weight=args.cap_weight,
        learn_heights=not args.no_learn_heights,
        c_bound=args.c_bound,
    )


if __name__ == "__main__":
    main()
