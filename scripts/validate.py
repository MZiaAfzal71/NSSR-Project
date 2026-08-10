"""Validate NSSR-V2 reconstruction safety and accuracy.

Final safety definition
-----------------------
A reconstruction is considered SAFE when:

1. it has no negative signed-Jacobian samples relative to the fixed
   classical pointwise orientation field;
2. it has no accidental Jacobian degeneracy away from intentional cap poles;
3. its maximum normalized cap radial turn-back is <= ``--max_cap_fold``.

Curvature is deliberately excluded from validity.  ``nssr.core.curvature``
may still be used separately for offline analysis.

Usage:
    python scripts/validate.py \
        --data data/synthetic \
        --split test \
        --N 7 \
        --ckpt runs/smoke_safe/best.pt \
        --m 128 \
        --n_u 16 \
        --limit 20 \
        --max_cap_fold 1e-3 \
        --out results/smoke_safe_validate.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nssr.geometry import evaluate_geometry, surface_points, zero_params
from nssr.losses import (
    cap_radial_fold_max,
    cap_radial_fold_measure,
    intentional_pole_mask,
)
from nssr.metrics import evaluate_surface
from nssr.networks import ParamNet, contour_features
from nssr.train import to_torch


def _load_state_dict(path, device, dtype):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {k: v.to(dtype=dtype) for k, v in state.items()}


def _classical_geometry(obj, n_u):
    """Build classical surface twice so both variants use the same reference."""
    N, m = obj["R"].shape[:2]
    p0 = zero_params(
        N,
        m,
        device=obj["R"].device,
        dtype=obj["R"].dtype,
    )

    # Pass 1: fixed pointwise orientation reference.
    ref = evaluate_geometry(
        obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"],
        p0,
        n_u=n_u,
        closed_top=obj.get("closed_top", True),
        base_circular=obj.get("base_circular", True),
        crown_circular=obj.get("crown_circular", True),
        compute_jacobian=False,
        compute_curvature=False,
        run_validation=False,
    )
    reference_normal = ref.surface.normals.detach()

    # Pass 2: score classical against exactly that field.
    classical = evaluate_geometry(
        obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"],
        p0,
        n_u=n_u,
        closed_top=obj.get("closed_top", True),
        base_circular=obj.get("base_circular", True),
        crown_circular=obj.get("crown_circular", True),
        compute_jacobian=True,
        compute_curvature=False,
        run_validation=False,
        reference_normal=reference_normal,
    )
    return classical, reference_normal


def _learned_geometry(net, obj, n_u, reference_normal):
    feats = contour_features(
        obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"]
    )
    params = net(feats)

    geom = evaluate_geometry(
        obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"],
        params,
        n_u=n_u,
        closed_top=obj.get("closed_top", True),
        base_circular=obj.get("base_circular", True),
        crown_circular=obj.get("crown_circular", True),
        compute_jacobian=True,
        compute_curvature=False,
        run_validation=False,
        reference_normal=reference_normal,
    )
    return geom


def _geometry_summary(geom, obj, max_cap_fold):
    jac = geom.jacobian
    if jac is None:
        raise RuntimeError("Jacobian diagnostics were not computed")

    eval_mask = ~intentional_pole_mask(
        geom.surface.xyz,
        closed_top=obj.get("closed_top", True),
    )

    # Count finite accidental failures explicitly.  Degenerate mask already
    # includes non-finite derivative/area samples.
    neg_mask = jac.flipped_mask & eval_mask
    deg_mask = jac.degenerate_mask & eval_mask

    denom = max(int(eval_mask.sum().item()), 1)
    neg_count = int(neg_mask.sum().item())
    deg_count = int(deg_mask.sum().item())

    j_valid = (neg_count == 0 and deg_count == 0)

    base_v, crown_v = cap_radial_fold_measure(
        geom.surface.xyz,
        obj["RB"],
        obj["RC"],
        closed_top=obj.get("closed_top", True),
    )

    base_max = float(base_v.max().item()) if base_v.numel() else 0.0
    crown_max = float(crown_v.max().item()) if crown_v.numel() else 0.0
    cap_max = float(
        cap_radial_fold_max(
            geom.surface.xyz,
            obj["RB"],
            obj["RC"],
            closed_top=obj.get("closed_top", True),
        ).item()
    )
    cap_safe = cap_max <= max_cap_fold
    safe = j_valid and cap_safe

    signed_eval = jac.signed[eval_mask]
    finite_signed = signed_eval[torch.isfinite(signed_eval)]
    minimum_signed = (
        float(finite_signed.min().item())
        if finite_signed.numel()
        else float("nan")
    )

    area_eval = jac.area_scale[eval_mask]
    finite_area = area_eval[torch.isfinite(area_eval)]
    minimum_area = (
        float(finite_area.min().item())
        if finite_area.numel()
        else float("nan")
    )

    return {
        "jacobian_valid": bool(j_valid),
        "negative_jacobian": neg_count,
        "negative_fraction": float(neg_count / denom),
        "degenerate": deg_count,
        "degenerate_fraction": float(deg_count / denom),
        "minimum_signed_jacobian": minimum_signed,
        "minimum_area_scale": minimum_area,
        "base_cap_fold_max": base_max,
        "crown_cap_fold_max": crown_max,
        "cap_fold_max": cap_max,
        "cap_safe": bool(cap_safe),
        "safe": bool(safe),
    }


def _gt_metrics(obj, geom):
    if obj.get("gt_pts") is None:
        return {}

    pts = surface_points(geom.surface.xyz)
    nrms = geom.surface.normals.reshape(-1, 3)

    result = evaluate_surface(
        pts,
        obj["gt_pts"],
        nrms,
        obj.get("gt_normals"),
    )
    return {k: float(v) for k, v in result.items()}


def _mean(rows, key):
    vals = [float(r[key]) for r in rows if key in r]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--split", default="test")
    ap.add_argument("--N", type=int, default=7)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n_u", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/validate.csv")
    ap.add_argument("--c_bound", type=float, default=1.0)
    ap.add_argument("--no_learn_heights", action="store_true")
    ap.add_argument(
        "--max_cap_fold",
        type=float,
        default=1e-3,
        help="maximum normalized radial cap turn-back considered safe",
    )
    ap.add_argument("--fp64", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64 if a.fp64 else torch.float32

    path = os.path.join(a.data, f"{a.split}_N{a.N}.pkl")
    with open(path, "rb") as f:
        samples = pickle.load(f)

    if a.limit > 0:
        samples = samples[:a.limit]

    if not samples:
        raise RuntimeError(f"empty split: {path}")

    net = None
    if a.ckpt:
        net = ParamNet(
            learn_heights=not a.no_learn_heights,
            c_bound=a.c_bound,
        ).to(device=device, dtype=dtype)
        net.load_state_dict(_load_state_dict(a.ckpt, device, dtype))
        net.eval()

    rows = []

    with torch.no_grad():
        for i, sample in enumerate(samples):
            obj = to_torch(
                sample,
                a.m,
                device,
                dtype,
                seed=40_000 + i,
            )

            classical, reference_normal = _classical_geometry(obj, a.n_u)
            csum = _geometry_summary(classical, obj, a.max_cap_fold)
            cmetrics = _gt_metrics(obj, classical)

            row = {
                "idx": i,
                **{f"classical_{k}": v for k, v in cmetrics.items()},
                **{f"classical_geom_{k}": v for k, v in csum.items()},
            }

            if net is not None:
                learned = _learned_geometry(
                    net,
                    obj,
                    a.n_u,
                    reference_normal,
                )
                lsum = _geometry_summary(learned, obj, a.max_cap_fold)
                lmetrics = _gt_metrics(obj, learned)

                row.update(
                    {f"learned_{k}": v for k, v in lmetrics.items()}
                )
                row.update(
                    {f"learned_geom_{k}": v for k, v in lsum.items()}
                )

                if not lsum["jacobian_valid"]:
                    status = "J-FAIL"
                elif not lsum["cap_safe"]:
                    status = "CAP-FAIL"
                else:
                    status = "OK"

                print(
                    f"[{status:8s}] {i:4d} "
                    f"classical Jneg={100*csum['negative_fraction']:.3f}% "
                    f"cap={csum['cap_fold_max']:.5f} | "
                    f"learned Jneg={100*lsum['negative_fraction']:.3f}% "
                    f"deg={100*lsum['degenerate_fraction']:.3f}% "
                    f"cap={lsum['cap_fold_max']:.5f}"
                )
            else:
                print(
                    f"[CLASS   ] {i:4d} "
                    f"Jneg={100*csum['negative_fraction']:.3f}% "
                    f"deg={100*csum['degenerate_fraction']:.3f}% "
                    f"cap={csum['cap_fold_max']:.5f}"
                )

            rows.append(row)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(a.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nsummary")
    for prefix in ("classical", "learned"):
        jkey = f"{prefix}_geom_jacobian_valid"
        if jkey not in rows[0]:
            continue

        j_valid = _mean(rows, jkey)
        cap_safe = _mean(rows, f"{prefix}_geom_cap_safe")
        safe = _mean(rows, f"{prefix}_geom_safe")
        neg = _mean(rows, f"{prefix}_geom_negative_fraction")
        deg = _mean(rows, f"{prefix}_geom_degenerate_fraction")
        cap = _mean(rows, f"{prefix}_geom_cap_fold_max")

        worst_cap = max(
            float(r[f"{prefix}_geom_cap_fold_max"])
            for r in rows
        )

        print(
            f"{prefix:10s} "
            f"J-valid={100*j_valid:.1f}% "
            f"cap-safe={100*cap_safe:.1f}% "
            f"SAFE={100*safe:.1f}% "
            f"mean negative-J={100*neg:.4f}% "
            f"mean degenerate={100*deg:.4f}% "
            f"mean cap-fold={cap:.6f} "
            f"worst cap-fold={worst_cap:.6f}"
        )

    print(f"threshold cap-fold <= {a.max_cap_fold:g}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
