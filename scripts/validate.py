"""Validate NSSR-V2 reconstructions on a dataset split.

Corrected version:
- builds a classical geometry first to obtain a fixed pointwise reference normal;
- recomputes BOTH classical and learned Jacobians against the same reference;
- excludes intentional cap-pole singularities;
- reports orientation/degeneracy validity separately from curvature diagnostics.

Usage:
    python scripts/validate.py \
        --data data/synthetic \
        --split test \
        --N 7 \
        --ckpt runs/smoke_jac/best.pt \
        --out results/smoke_jac_validate.csv
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

from nssr.geometry import evaluate_geometry, zero_params, surface_points
from nssr.metrics import evaluate_surface
from nssr.networks import ParamNet, contour_features
from nssr.train import to_torch


def intentional_pole_mask(surface_shape, closed_top: bool, device):
    P, n_u, m, _ = surface_shape
    mask = torch.zeros(P, n_u, m, dtype=torch.bool, device=device)
    mask[0, 0, :] = True
    if closed_top:
        mask[-1, -1, :] = True
    return mask


def build_classical(obj, n_u):
    N, m = obj["R"].shape[:2]
    p0 = zero_params(N, m, device=obj["R"].device, dtype=obj["R"].dtype)

    # Pass 1: obtain the classical pointwise normal field.
    with torch.no_grad():
        base = evaluate_geometry(
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

        # Pass 2: score classical geometry against that SAME fixed reference
        # used later for the learned reconstruction.
        geom = evaluate_geometry(
            obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"],
            p0,
            n_u=n_u,
            closed_top=obj.get("closed_top", True),
            base_circular=obj.get("base_circular", True),
            crown_circular=obj.get("crown_circular", True),
            compute_jacobian=True,
            compute_curvature=True,
            run_validation=False,
            reference_normal=base.surface.normals,
        )

    return geom, base.surface.normals


def learned_params(net, obj):
    feats = contour_features(
        obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"]
    )
    return net(feats)


def summarize_geometry(geom, pole_mask, max_abs_curvature):
    jac = geom.jacobian
    curv = geom.curvature
    eval_mask = ~pole_mask

    neg = jac.flipped_mask & eval_mask
    deg = jac.degenerate_mask & eval_mask

    k1 = curv.principal_1
    k2 = curv.principal_2
    kmag = torch.maximum(torch.abs(k1), torch.abs(k2))
    curv_bad = (kmag > max_abs_curvature) & eval_mask

    denom = max(int(eval_mask.sum().item()), 1)

    # Keep topology/orientation validity separate from curvature diagnostics.
    jacobian_valid = (
        int(neg.sum().item()) == 0
        and int(deg.sum().item()) == 0
    )

    curvature_valid = int(curv_bad.sum().item()) == 0

    return {
        "negative_jacobian": int(neg.sum().item()),
        "negative_fraction": float(neg.sum().item() / denom),
        "degenerate": int(deg.sum().item()),
        "degenerate_fraction": float(deg.sum().item() / denom),
        "jacobian_valid": bool(jacobian_valid),
        "curvature_violations": int(curv_bad.sum().item()),
        "curvature_valid": bool(curvature_valid),
        "max_abs_curvature": float(
            kmag[eval_mask].amax().item() if bool(eval_mask.any()) else float("nan")
        ),
        # Overall validity retained for compatibility, but do not use this as
        # the sole sweep gate until curvature sampling is further hardened.
        "valid": bool(jacobian_valid and curvature_valid),
    }


def gt_metrics(obj, geom):
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
    ap.add_argument("--max_abs_curvature", type=float, default=100.0)
    ap.add_argument("--fp64", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float64 if a.fp64 else torch.float32

    path = os.path.join(a.data, f"{a.split}_N{a.N}.pkl")
    with open(path, "rb") as f:
        samples = pickle.load(f)
    if a.limit:
        samples = samples[:a.limit]

    net = None
    if a.ckpt:
        net = ParamNet(
            learn_heights=not a.no_learn_heights,
            c_bound=a.c_bound,
        ).to(device=dev, dtype=dt)

        state = torch.load(a.ckpt, map_location=dev)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state)
        net.eval()

    rows = []

    with torch.no_grad():
        for i, sample in enumerate(samples):
            obj = to_torch(sample, a.m, dev, dt, seed=40000 + i)

            classical, reference_normal = build_classical(obj, a.n_u)

            pole_mask = intentional_pole_mask(
                classical.surface.xyz.shape,
                obj.get("closed_top", True),
                classical.surface.xyz.device,
            )

            csum = summarize_geometry(
                classical,
                pole_mask,
                a.max_abs_curvature,
            )
            cm = gt_metrics(obj, classical)

            row = {
                "idx": i,
                **{f"classical_{k}": v for k, v in cm.items()},
                **{f"classical_geom_{k}": v for k, v in csum.items()},
            }

            if net is not None:
                params = learned_params(net, obj)

                learned = evaluate_geometry(
                    obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"],
                    params,
                    n_u=a.n_u,
                    closed_top=obj.get("closed_top", True),
                    base_circular=obj.get("base_circular", True),
                    crown_circular=obj.get("crown_circular", True),
                    compute_jacobian=True,
                    compute_curvature=True,
                    run_validation=False,
                    reference_normal=reference_normal,
                )

                lsum = summarize_geometry(
                    learned,
                    pole_mask,
                    a.max_abs_curvature,
                )
                lm = gt_metrics(obj, learned)

                row.update({f"learned_{k}": v for k, v in lm.items()})
                row.update({f"learned_geom_{k}": v for k, v in lsum.items()})

            rows.append(row)

            if net is not None:
                status = "OK" if row["learned_geom_jacobian_valid"] else "J-FAIL"
                print(
                    f"[{status:6s}] {i:4d} "
                    f"classical neg={100*csum['negative_fraction']:.3f}% "
                    f"deg={100*csum['degenerate_fraction']:.3f}% | "
                    f"learned neg={100*row['learned_geom_negative_fraction']:.3f}% "
                    f"deg={100*row['learned_geom_degenerate_fraction']:.3f}% "
                    f"curv_bad={row['learned_geom_curvature_violations']}"
                )
            else:
                print(
                    f"[CLASS ] {i:4d} "
                    f"neg={100*csum['negative_fraction']:.3f}% "
                    f"deg={100*csum['degenerate_fraction']:.3f}% "
                    f"curv_bad={csum['curvature_violations']}"
                )

    if not rows:
        raise RuntimeError("no samples to validate")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\nsummary")
    for prefix in ("classical", "learned"):
        key = f"{prefix}_geom_jacobian_valid"
        if key not in rows[0]:
            continue

        jac_valid = np.mean([float(r[key]) for r in rows])
        full_valid = np.mean([float(r[f"{prefix}_geom_valid"]) for r in rows])
        neg = np.mean([r[f"{prefix}_geom_negative_fraction"] for r in rows])
        deg = np.mean([r[f"{prefix}_geom_degenerate_fraction"] for r in rows])
        curv = np.mean([r[f"{prefix}_geom_curvature_violations"] for r in rows])

        print(
            f"{prefix:10s} "
            f"J-valid={100*jac_valid:.1f}% "
            f"full-valid={100*full_valid:.1f}% "
            f"mean negative-J={100*neg:.4f}% "
            f"mean degenerate={100*deg:.4f}% "
            f"mean curvature violations={curv:.2f}"
        )

    print("wrote", a.out)


if __name__ == "__main__":
    main()
