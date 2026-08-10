"""Evaluate raw and post-projection NSSR-V2 safety.

Uses ``nssr.safety`` for both raw evaluation and projection, so raw safety
must match ``scripts/validate.py`` when checkpoint/options are identical.
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

from nssr.networks import ParamNet
from nssr.safety import (
    classical_geometry_and_reference,
    params_from_net,
    project_all_to_safe,
)
from nssr.train import to_torch


def _load_state_dict(path, device, dtype):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {
        k: v.to(device=device, dtype=dtype)
        for k, v in state.items()
    }


def _mean(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate raw and post-projection NSSR-V2 safety.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--split", default="test")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--n_u", type=int, default=16)
    ap.add_argument("--gt_sub", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed_base", type=int, default=40000)
    ap.add_argument("--c_bound", type=float, default=1.0)
    ap.add_argument("--no_learn_heights", action="store_true")
    ap.add_argument("--max_cap_fold", type=float, default=1e-3)
    ap.add_argument("--projection_iters", type=int, default=40)
    ap.add_argument("--fp64", action="store_true")
    a = ap.parse_args()

    if a.projection_iters < 1:
        raise SystemExit("--projection_iters must be >= 1")
    if a.max_cap_fold <= 0:
        raise SystemExit("--max_cap_fold must be > 0")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64 if a.fp64 else torch.float32

    path = os.path.join(a.data, f"{a.split}_N{a.N}.pkl")
    with open(path, "rb") as f:
        samples = pickle.load(f)

    if a.limit > 0:
        samples = samples[:a.limit]
    if not samples:
        raise RuntimeError(f"empty split: {path}")

    print("NSSR-V2 projection evaluation")
    print(f"  data               : {path}")
    print(f"  objects            : {len(samples)}")
    print(f"  checkpoint         : {a.ckpt}")
    print(f"  safety samples n_u : {a.n_u}")
    print(f"  c_bound            : {a.c_bound:g}")
    print(f"  cap threshold      : {a.max_cap_fold:g}")
    print("  safety source      : nssr.safety (shared with validate.py)")

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
                m=a.m,
                device=device,
                dtype=dtype,
                gt_subsample=a.gt_sub,
                seed=a.seed_base + i,
            )

            _, reference_normal = classical_geometry_and_reference(
                obj, a.n_u
            )
            params = params_from_net(net, obj)

            (
                _,
                alpha,
                raw_safety,
                raw_metrics,
                post_safety,
                post_metrics,
            ) = project_all_to_safe(
                obj,
                params,
                a.n_u,
                reference_normal,
                max_cap_fold=a.max_cap_fold,
                max_iter=a.projection_iters,
                with_metrics=True,
            )

            activated = alpha < 1.0 - 1e-9

            row = {
                "index": i,

                "raw_j_valid": int(raw_safety["jacobian_valid"]),
                "raw_cap_safe": int(raw_safety["cap_safe"]),
                "raw_safe": int(raw_safety["safe"]),
                "raw_negative_jacobian": raw_safety["negative_jacobian"],
                "raw_negative_fraction": raw_safety["negative_fraction"],
                "raw_degenerate": raw_safety["degenerate"],
                "raw_degenerate_fraction": raw_safety["degenerate_fraction"],
                "raw_minimum_signed_jacobian":
                    raw_safety["minimum_signed_jacobian"],
                "raw_minimum_area_scale":
                    raw_safety["minimum_area_scale"],
                "raw_base_cap_fold_max":
                    raw_safety["base_cap_fold_max"],
                "raw_crown_cap_fold_max":
                    raw_safety["crown_cap_fold_max"],
                "raw_cap_fold": raw_safety["cap_fold_max"],
                "raw_chamfer_l2":
                    raw_metrics.get("chamfer_l2", float("nan")),
                "raw_hausdorff":
                    raw_metrics.get("hausdorff", float("nan")),

                "projection_activated": int(activated),
                "alpha": float(alpha),
                "retained_percent": 100.0 * float(alpha),

                "post_j_valid": int(post_safety["jacobian_valid"]),
                "post_cap_safe": int(post_safety["cap_safe"]),
                "post_safe": int(post_safety["safe"]),
                "post_negative_jacobian":
                    post_safety["negative_jacobian"],
                "post_negative_fraction":
                    post_safety["negative_fraction"],
                "post_degenerate": post_safety["degenerate"],
                "post_degenerate_fraction":
                    post_safety["degenerate_fraction"],
                "post_minimum_signed_jacobian":
                    post_safety["minimum_signed_jacobian"],
                "post_minimum_area_scale":
                    post_safety["minimum_area_scale"],
                "post_base_cap_fold_max":
                    post_safety["base_cap_fold_max"],
                "post_crown_cap_fold_max":
                    post_safety["crown_cap_fold_max"],
                "post_cap_fold": post_safety["cap_fold_max"],
                "post_chamfer_l2":
                    post_metrics.get("chamfer_l2", float("nan")),
                "post_hausdorff":
                    post_metrics.get("hausdorff", float("nan")),
            }
            row["delta_chamfer_l2"] = (
                row["post_chamfer_l2"] - row["raw_chamfer_l2"]
            )
            rows.append(row)

            print(
                f"[{i:3d}] "
                f"raw {'SAFE' if raw_safety['safe'] else 'FAIL':4s} "
                f"J={'OK' if raw_safety['jacobian_valid'] else 'FAIL':4s} "
                f"cap={raw_safety['cap_fold_max']:.5f} "
                f"-> {'SAFE' if post_safety['safe'] else 'FAIL':4s} "
                f"{'alpha='+format(alpha,'.3f') if activated else 'unchanged'}"
            )

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    activated_rows = [
        r for r in rows if r["projection_activated"]
    ]
    alphas = [r["alpha"] for r in activated_rows]

    print("\nsummary")
    print(
        f"raw       "
        f"J-valid={100*_mean(rows,'raw_j_valid'):.1f}% "
        f"cap-safe={100*_mean(rows,'raw_cap_safe'):.1f}% "
        f"SAFE={100*_mean(rows,'raw_safe'):.1f}%"
    )
    print(
        f"projected "
        f"J-valid={100*_mean(rows,'post_j_valid'):.1f}% "
        f"cap-safe={100*_mean(rows,'post_cap_safe'):.1f}% "
        f"SAFE={100*_mean(rows,'post_safe'):.1f}%"
    )
    print(
        f"projection activation="
        f"{100*_mean(rows,'projection_activated'):.1f}% "
        f"({len(activated_rows)}/{len(rows)})"
    )

    if alphas:
        print(
            f"alpha on projected objects: "
            f"mean={np.mean(alphas):.4f} "
            f"median={np.median(alphas):.4f} "
            f"min={np.min(alphas):.4f}"
        )
    else:
        print("alpha on projected objects: no projection required")

    print(
        f"Chamfer L2 mean: "
        f"raw={_mean(rows,'raw_chamfer_l2'):.6f} "
        f"projected={_mean(rows,'post_chamfer_l2'):.6f} "
        f"delta={_mean(rows,'delta_chamfer_l2'):+.6f}"
    )
    print(
        f"worst cap-fold: "
        f"raw={max(r['raw_cap_fold'] for r in rows):.6f} "
        f"projected={max(r['post_cap_fold'] for r in rows):.6f}"
    )
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
