"""Validate NSSR-V2 reconstruction safety and accuracy.

Uses ``nssr.safety`` as the single source of truth for:
    SAFE = Jacobian-valid AND cap-safe
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
    geometry_from_params,
    geometry_safety_summary,
    params_from_net,
    reconstruction_metrics,
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
    vals = [float(r[key]) for r in rows if key in r]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--split", default="test")
    ap.add_argument("--N", type=int, default=15)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n_u", type=int, default=32)
    ap.add_argument("--gt_sub", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/validate.csv")
    ap.add_argument("--c_bound", type=float, default=1.0)
    ap.add_argument("--no_learn_heights", action="store_true")
    ap.add_argument("--max_cap_fold", type=float, default=1e-3)
    ap.add_argument("--seed_base", type=int, default=40000)
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
                m=a.m,
                device=device,
                dtype=dtype,
                gt_subsample=a.gt_sub,
                seed=a.seed_base + i,
            )

            classical, reference_normal = (
                classical_geometry_and_reference(obj, a.n_u)
            )
            csum = geometry_safety_summary(
                classical, obj, a.max_cap_fold
            )
            cmetrics = reconstruction_metrics(obj, classical)

            row = {
                "idx": i,
                **{f"classical_{k}": v for k, v in cmetrics.items()},
                **{f"classical_geom_{k}": v for k, v in csum.items()},
            }

            if net is not None:
                params = params_from_net(net, obj)
                learned = geometry_from_params(
                    obj, params, a.n_u, reference_normal
                )
                lsum = geometry_safety_summary(
                    learned, obj, a.max_cap_fold
                )
                lmetrics = reconstruction_metrics(obj, learned)

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
    with open(a.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
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
