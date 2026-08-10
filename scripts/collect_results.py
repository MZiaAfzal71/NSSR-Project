"""Collect NSSR-V2 evaluation and safety CSVs into final summary tables.

Expected naming convention
--------------------------
Evaluation:
    results/eval_N5_baseline.csv
    results/eval_N5_safe.csv
    ...

Validation:
    results/validate_N5_baseline.csv
    results/validate_N5_safe.csv
    ...

The collector reports accuracy AND the two active safety mechanisms:
signed-Jacobian validity and radial cap-fold safety.  Curvature is intentionally
not part of the final summary.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re

import numpy as np


EVAL_RE = re.compile(r"eval_N(\d+)_(.+)\.csv$")
VAL_RE = re.compile(r"validate_N(\d+)_(.+)\.csv$")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def numeric_values(rows, key):
    vals = []
    for r in rows:
        v = _to_float(r.get(key))
        if math.isfinite(v):
            vals.append(v)
    return vals


def numeric_mean(rows, key):
    vals = numeric_values(rows, key)
    return float(np.mean(vals)) if vals else float("nan")


def numeric_min(rows, key):
    vals = numeric_values(rows, key)
    return float(np.min(vals)) if vals else float("nan")


def numeric_max(rows, key):
    vals = numeric_values(rows, key)
    return float(np.max(vals)) if vals else float("nan")


def pct_improvement(classical, learned, higher_is_better=False):
    if not (math.isfinite(classical) and math.isfinite(learned)):
        return float("nan")
    denom = max(abs(classical), 1e-12)
    if higher_is_better:
        return 100.0 * (learned - classical) / denom
    return 100.0 * (classical - learned) / denom


def discover(pattern, regex):
    found = {}
    for path in sorted(glob.glob(pattern)):
        match = regex.search(os.path.basename(path))
        if not match:
            continue
        key = (int(match.group(1)), match.group(2))
        found[key] = path
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/summary.csv")
    ap.add_argument(
        "--safety_out",
        default="",
        help="optional separate classical-vs-learned safety summary CSV",
    )
    a = ap.parse_args()

    eval_files = discover(
        os.path.join(a.results, "eval_N*_*.csv"),
        EVAL_RE,
    )
    val_files = discover(
        os.path.join(a.results, "validate_N*_*.csv"),
        VAL_RE,
    )

    keys = sorted(set(eval_files) | set(val_files))
    if not keys:
        raise RuntimeError(
            f"no eval_N*_*.csv or validate_N*_*.csv files found in {a.results}"
        )

    summary = []
    safety_rows = []

    for N, variant in keys:
        row = {
            "N": N,
            "variant": variant,
        }

        eval_path = eval_files.get((N, variant))
        if eval_path:
            erows = read_csv(eval_path)

            for metric in (
                "chamfer_l2",
                "chamfer_l1",
                "hausdorff",
                "hausdorff95",
                "normal_consistency",
            ):
                c = numeric_mean(erows, f"classical_{metric}")
                l = numeric_mean(erows, f"learned_{metric}")

                row[f"classical_{metric}"] = c
                row[metric] = l
                row[f"{metric}_improvement_pct"] = pct_improvement(
                    c,
                    l,
                    higher_is_better=(metric == "normal_consistency"),
                )

            row["learned_c1_min"] = numeric_min(
                erows, "learned_c1_min"
            )

        val_path = val_files.get((N, variant))
        if val_path:
            vrows = read_csv(val_path)

            row.update({
                "classical_j_valid_rate": numeric_mean(
                    vrows, "classical_geom_jacobian_valid"
                ),
                "j_valid_rate": numeric_mean(
                    vrows, "learned_geom_jacobian_valid"
                ),
                "classical_cap_safe_rate": numeric_mean(
                    vrows, "classical_geom_cap_safe"
                ),
                "cap_safe_rate": numeric_mean(
                    vrows, "learned_geom_cap_safe"
                ),
                "classical_safe_rate": numeric_mean(
                    vrows, "classical_geom_safe"
                ),
                "safe_rate": numeric_mean(
                    vrows, "learned_geom_safe"
                ),
                "classical_negative_fraction": numeric_mean(
                    vrows, "classical_geom_negative_fraction"
                ),
                "negative_fraction": numeric_mean(
                    vrows, "learned_geom_negative_fraction"
                ),
                "classical_degenerate_fraction": numeric_mean(
                    vrows, "classical_geom_degenerate_fraction"
                ),
                "degenerate_fraction": numeric_mean(
                    vrows, "learned_geom_degenerate_fraction"
                ),
                "classical_cap_fold_mean": numeric_mean(
                    vrows, "classical_geom_cap_fold_max"
                ),
                "cap_fold_mean": numeric_mean(
                    vrows, "learned_geom_cap_fold_max"
                ),
                "classical_cap_fold_worst": numeric_max(
                    vrows, "classical_geom_cap_fold_max"
                ),
                "cap_fold_worst": numeric_max(
                    vrows, "learned_geom_cap_fold_max"
                ),
                "minimum_signed_jacobian": numeric_min(
                    vrows, "learned_geom_minimum_signed_jacobian"
                ),
                "minimum_area_scale": numeric_min(
                    vrows, "learned_geom_minimum_area_scale"
                ),
            })

            safety_rows.append({
                "N": N,
                "variant": variant,
                "classical_j_valid_rate": row["classical_j_valid_rate"],
                "learned_j_valid_rate": row["j_valid_rate"],
                "classical_cap_safe_rate": row["classical_cap_safe_rate"],
                "learned_cap_safe_rate": row["cap_safe_rate"],
                "classical_safe_rate": row["classical_safe_rate"],
                "learned_safe_rate": row["safe_rate"],
                "classical_mean_negative_J_pct":
                    100.0 * row["classical_negative_fraction"],
                "learned_mean_negative_J_pct":
                    100.0 * row["negative_fraction"],
                "classical_cap_fold_worst":
                    row["classical_cap_fold_worst"],
                "learned_cap_fold_worst":
                    row["cap_fold_worst"],
            })

        summary.append(row)

    fields = [
        "N",
        "variant",
        "classical_chamfer_l2",
        "chamfer_l2",
        "chamfer_l2_improvement_pct",
        "classical_chamfer_l1",
        "chamfer_l1",
        "chamfer_l1_improvement_pct",
        "classical_hausdorff",
        "hausdorff",
        "hausdorff_improvement_pct",
        "classical_hausdorff95",
        "hausdorff95",
        "hausdorff95_improvement_pct",
        "classical_normal_consistency",
        "normal_consistency",
        "normal_consistency_improvement_pct",
        "learned_c1_min",
        "classical_j_valid_rate",
        "j_valid_rate",
        "classical_cap_safe_rate",
        "cap_safe_rate",
        "classical_safe_rate",
        "safe_rate",
        "classical_negative_fraction",
        "negative_fraction",
        "classical_degenerate_fraction",
        "degenerate_fraction",
        "classical_cap_fold_mean",
        "cap_fold_mean",
        "classical_cap_fold_worst",
        "cap_fold_worst",
        "minimum_signed_jacobian",
        "minimum_area_scale",
    ]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"wrote {a.out}\n")

    for r in summary:
        print(
            f"N={r['N']:>2} {r['variant']:<14s} "
            f"CD={r.get('chamfer_l2', float('nan')):.6f} "
            f"({r.get('chamfer_l2_improvement_pct', float('nan')):+.1f}%) "
            f"H95={r.get('hausdorff95', float('nan')):.6f} "
            f"SAFE={100*r.get('safe_rate', float('nan')):.1f}% "
            f"J={100*r.get('j_valid_rate', float('nan')):.1f}% "
            f"cap={100*r.get('cap_safe_rate', float('nan')):.1f}% "
            f"worst-fold={r.get('cap_fold_worst', float('nan')):.6f}"
        )

    safety_out = a.safety_out
    if safety_out:
        os.makedirs(os.path.dirname(safety_out) or ".", exist_ok=True)
        if safety_rows:
            with open(safety_out, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(safety_rows[0].keys()),
                )
                writer.writeheader()
                writer.writerows(safety_rows)
            print(f"\nwrote {safety_out}")


if __name__ == "__main__":
    main()
