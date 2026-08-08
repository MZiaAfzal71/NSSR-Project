"""Collect NSSR-V2 CSV outputs into a compact summary CSV."""

import argparse
import csv
import glob
import os
import re
import numpy as np


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def numeric_mean(rows, key):
    vals = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except (KeyError, TypeError, ValueError):
            pass
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/summary.csv")
    a = ap.parse_args()

    summary = []

    for path in sorted(glob.glob(os.path.join(a.results, "eval_N*_*.csv"))):
        rows = read_csv(path)
        m = re.search(r"eval_N(\d+)_(.+)\.csv$", os.path.basename(path))
        if not m or not rows:
            continue
        N, variant = int(m.group(1)), m.group(2)
        summary.append({
            "N": N,
            "variant": variant,
            "chamfer_l2": numeric_mean(rows, "learned_chamfer_l2"),
            "chamfer_l1": numeric_mean(rows, "learned_chamfer_l1"),
            "hausdorff": numeric_mean(rows, "learned_hausdorff"),
            "hausdorff95": numeric_mean(rows, "learned_hausdorff95"),
            "normal_consistency": numeric_mean(rows, "learned_normal_consistency"),
        })

    val_lookup = {}
    for path in sorted(glob.glob(os.path.join(a.results, "validate_N*_*.csv"))):
        rows = read_csv(path)
        m = re.search(r"validate_N(\d+)_(.+)\.csv$", os.path.basename(path))
        if not m or not rows:
            continue
        key = (int(m.group(1)), m.group(2))
        val_lookup[key] = {
            "valid_rate": numeric_mean(rows, "learned_geom_valid"),
            "negative_fraction": numeric_mean(rows, "learned_geom_negative_fraction"),
            "degenerate_fraction": numeric_mean(rows, "learned_geom_degenerate_fraction"),
            "max_abs_curvature": numeric_mean(rows, "learned_geom_max_abs_curvature"),
        }

    for row in summary:
        row.update(val_lookup.get((row["N"], row["variant"]), {}))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fields = [
        "N", "variant", "chamfer_l2", "chamfer_l1", "hausdorff",
        "hausdorff95", "normal_consistency", "valid_rate",
        "negative_fraction", "degenerate_fraction", "max_abs_curvature",
    ]

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    print("wrote", a.out)
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
