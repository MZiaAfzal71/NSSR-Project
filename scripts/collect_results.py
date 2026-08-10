"""Collect final NSSR-V2 multi-N safety/projection results.

Input files produced by ``run_full_sweep.py``:
    N{N}_validate.csv
    N{N}_projection.csv

The final table reports raw classical/learned accuracy, raw sampled safety,
post-projection safety, activation/stage rates, retained alpha by stage, and
projection-induced reconstruction error.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re

import numpy as np


VAL_RE = re.compile(r"N(\d+)_validate\.csv$")
PROJ_RE = re.compile(r"N(\d+)_projection\.csv$")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fval(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def values(rows, key):
    out = []
    for r in rows:
        x = fval(r.get(key))
        if math.isfinite(x):
            out.append(x)
    return out


def mean(rows, key):
    v = values(rows, key)
    return float(np.mean(v)) if v else float("nan")


def vmax(rows, key):
    v = values(rows, key)
    return float(np.max(v)) if v else float("nan")


def vmin(rows, key):
    v = values(rows, key)
    return float(np.min(v)) if v else float("nan")


def median(rows, key):
    v = values(rows, key)
    return float(np.median(v)) if v else float("nan")


def discover(results, pattern, regex):
    found = {}
    for path in glob.glob(os.path.join(results, pattern)):
        m = regex.search(os.path.basename(path))
        if m:
            found[int(m.group(1))] = path
    return found


def stage_rows(rows, stage):
    return [r for r in rows if r.get("projection_stage") == stage]


def pct_delta(old, new):
    if not (math.isfinite(old) and math.isfinite(new)):
        return float("nan")
    if abs(old) < 1e-12:
        return float("nan")
    return 100.0 * (new - old) / abs(old)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--results", default="results/final_sweep")
    ap.add_argument("--out", default="results/final_sweep/summary.csv")
    a = ap.parse_args()

    vals = discover(a.results, "N*_validate.csv", VAL_RE)
    projs = discover(a.results, "N*_projection.csv", PROJ_RE)
    Ns = sorted(set(vals) | set(projs))
    if not Ns:
        raise RuntimeError(f"no N*_validate/projection CSVs found in {a.results}")

    summary = []

    for N in Ns:
        row = {"N": N}

        if N in vals:
            vr = read_csv(vals[N])

            # Accuracy from the shared validator.
            for metric in (
                "chamfer_l2",
                "chamfer_l1",
                "hausdorff",
                "hausdorff95",
                "normal_consistency",
            ):
                ck = f"classical_{metric}"
                lk = f"learned_{metric}"
                if ck in vr[0]:
                    row[ck] = mean(vr, ck)
                if lk in vr[0]:
                    row[lk] = mean(vr, lk)

            # Canonical raw safety.
            row.update({
                "classical_j_valid_rate": mean(
                    vr, "classical_geom_jacobian_valid"
                ),
                "classical_cap_safe_rate": mean(
                    vr, "classical_geom_cap_safe"
                ),
                "classical_safe_rate": mean(
                    vr, "classical_geom_safe"
                ),
                "raw_j_valid_rate": mean(
                    vr, "learned_geom_jacobian_valid"
                ),
                "raw_cap_safe_rate": mean(
                    vr, "learned_geom_cap_safe"
                ),
                "raw_safe_rate": mean(
                    vr, "learned_geom_safe"
                ),
                "raw_mean_negative_j_fraction": mean(
                    vr, "learned_geom_negative_fraction"
                ),
                "raw_mean_degenerate_fraction": mean(
                    vr, "learned_geom_degenerate_fraction"
                ),
                "raw_mean_cap_fold": mean(
                    vr, "learned_geom_cap_fold_max"
                ),
                "raw_worst_cap_fold": vmax(
                    vr, "learned_geom_cap_fold_max"
                ),
            })

        if N in projs:
            pr = read_csv(projs[N])
            projected = [r for r in pr if fval(r.get("projection_activated")) > 0.5]

            raw_cd = mean(pr, "raw_chamfer_l2")
            post_cd = mean(pr, "post_chamfer_l2")
            raw_h = mean(pr, "raw_hausdorff")
            post_h = mean(pr, "post_hausdorff")

            row.update({
                "projection_raw_j_valid_rate": mean(pr, "raw_j_valid"),
                "projection_raw_cap_safe_rate": mean(pr, "raw_cap_safe"),
                "projection_raw_safe_rate": mean(pr, "raw_safe"),
                "post_j_valid_rate": mean(pr, "post_j_valid"),
                "post_cap_safe_rate": mean(pr, "post_cap_safe"),
                "post_safe_rate": mean(pr, "post_safe"),
                "projection_activation_rate": mean(
                    pr, "projection_activated"
                ),
                "projected_object_count": len(projected),
                "alpha_mean_projected": mean(projected, "alpha") if projected else float("nan"),
                "alpha_median_projected": median(projected, "alpha") if projected else float("nan"),
                "alpha_min_projected": vmin(projected, "alpha") if projected else float("nan"),
                "raw_chamfer_l2_projection_eval": raw_cd,
                "post_chamfer_l2": post_cd,
                "chamfer_l2_projection_delta": post_cd - raw_cd,
                "chamfer_l2_projection_delta_pct": pct_delta(raw_cd, post_cd),
                "raw_hausdorff_projection_eval": raw_h,
                "post_hausdorff": post_h,
                "hausdorff_projection_delta": post_h - raw_h,
                "raw_worst_cap_fold_projection_eval": vmax(pr, "raw_cap_fold"),
                "post_worst_cap_fold": vmax(pr, "post_cap_fold"),
            })

            for stage in ("cap_all", "tangent", "all"):
                sr = stage_rows(pr, stage)
                row[f"stage_{stage}_count"] = len(sr)
                row[f"stage_{stage}_rate"] = len(sr) / len(pr) if pr else float("nan")
                row[f"stage_{stage}_alpha_mean"] = mean(sr, "alpha") if sr else float("nan")
                row[f"stage_{stage}_alpha_median"] = median(sr, "alpha") if sr else float("nan")
                row[f"stage_{stage}_alpha_min"] = vmin(sr, "alpha") if sr else float("nan")

        # Cross-check raw rates from validate vs projection evaluator.
        if all(k in row for k in (
            "raw_j_valid_rate", "projection_raw_j_valid_rate",
            "raw_cap_safe_rate", "projection_raw_cap_safe_rate",
            "raw_safe_rate", "projection_raw_safe_rate",
        )):
            diffs = [
                abs(row["raw_j_valid_rate"] - row["projection_raw_j_valid_rate"]),
                abs(row["raw_cap_safe_rate"] - row["projection_raw_cap_safe_rate"]),
                abs(row["raw_safe_rate"] - row["projection_raw_safe_rate"]),
            ]
            row["raw_evaluators_match"] = int(max(diffs) < 1e-12)

        summary.append(row)

    # Stable, paper-friendly ordering; extras are appended automatically.
    preferred = [
        "N",
        "classical_chamfer_l2", "learned_chamfer_l2",
        "raw_j_valid_rate", "raw_cap_safe_rate", "raw_safe_rate",
        "post_j_valid_rate", "post_cap_safe_rate", "post_safe_rate",
        "projection_activation_rate", "projected_object_count",
        "raw_chamfer_l2_projection_eval", "post_chamfer_l2",
        "chamfer_l2_projection_delta", "chamfer_l2_projection_delta_pct",
        "alpha_mean_projected", "alpha_median_projected", "alpha_min_projected",
        "stage_cap_all_count", "stage_cap_all_alpha_mean",
        "stage_tangent_count", "stage_tangent_alpha_mean",
        "stage_all_count", "stage_all_alpha_mean",
        "raw_mean_negative_j_fraction", "raw_mean_degenerate_fraction",
        "raw_mean_cap_fold", "raw_worst_cap_fold", "post_worst_cap_fold",
        "raw_evaluators_match",
    ]
    all_fields = set()
    for r in summary:
        all_fields.update(r.keys())
    fields = preferred + sorted(all_fields - set(preferred))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)

    print(f"wrote {a.out}\n")
    print(
        " N | raw J | raw cap | raw SAFE | post SAFE | proj% | "
        "raw CD | post CD | dCD% | stages(cap/tan/all)"
    )
    print("-" * 104)
    for r in summary:
        print(
            f"{r['N']:2d} | "
            f"{100*r.get('raw_j_valid_rate', float('nan')):5.1f}% | "
            f"{100*r.get('raw_cap_safe_rate', float('nan')):7.1f}% | "
            f"{100*r.get('raw_safe_rate', float('nan')):7.1f}% | "
            f"{100*r.get('post_safe_rate', float('nan')):8.1f}% | "
            f"{100*r.get('projection_activation_rate', float('nan')):5.1f}% | "
            f"{r.get('raw_chamfer_l2_projection_eval', float('nan')):.6f} | "
            f"{r.get('post_chamfer_l2', float('nan')):.6f} | "
            f"{r.get('chamfer_l2_projection_delta_pct', float('nan')):+5.2f}% | "
            f"{r.get('stage_cap_all_count', 0):d}/"
            f"{r.get('stage_tangent_count', 0):d}/"
            f"{r.get('stage_all_count', 0):d}"
        )

        if r.get("raw_evaluators_match") == 0:
            print(
                f"    WARNING N={r['N']}: validate.py and "
                "evaluate_projection.py raw safety rates disagree"
            )


if __name__ == "__main__":
    main()
