"""Quantitative base/crown analysis for NSSR-V2 supplementary material.

For each test object this reports classical, raw learned, and optionally
failure-aware projected errors separately on:
  - base cap patch,
  - crown cap patch,
  - full surface,
plus base/crown cap-turnback and projection stage.

Nearest-neighbor distance is measured from reconstructed samples to GT.
"""
from __future__ import annotations
import argparse, csv, os, pickle, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

from nssr.networks import ParamNet
from nssr.safety import (
    classical_geometry_and_reference,
    geometry_from_params,
    geometry_safety_summary,
    params_from_net,
    project_staged_to_safe,
)
from nssr.train import to_torch


def state_dict(path, dev):
    s = torch.load(path, map_location=dev)
    return s["state_dict"] if isinstance(s, dict) and "state_dict" in s else s


def mean_nn(P, gt):
    if P.size == 0:
        return float("nan")
    return float(cKDTree(gt).query(P.reshape(-1,3))[0].mean())


def geom_errors(geom, gt, closed_top=True):
    S = geom.surface.xyz.detach().cpu().numpy()
    full = mean_nn(S, gt)
    base = mean_nn(S[0], gt)
    crown = mean_nn(S[-1], gt) if closed_top else float("nan")
    return full, base, crown


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--data", default="data/real")
    ap.add_argument("--split", default="test")
    ap.add_argument("--N", type=int, default=15)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--n_u", type=int, default=16)
    ap.add_argument("--c_bound", type=float, default=1.0)
    ap.add_argument("--max_cap_fold", type=float, default=1e-3)
    ap.add_argument("--project_safe", action="store_true")
    ap.add_argument("--out", default="results/base_crown.csv")
    a = ap.parse_args()

    path = os.path.join(a.data, f"{a.split}_N{a.N}.pkl")
    with open(path, "rb") as f:
        samples = pickle.load(f)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32
    net = ParamNet(c_bound=a.c_bound).to(device=dev, dtype=dt)
    net.load_state_dict(state_dict(a.ckpt, dev))
    net.eval()

    rows = []
    with torch.no_grad():
        for i, sample in enumerate(samples):
            obj = to_torch(
                sample, m=a.m, device=dev, dtype=dt,
                gt_subsample=20000, seed=40000+i
            )
            gt = obj["gt_pts"].cpu().numpy()
            classical, ref = classical_geometry_and_reference(obj, a.n_u)
            params = params_from_net(net, obj)
            learned = geometry_from_params(obj, params, a.n_u, ref)

            cs = geometry_safety_summary(classical, obj, a.max_cap_fold)
            ls = geometry_safety_summary(learned, obj, a.max_cap_fold)
            ce = geom_errors(classical, gt, obj.get("closed_top", True))
            le = geom_errors(learned, gt, obj.get("closed_top", True))

            stage, alpha = "none", 1.0
            projected = learned
            ps = ls
            pe = le
            if a.project_safe:
                params_p, alpha, stage, _, _, ps, _ = project_staged_to_safe(
                    obj, params, a.n_u, ref,
                    max_cap_fold=a.max_cap_fold,
                    max_iter=40, with_metrics=False
                )
                projected = geometry_from_params(obj, params_p, a.n_u, ref)
                pe = geom_errors(projected, gt, obj.get("closed_top", True))

            row = {
                "index": i,
                "path": sample.get("path", ""),
                "projection_stage": stage,
                "alpha": alpha,
                "classical_full_err": ce[0],
                "classical_base_err": ce[1],
                "classical_crown_err": ce[2],
                "learned_full_err": le[0],
                "learned_base_err": le[1],
                "learned_crown_err": le[2],
                "projected_full_err": pe[0],
                "projected_base_err": pe[1],
                "projected_crown_err": pe[2],
                "learned_base_cap_fold": ls["base_cap_fold_max"],
                "learned_crown_cap_fold": ls["crown_cap_fold_max"],
                "projected_base_cap_fold": ps["base_cap_fold_max"],
                "projected_crown_cap_fold": ps["crown_cap_fold_max"],
                "raw_safe": int(ls["safe"]),
                "post_safe": int(ps["safe"]),
            }
            rows.append(row)
            print(
                f"[{i:3d}] base {ce[1]:.5f}->{le[1]:.5f}->{pe[1]:.5f} "
                f"| crown {ce[2]:.5f}->{le[2]:.5f}->{pe[2]:.5f} "
                f"| {stage} a={alpha:.3f}"
            )

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def avg(k):
        v = [float(r[k]) for r in rows if np.isfinite(float(r[k]))]
        return np.mean(v) if v else float("nan")

    print("\nsummary")
    for region in ("full","base","crown"):
        c, l, p = (avg(f"{x}_{region}_err") for x in ("classical","learned","projected"))
        imp = 100*(c-l)/max(c,1e-12)
        print(
            f"{region:5s}: classical={c:.6f} learned={l:.6f} "
            f"projected={p:.6f} learned gain={imp:+.1f}%"
        )
    print("wrote", a.out)


if __name__ == "__main__":
    main()
