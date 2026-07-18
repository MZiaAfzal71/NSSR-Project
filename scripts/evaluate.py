"""Evaluate a trained model vs the classical baseline on a test split.

Usage:  python scripts/evaluate.py --data data/synthetic --N 7 \
            --ckpt runs/exp1/best.pt --out results/eval_N7.csv
"""
import sys, os, argparse, pickle, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from nssr.train import to_torch, forward_object
from nssr.geometry import hermite_surface, zero_params, surface_points, \
    surface_normals, tangent_field
from nssr.networks import ParamNet
from nssr.metrics import evaluate_surface, c1_diagnostic

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--N", type=int, default=7)
    ap.add_argument("--ckpt", default="runs/exp1/best.pt")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n_u", type=int, default=32)
    ap.add_argument("--out", default="results/eval.csv")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(a.data, f"test_N{a.N}.pkl"), "rb") as f:
        test = pickle.load(f)
    net = ParamNet().to(dev)
    net.load_state_dict(torch.load(a.ckpt, map_location=dev))
    net.eval()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows = []
    with torch.no_grad():
        for i, s in enumerate(test):
            obj = to_torch(s, a.m, dev, torch.float32, seed=99000 + i)
            # classical
            p0 = zero_params(obj["R"].shape[0], a.m, dev, torch.float32)
            S0 = hermite_surface(obj["R"], obj["Z"], obj["RB"], obj["RC"],
                                 obj["Bh"], obj["Th"], p0, n_u=a.n_u)
            m0 = evaluate_surface(surface_points(S0), obj["gt_pts"],
                                  surface_normals(S0).reshape(-1, 3),
                                  obj["gt_normals"])
            # learned
            S1, pts, nrms, params = forward_object(net, obj, n_u=a.n_u)
            m1 = evaluate_surface(pts, obj["gt_pts"], nrms, obj["gt_normals"])
            gR, gZ = tangent_field(obj["R"], obj["Z"], obj["RB"], obj["RC"],
                                   obj["Bh"], obj["Th"], params)
            c1 = c1_diagnostic(gR, gZ)["global_min"]
            rows.append({"idx": i,
                         **{f"classical_{k}": v for k, v in m0.items()},
                         **{f"learned_{k}": v for k, v in m1.items()},
                         "learned_c1_min": c1})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    for key in ("chamfer_l2", "chamfer_l1", "hausdorff", "normal_consistency"):
        c = np.mean([r[f"classical_{key}"] for r in rows])
        l = np.mean([r[f"learned_{key}"] for r in rows])
        print(f"{key:20s} classical {c:.6f}   learned {l:.6f}   "
              f"improvement {100*(c-l)/max(c,1e-12):+.1f}%")
    print("min C1 diagnostic over test set:",
          min(r["learned_c1_min"] for r in rows))
    print("wrote", a.out)

if __name__ == "__main__":
    main()
