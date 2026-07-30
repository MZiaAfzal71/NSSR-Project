"""Ablation: how much of NSSR's gain is reachable by simply RETUNING THE
CLASSICAL CONSTANTS globally, with no network and no per-point adaptation?

This is the first question a reviewer will ask ("the classical constants
were just badly chosen -- fix them and you get the same benefit"), so it
needs a rigorous answer under exactly the same protocol as the main
evaluation, not a back-of-envelope estimate.

Protocol (identical to scripts/evaluate.py):
  * same to_torch preprocessing, same m / n_u,
  * same metrics via nssr.metrics.evaluate_surface,
  * global constants are FIT ON THE TRAIN SPLIT and scored on the TEST
    split -- never tuned on the data they are scored on,
  * three-way comparison: classical (s=0) / best global constants /
    learned network.

Usage:
    python scripts/ablation_global_constants.py --data data/synthetic \
        --N 7 --ckpt runs/exp_v2/best.pt --out results/ablation_N7.csv
"""
import sys, os, argparse, pickle, csv, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

from nssr.train import to_torch, forward_object
from nssr.geometry import hermite_surface, zero_params, surface_points, \
    surface_normals
from nssr.networks import ParamNet
from nssr.metrics import evaluate_surface


def const_params(N, m, sa, sb, st, device, dtype):
    p = zero_params(N, m, device=device, dtype=dtype)
    p["s_a"] = p["s_a"] + sa
    p["s_b"] = p["s_b"] + sb
    p["s_tau"] = p["s_tau"] + st
    return p


def surf_with(obj, params, n_u):
    return hermite_surface(obj["R"], obj["Z"], obj["RB"], obj["RC"],
                           obj["Bh"], obj["Th"], params, n_u=n_u,
                           closed_top=obj.get("closed_top", True),
                           base_circular=obj.get("base_circular", True),
                           crown_circular=obj.get("crown_circular", True))


@torch.no_grad()
def mean_chamfer(objs, sa, sb, st, n_u):
    tot = 0.0
    for obj in objs:
        N, m = obj["R"].shape[0], obj["R"].shape[1]
        p = const_params(N, m, sa, sb, st, obj["R"].device, obj["R"].dtype)
        S = surf_with(obj, p, n_u)
        tot += evaluate_surface(surface_points(S), obj["gt_pts"])["chamfer_l2"]
    return tot / len(objs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--N", type=int, default=7)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n_u", type=int, default=32)
    ap.add_argument("--fit_objects", type=int, default=40,
                    help="how many TRAIN objects to fit the constants on")
    ap.add_argument("--coarse", type=int, default=7,
                    help="grid points per axis in the coarse sweep")
    ap.add_argument("--out", default="results/ablation_global.csv")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32

    with open(os.path.join(a.data, f"train_N{a.N}.pkl"), "rb") as f:
        train = pickle.load(f)[:a.fit_objects]
    with open(os.path.join(a.data, f"test_N{a.N}.pkl"), "rb") as f:
        test = pickle.load(f)

    print(f"preprocessing {len(train)} fit / {len(test)} test objects ...")
    fit_objs = [to_torch(s, a.m, dev, dt, seed=i) for i, s in enumerate(train)]
    test_objs = [to_torch(s, a.m, dev, dt, seed=5000 + i)
                 for i, s in enumerate(test)]

    # ---- fit global constants on TRAIN ------------------------------------
    # coarse grid, then a local refinement around the best point.
    lo, hi = -1.5, 1.5
    g = np.linspace(lo, hi, a.coarse)
    best = (float("inf"), 0.0, 0.0, 0.0)
    print("coarse grid search on train split ...")
    for sa, st in itertools.product(g, g):
        v = mean_chamfer(fit_objs, sa, -sa, st, a.n_u)
        if v < best[0]:
            best = (v, sa, -sa, st)
    step = (hi - lo) / (a.coarse - 1)
    print(f"  coarse best: chamfer {best[0]:.6f} "
          f"(s_a={best[1]:+.2f} s_tau={best[3]:+.2f}) -> refining")
    for _ in range(2):
        sa0, st0 = best[1], best[3]
        step /= 2
        for sa in (sa0 - step, sa0, sa0 + step):
            for st in (st0 - step, st0, st0 + step):
                v = mean_chamfer(fit_objs, sa, -sa, st, a.n_u)
                if v < best[0]:
                    best = (v, sa, -sa, st)
    _, SA, SB, ST = best
    print(f"fitted global constants: s_a={SA:+.3f} s_b={SB:+.3f} s_tau={ST:+.3f}")
    if min(abs(SA - lo), abs(SA - hi), abs(ST - lo), abs(ST - hi)) < 1e-6:
        print("  WARNING: optimum is on the grid boundary -- widen the range;"
              " the fitted constants may be exploiting resolution rather than"
              " genuinely improving the surface.")

    # ---- score all three on TEST ------------------------------------------
    net = None
    if a.ckpt and os.path.exists(a.ckpt):
        net = ParamNet().to(device=dev, dtype=dt)
        net.load_state_dict(torch.load(a.ckpt, map_location=dev))
        net.eval()
    else:
        print("(no --ckpt given: reporting classical vs global only)")

    rows = []
    with torch.no_grad():
        for i, obj in enumerate(test_objs):
            N, m = obj["R"].shape[0], obj["R"].shape[1]
            row = {"idx": i}
            p0 = zero_params(N, m, device=dev, dtype=dt)
            S0 = surf_with(obj, p0, a.n_u)
            m0 = evaluate_surface(surface_points(S0), obj["gt_pts"],
                                  surface_normals(S0).reshape(-1, 3),
                                  obj["gt_normals"])
            pg = const_params(N, m, SA, SB, ST, dev, dt)
            Sg = surf_with(obj, pg, a.n_u)
            mg = evaluate_surface(surface_points(Sg), obj["gt_pts"],
                                  surface_normals(Sg).reshape(-1, 3),
                                  obj["gt_normals"])
            row.update({f"classical_{k}": v for k, v in m0.items()})
            row.update({f"global_{k}": v for k, v in mg.items()})
            if net is not None:
                _, pts, nrms, _ = forward_object(net, obj, n_u=a.n_u)
                ml = evaluate_surface(pts, obj["gt_pts"], nrms,
                                      obj["gt_normals"])
                row.update({f"learned_{k}": v for k, v in ml.items()})
            rows.append(row)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    HIGHER = {"normal_consistency"}
    variants = ["classical", "global"] + (["learned"] if net else [])
    print(f"\n{'metric':22s} " + "".join(f"{v:>14s}" for v in variants)
          + "   improvement vs classical")
    for key in ("chamfer_l2", "chamfer_l1", "hausdorff", "hausdorff95",
                "normal_consistency"):
        if f"classical_{key}" not in rows[0]:
            continue
        vals = {v: float(np.mean([r[f"{v}_{key}"] for r in rows]))
                for v in variants}
        c = vals["classical"]
        imps = []
        for v in variants[1:]:
            imp = (100 * (vals[v] - c) / abs(c) if key in HIGHER
                   else 100 * (c - vals[v]) / abs(c))
            imps.append(f"{v} {imp:+.1f}%")
        print(f"{key:22s} " + "".join(f"{vals[v]:14.6f}" for v in variants)
              + "   " + "  ".join(imps))
    print(f"\nwrote {a.out}")
    print("\nThis is the number to cite for 'could you just retune the "
          "constants?' -- fit on train, scored on held-out test, identical "
          "metrics and resolution to the main evaluation.")


if __name__ == "__main__":
    main()
