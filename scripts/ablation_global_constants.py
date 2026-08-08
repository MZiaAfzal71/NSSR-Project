"""Ablation: learned NSSR vs globally retuned classical constants (V2-compatible)."""
import sys, os, argparse, pickle, csv, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from nssr.train import to_torch, forward_object
from nssr.geometry import hermite_surface, zero_params, surface_points, surface_normals
from nssr.networks import ParamNet
from nssr.metrics import evaluate_surface

def const_params(N, m, sa, sb, st, device, dtype):
    p = zero_params(N, m, device=device, dtype=dtype)
    p["s_a"] = p["s_a"] + sa
    p["s_b"] = p["s_b"] + sb
    p["s_tau"] = p["s_tau"] + st
    return p

def surf_with(obj, params, n_u):
    return hermite_surface(
        obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"], params,
        n_u=n_u,
        closed_top=obj.get("closed_top", True),
        base_circular=obj.get("base_circular", True),
        crown_circular=obj.get("crown_circular", True),
    )

@torch.no_grad()
def mean_chamfer(objs, sa, sb, st, n_u):
    tot = 0.0
    for obj in objs:
        N, m = obj["R"].shape[:2]
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
    ap.add_argument("--fit_objects", type=int, default=40)
    ap.add_argument("--coarse", type=int, default=7)
    ap.add_argument("--out", default="results/ablation_global.csv")
    ap.add_argument("--c_bound", type=float, default=1.0)
    ap.add_argument("--no_learn_heights", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32

    with open(os.path.join(a.data, f"train_N{a.N}.pkl"), "rb") as f:
        train = pickle.load(f)[:a.fit_objects]
    with open(os.path.join(a.data, f"test_N{a.N}.pkl"), "rb") as f:
        test = pickle.load(f)

    fit_objs = [to_torch(s, a.m, dev, dt, seed=i) for i, s in enumerate(train)]
    test_objs = [to_torch(s, a.m, dev, dt, seed=5000 + i) for i, s in enumerate(test)]

    lo, hi = -1.5, 1.5
    g = np.linspace(lo, hi, a.coarse)
    best = (float("inf"), 0.0, 0.0, 0.0)

    print("coarse grid search on train split ...")
    for sa, st in itertools.product(g, g):
        v = mean_chamfer(fit_objs, sa, -sa, st, a.n_u)
        if v < best[0]:
            best = (v, sa, -sa, st)

    step = (hi - lo) / (a.coarse - 1)
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

    net = None
    if a.ckpt and os.path.exists(a.ckpt):
        net = ParamNet(
            learn_heights=not a.no_learn_heights,
            c_bound=a.c_bound,
        ).to(device=dev, dtype=dt)
        net.load_state_dict(torch.load(a.ckpt, map_location=dev))
        net.eval()
    else:
        print("(no --ckpt given: reporting classical vs global only)")

    rows = []
    with torch.no_grad():
        for i, obj in enumerate(test_objs):
            N, m = obj["R"].shape[:2]
            row = {"idx": i}

            p0 = zero_params(N, m, device=dev, dtype=dt)
            S0 = surf_with(obj, p0, a.n_u)
            m0 = evaluate_surface(
                surface_points(S0), obj["gt_pts"],
                surface_normals(S0).reshape(-1, 3), obj["gt_normals"],
            )

            pg = const_params(N, m, SA, SB, ST, dev, dt)
            Sg = surf_with(obj, pg, a.n_u)
            mg = evaluate_surface(
                surface_points(Sg), obj["gt_pts"],
                surface_normals(Sg).reshape(-1, 3), obj["gt_normals"],
            )

            row.update({f"classical_{k}": v for k, v in m0.items()})
            row.update({f"global_{k}": v for k, v in mg.items()})

            if net is not None:
                _, pts, nrms, _, _geometry = forward_object(net, obj, n_u=a.n_u)
                ml = evaluate_surface(pts, obj["gt_pts"], nrms, obj["gt_normals"])
                row.update({f"learned_{k}": v for k, v in ml.items()})

            rows.append(row)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    higher = {"normal_consistency"}
    variants = ["classical", "global"] + (["learned"] if net is not None else [])
    for key in ("chamfer_l2", "chamfer_l1", "hausdorff", "hausdorff95", "normal_consistency"):
        if f"classical_{key}" not in rows[0]:
            continue
        vals = {v: float(np.mean([r[f"{v}_{key}"] for r in rows])) for v in variants}
        c = vals["classical"]
        pieces = []
        for v in variants[1:]:
            imp = (100 * (vals[v] - c) / max(abs(c), 1e-12)
                   if key in higher else
                   100 * (c - vals[v]) / max(abs(c), 1e-12))
            pieces.append(f"{v} {imp:+.1f}%")
        print(f"{key:22s} " + " ".join(f"{v}={vals[v]:.6f}" for v in variants)
              + "   " + "  ".join(pieces))

    print("wrote", a.out)

if __name__ == "__main__":
    main()
