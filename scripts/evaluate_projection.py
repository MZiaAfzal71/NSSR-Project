"""Batch evaluation of NSSR-V2 joint safety projection."""

from __future__ import annotations
import argparse, csv, os, pickle, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nssr.geometry import evaluate_geometry, surface_points
from nssr.losses import cap_radial_fold_max, intentional_pole_mask
from nssr.metrics import evaluate_surface
from nssr.networks import ParamNet, contour_features
from nssr.train import to_torch


def _zero_params_like(obj):
    N, m = obj["R"].shape[:2]
    kw = dict(device=obj["R"].device, dtype=obj["R"].dtype)
    return {
        "s_a": torch.zeros(N, m, **kw),
        "s_b": torch.zeros(N, m, **kw),
        "s_tau": torch.zeros(N, m, **kw),
        "s_fB": torch.zeros(m, **kw),
        "s_fC": torch.zeros(m, **kw),
        "s_bh": torch.zeros((), **kw),
        "s_th": torch.zeros((), **kw),
    }


def _scaled_params(params, alpha):
    return {k: v * alpha for k, v in params.items()}


def _load_state_dict(path, device, dtype):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {k: v.to(device=device, dtype=dtype) for k, v in state.items()}


@torch.no_grad()
def _classical_reference_normal(obj, n_u):
    geom = evaluate_geometry(
        obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"],
        _zero_params_like(obj),
        n_u=n_u,
        closed_top=obj.get("closed_top", True),
        base_circular=obj.get("base_circular", True),
        crown_circular=obj.get("crown_circular", True),
        compute_jacobian=False,
        compute_curvature=False,
        run_validation=False,
    )
    return geom.surface.normals.detach()


@torch.no_grad()
def _evaluate_state(obj, params, n_u, reference_normal, max_cap_fold):
    geom = evaluate_geometry(
        obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"],
        params,
        n_u=n_u,
        closed_top=obj.get("closed_top", True),
        base_circular=obj.get("base_circular", True),
        crown_circular=obj.get("crown_circular", True),
        compute_jacobian=True,
        compute_curvature=False,
        run_validation=False,
        reference_normal=reference_normal,
    )

    jac = geom.jacobian
    evaluable = ~intentional_pole_mask(
        geom.surface.xyz,
        closed_top=obj.get("closed_top", True),
    )
    neg = jac.flipped_mask & evaluable
    deg = jac.degenerate_mask & evaluable
    denom = max(int(evaluable.sum().item()), 1)

    neg_count = int(neg.sum().item())
    deg_count = int(deg.sum().item())
    cap_fold = float(
        cap_radial_fold_max(
            geom.surface.xyz, obj["RB"], obj["RC"],
            closed_top=obj.get("closed_top", True),
        ).item()
    )

    j_valid = neg_count == 0 and deg_count == 0
    cap_safe = cap_fold <= max_cap_fold

    pts = surface_points(geom.surface.xyz)
    nrms = geom.surface.normals.reshape(-1, 3)
    metrics = evaluate_surface(pts, obj["gt_pts"], nrms, obj["gt_normals"])

    return {
        "j_valid": bool(j_valid),
        "cap_safe": bool(cap_safe),
        "safe": bool(j_valid and cap_safe),
        "negative_fraction": neg_count / denom,
        "degenerate_fraction": deg_count / denom,
        "cap_fold": cap_fold,
        "chamfer_l2": float(metrics["chamfer_l2"]),
        "hausdorff": float(metrics["hausdorff"]),
    }


@torch.no_grad()
def _project_to_safe(obj, params, n_u, reference_normal, max_cap_fold, max_iter=40):
    raw = _evaluate_state(obj, params, n_u, reference_normal, max_cap_fold)
    if raw["safe"]:
        return params, 1.0, raw, raw

    p0 = _scaled_params(params, 0.0)
    s0 = _evaluate_state(obj, p0, n_u, reference_normal, max_cap_fold)
    if not s0["safe"]:
        raise RuntimeError("Classical endpoint is not sampled-safe.")

    lo, hi = 0.0, 1.0
    best_params, best_state = p0, s0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cand = _scaled_params(params, mid)
        state = _evaluate_state(obj, cand, n_u, reference_normal, max_cap_fold)
        if state["safe"]:
            lo = mid
            best_params, best_state = cand, state
        else:
            hi = mid
    return best_params, lo, raw, best_state


def _load_split(data_dir, split, N):
    path = os.path.join(data_dir, f"{split}_N{N}.pkl")
    with open(path, "rb") as f:
        samples = pickle.load(f)
    if not samples:
        raise RuntimeError(f"empty split: {path}")
    return samples, path


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c_bound", type=float, default=1.0)
    ap.add_argument("--no_learn_heights", action="store_true")
    ap.add_argument("--max_cap_fold", type=float, default=1e-3)
    ap.add_argument("--projection_iters", type=int, default=40)
    ap.add_argument("--fp64", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64 if a.fp64 else torch.float32

    samples, data_path = _load_split(a.data, a.split, a.N)
    if a.limit > 0:
        samples = samples[:a.limit]

    print("NSSR-V2 projection evaluation")
    print(f"  data               : {data_path}")
    print(f"  objects            : {len(samples)}")
    print(f"  checkpoint         : {a.ckpt}")
    print(f"  safety samples n_u : {a.n_u}")
    print(f"  c_bound            : {a.c_bound:g}")
    print(f"  cap threshold      : {a.max_cap_fold:g}")

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
                sample, m=a.m, device=device, dtype=dtype,
                gt_subsample=a.gt_sub, seed=a.seed + i,
            )
            feats = contour_features(
                obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"]
            )
            params = net(feats)
            ref = _classical_reference_normal(obj, a.n_u)

            _, alpha, raw, post = _project_to_safe(
                obj, params, a.n_u, ref, a.max_cap_fold, a.projection_iters
            )
            activated = alpha < 1.0 - 1e-9

            row = {
                "index": i,
                "raw_j_valid": int(raw["j_valid"]),
                "raw_cap_safe": int(raw["cap_safe"]),
                "raw_safe": int(raw["safe"]),
                "raw_negative_fraction": raw["negative_fraction"],
                "raw_degenerate_fraction": raw["degenerate_fraction"],
                "raw_cap_fold": raw["cap_fold"],
                "raw_chamfer_l2": raw["chamfer_l2"],
                "raw_hausdorff": raw["hausdorff"],
                "projection_activated": int(activated),
                "alpha": float(alpha),
                "retained_percent": 100.0 * float(alpha),
                "post_j_valid": int(post["j_valid"]),
                "post_cap_safe": int(post["cap_safe"]),
                "post_safe": int(post["safe"]),
                "post_negative_fraction": post["negative_fraction"],
                "post_degenerate_fraction": post["degenerate_fraction"],
                "post_cap_fold": post["cap_fold"],
                "post_chamfer_l2": post["chamfer_l2"],
                "post_hausdorff": post["hausdorff"],
                "delta_chamfer_l2": post["chamfer_l2"] - raw["chamfer_l2"],
            }
            rows.append(row)

            print(
                f"[{i:3d}] raw {'SAFE' if raw['safe'] else 'FAIL':4s} "
                f"J={'OK' if raw['j_valid'] else 'FAIL':4s} "
                f"cap={raw['cap_fold']:.5f} "
                f"-> {'SAFE' if post['safe'] else 'FAIL':4s} "
                f"{'alpha='+format(alpha,'.3f') if activated else 'unchanged'}"
            )

    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    activated_rows = [r for r in rows if r["projection_activated"]]
    alphas = [r["alpha"] for r in activated_rows]

    print("\nsummary")
    print(
        f"raw       J-valid={100*_mean(rows,'raw_j_valid'):.1f}% "
        f"cap-safe={100*_mean(rows,'raw_cap_safe'):.1f}% "
        f"SAFE={100*_mean(rows,'raw_safe'):.1f}%"
    )
    print(
        f"projected J-valid={100*_mean(rows,'post_j_valid'):.1f}% "
        f"cap-safe={100*_mean(rows,'post_cap_safe'):.1f}% "
        f"SAFE={100*_mean(rows,'post_safe'):.1f}%"
    )
    print(
        f"projection activation={100*_mean(rows,'projection_activated'):.1f}% "
        f"({len(activated_rows)}/{len(rows)})"
    )
    if alphas:
        print(
            f"alpha on projected objects: mean={np.mean(alphas):.4f} "
            f"median={np.median(alphas):.4f} min={np.min(alphas):.4f}"
        )
    else:
        print("alpha on projected objects: no projection required")
    print(
        f"Chamfer L2 mean: raw={_mean(rows,'raw_chamfer_l2'):.6f} "
        f"projected={_mean(rows,'post_chamfer_l2'):.6f} "
        f"delta={_mean(rows,'delta_chamfer_l2'):+.6f}"
    )
    print(
        f"worst cap-fold: raw={max(r['raw_cap_fold'] for r in rows):.6f} "
        f"projected={max(r['post_cap_fold'] for r in rows):.6f}"
    )
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
