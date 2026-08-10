"""Reconstruct one NSSR-V2 object and export safety diagnostics.

Modes
-----
classical
    Classical zero-parameter reconstruction.

net
    Reconstruction from a ParamNet checkpoint.

Safety reporting
----------------
The script reports:
- signed-Jacobian validity against the classical pointwise orientation field;
- accidental degeneracy away from intentional cap poles;
- normalized base/crown radial turn-back;
- overall SAFE status.

Curvature is intentionally excluded from the active inference/validity path.

By default the script reports the RAW reconstruction.  ``--project_safe`` can
optionally scale learned corrections toward the classical solution until both
Jacobian and cap-fold constraints pass.  The raw and projected diagnostics are
both written to the JSON report.

Example:
    python scripts/reconstruct.py \
        --data data/synthetic \
        --split test \
        --N 7 \
        --index 0 \
        --mode net \
        --ckpt runs/N7_safe/best.pt \
        --project_safe \
        --out results/reconstruct_N7_0
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nssr.geometry import evaluate_geometry, surface_points, zero_params
from nssr.losses import (
    cap_radial_fold_max,
    cap_radial_fold_measure,
    intentional_pole_mask,
)
from nssr.metrics import evaluate_surface
from nssr.networks import ParamNet, contour_features
from nssr.preprocess import preprocess_object


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_sample(input_path, data_dir, split, N, index):
    if input_path:
        data = _load_pickle(input_path)
        if isinstance(data, (list, tuple)):
            if not data:
                raise RuntimeError(f"empty input sequence: {input_path}")
            return data[index]
        return data

    if not data_dir:
        raise ValueError("provide --input or --data")

    path = os.path.join(data_dir, f"{split}_N{N}.pkl")
    samples = _load_pickle(path)
    if index < 0 or index >= len(samples):
        raise IndexError(
            f"--index {index} outside [0,{len(samples)-1}] for {path}"
        )
    return samples[index]


def prepare_object(sample, m, device, dtype):
    pre = preprocess_object(
        sample["contours"],
        sample["Z"],
        m=m,
        base_circular=sample.get("base_circular", True),
        crown_circular=sample.get("crown_circular", True),
        closed_top=sample.get("closed_top", True),
    )

    T = lambda x: torch.as_tensor(
        np.asarray(x),
        device=device,
        dtype=dtype,
    )

    return {
        "R": T(pre["R"]),
        "Z": T(pre["Z"]),
        "RB": T(pre["RB"]),
        "RC": T(pre["RC"]),
        "Bh": T(pre["Bh"]),
        "Th": T(pre["Th"]),
        "base_circular": pre["base_circular"],
        "crown_circular": pre["crown_circular"],
        "closed_top": pre["closed_top"],
        "norm": pre["norm"],
    }


def _load_state_dict(path, device, dtype):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {k: v.to(dtype=dtype) for k, v in state.items()}


def load_network(path, device, dtype, learn_heights, c_bound):
    if not path:
        raise ValueError("--ckpt is required for --mode net")

    net = ParamNet(
        learn_heights=learn_heights,
        c_bound=c_bound,
    ).to(device=device, dtype=dtype)
    net.load_state_dict(_load_state_dict(path, device, dtype))
    net.eval()
    return net


def predict_params(
    obj,
    mode,
    checkpoint,
    device,
    dtype,
    learn_heights,
    c_bound,
):
    N, m = obj["R"].shape[:2]
    if mode == "classical":
        return zero_params(
            N,
            m,
            device=device,
            dtype=dtype,
        )

    net = load_network(
        checkpoint,
        device,
        dtype,
        learn_heights,
        c_bound,
    )

    with torch.no_grad():
        feats = contour_features(
            obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"]
        )
        return net(feats)


def classical_reference_normal(obj, n_u):
    N, m = obj["R"].shape[:2]
    p0 = zero_params(
        N,
        m,
        device=obj["R"].device,
        dtype=obj["R"].dtype,
    )

    with torch.no_grad():
        geom = evaluate_geometry(
            obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"],
            p0,
            n_u=n_u,
            closed_top=obj["closed_top"],
            base_circular=obj["base_circular"],
            crown_circular=obj["crown_circular"],
            compute_jacobian=False,
            compute_curvature=False,
            run_validation=False,
        )
    return geom.surface.normals.detach()


def evaluate(obj, params, n_u, reference_normal):
    with torch.no_grad():
        return evaluate_geometry(
            obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"],
            params,
            n_u=n_u,
            closed_top=obj["closed_top"],
            base_circular=obj["base_circular"],
            crown_circular=obj["crown_circular"],
            compute_jacobian=True,
            compute_curvature=False,
            run_validation=False,
            reference_normal=reference_normal,
        )


def safety_summary(geom, obj, max_cap_fold):
    jac = geom.jacobian
    eval_mask = ~intentional_pole_mask(
        geom.surface.xyz,
        closed_top=obj["closed_top"],
    )

    neg = jac.flipped_mask & eval_mask
    deg = jac.degenerate_mask & eval_mask

    denom = max(int(eval_mask.sum().item()), 1)
    neg_count = int(neg.sum().item())
    deg_count = int(deg.sum().item())

    b, c = cap_radial_fold_measure(
        geom.surface.xyz,
        obj["RB"],
        obj["RC"],
        closed_top=obj["closed_top"],
    )
    base_fold = float(b.max().item()) if b.numel() else 0.0
    crown_fold = float(c.max().item()) if c.numel() else 0.0
    cap_fold = float(
        cap_radial_fold_max(
            geom.surface.xyz,
            obj["RB"],
            obj["RC"],
            closed_top=obj["closed_top"],
        ).item()
    )

    j_valid = neg_count == 0 and deg_count == 0
    cap_safe = cap_fold <= max_cap_fold

    signed = jac.signed[eval_mask]
    signed = signed[torch.isfinite(signed)]
    area = jac.area_scale[eval_mask]
    area = area[torch.isfinite(area)]

    return {
        "jacobian_valid": bool(j_valid),
        "negative_jacobian": neg_count,
        "negative_fraction": float(neg_count / denom),
        "degenerate": deg_count,
        "degenerate_fraction": float(deg_count / denom),
        "minimum_signed_jacobian":
            float(signed.min().item()) if signed.numel() else None,
        "minimum_area_scale":
            float(area.min().item()) if area.numel() else None,
        "base_cap_fold_max": base_fold,
        "crown_cap_fold_max": crown_fold,
        "cap_fold_max": cap_fold,
        "cap_safe": bool(cap_safe),
        "safe": bool(j_valid and cap_safe),
    }


def scale_params(params, alpha):
    return {k: v * alpha for k, v in params.items()}


def _project_params_to_sampled_safe(
    obj,
    params,
    n_u,
    reference_normal,
    max_cap_fold,
    iters=20,
):
    """Scale all learned corrections toward classical until safety passes."""
    raw = evaluate(obj, params, n_u, reference_normal)
    raw_safety = safety_summary(raw, obj, max_cap_fold)

    if raw_safety["safe"]:
        return params, raw, raw_safety, 1.0

    # alpha=0 corresponds to classical parameters.
    p0 = scale_params(params, 0.0)
    g0 = evaluate(obj, p0, n_u, reference_normal)
    s0 = safety_summary(g0, obj, max_cap_fold)

    if not s0["safe"]:
        raise RuntimeError(
            "classical endpoint itself is not safe under the requested "
            "threshold, so projection has no guaranteed feasible endpoint"
        )

    lo, hi = 0.0, 1.0
    best_geom, best_safety = g0, s0

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pm = scale_params(params, mid)
        gm = evaluate(obj, pm, n_u, reference_normal)
        sm = safety_summary(gm, obj, max_cap_fold)

        if sm["safe"]:
            lo = mid
            best_geom = gm
            best_safety = sm
        else:
            hi = mid

    return scale_params(params, lo), best_geom, best_safety, lo


def denormalize_points(points, norm):
    offset = np.asarray(
        [norm["center_xy"][0], norm["center_xy"][1], norm["zmid"]],
        dtype=points.dtype,
    )
    return points * float(norm["scale"]) + offset


def add_gt_metrics(report, geom, sample, norm):
    if "gt_pts" not in sample:
        return

    pts = surface_points(geom.surface.xyz)
    nrms = geom.surface.normals.reshape(-1, 3)

    gt_np = np.asarray(sample["gt_pts"])
    offset = np.asarray(
        [norm["center_xy"][0], norm["center_xy"][1], norm["zmid"]],
        dtype=gt_np.dtype,
    )
    gt_norm = (gt_np - offset) / float(norm["scale"])
    gt_pts = torch.as_tensor(
        gt_norm,
        device=pts.device,
        dtype=pts.dtype,
    )

    gt_normals = None
    if sample.get("gt_normals") is not None:
        gt_normals = torch.as_tensor(
            np.asarray(sample["gt_normals"]),
            device=pts.device,
            dtype=pts.dtype,
        )

    metrics = evaluate_surface(
        pts,
        gt_pts,
        nrms,
        gt_normals,
    )
    report["ground_truth_metrics"] = {
        k: float(v) for k, v in metrics.items()
    }


def export_npz(path, geom, obj, params):
    surface = geom.surface.xyz.detach().cpu().numpy()
    normals = geom.surface.normals.detach().cpu().numpy()
    points_normalized = surface.reshape(-1, 3)
    points_world = denormalize_points(
        points_normalized,
        obj["norm"],
    )

    arrays = {
        "surface_normalized": surface,
        "points_normalized": points_normalized,
        "points_world": points_world,
        "normals": normals.reshape(-1, 3),
    }

    for k, v in params.items():
        arrays[f"param_{k}"] = v.detach().cpu().numpy()

    if geom.jacobian is not None:
        arrays.update({
            "jacobian_signed":
                geom.jacobian.signed.detach().cpu().numpy(),
            "jacobian_area_scale":
                geom.jacobian.area_scale.detach().cpu().numpy(),
            "jacobian_flipped":
                geom.jacobian.flipped_mask.detach().cpu().numpy(),
            "jacobian_degenerate":
                geom.jacobian.degenerate_mask.detach().cpu().numpy(),
        })

    np.savez_compressed(path, **arrays)



def _sampled_safety(obj, params, n_u, max_cap_fold):
    """Evaluate final sampled NSSR safety: Jacobian-valid AND cap-safe."""
    ref = evaluate_geometry(
        obj["R"], obj["Z"], obj["RB"], obj["RC"],
        obj["Bh"], obj["Th"],
        _zero_params_like(obj),
        n_u=n_u,
        closed_top=obj.get("closed_top", True),
        base_circular=obj.get("base_circular", True),
        crown_circular=obj.get("crown_circular", True),
        compute_jacobian=False,
        compute_curvature=False,
        run_validation=False,
    ).surface.normals.detach()

    geom = evaluate_geometry(
        obj["R"], obj["Z"], obj["RB"], obj["RC"],
        obj["Bh"], obj["Th"],
        params,
        n_u=n_u,
        closed_top=obj.get("closed_top", True),
        base_circular=obj.get("base_circular", True),
        crown_circular=obj.get("crown_circular", True),
        compute_jacobian=True,
        compute_curvature=False,
        run_validation=False,
        reference_normal=ref,
    )

    evaluable = ~intentional_pole_mask(
        geom.surface.xyz,
        closed_top=obj.get("closed_top", True),
    )
    jac = geom.jacobian
    neg = jac.flipped_mask & evaluable
    deg = jac.degenerate_mask & evaluable

    cap = float(
        cap_radial_fold_max(
            geom.surface.xyz,
            obj["RB"],
            obj["RC"],
            closed_top=obj.get("closed_top", True),
        ).item()
    )

    j_valid = bool(
        (neg.sum() == 0).item()
        and (deg.sum() == 0).item()
    )
    cap_safe = cap <= max_cap_fold

    return {
        "geometry": geom,
        "j_valid": j_valid,
        "cap_safe": cap_safe,
        "safe": bool(j_valid and cap_safe),
        "negative_count": int(neg.sum().item()),
        "degenerate_count": int(deg.sum().item()),
        "cap_fold": cap,
    }


def _scale_all_params(params, alpha):
    out = {}
    for k, v in params.items():
        out[k] = alpha * v
    return out


def _project_params_to_sampled_safe(
    obj,
    params,
    n_u,
    max_cap_fold=1e-3,
    max_iter=40,
):
    """Scale all learned corrections toward classical until sampled SAFE."""
    initial = _sampled_safety(obj, params, n_u, max_cap_fold)
    if initial["safe"]:
        return params, 1.0, initial

    classical = _scale_all_params(params, 0.0)
    cstate = _sampled_safety(obj, classical, n_u, max_cap_fold)
    if not cstate["safe"]:
        raise RuntimeError(
            "Classical endpoint is not safe under the requested sampled "
            "Jacobian/cap constraints."
        )

    lo, hi = 0.0, 1.0
    best = classical
    best_state = cstate

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cand = _scale_all_params(params, mid)
        state = _sampled_safety(obj, cand, n_u, max_cap_fold)
        if state["safe"]:
            lo = mid
            best = cand
            best_state = state
        else:
            hi = mid

    return best, lo, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--N", type=int, default=7)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument(
        "--mode",
        choices=["classical", "net"],
        default="net",
    )
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n_u", type=int, default=32)
    ap.add_argument("--c_bound", type=float, default=1.0)
    ap.add_argument("--no_learn_heights", action="store_true")
    ap.add_argument("--max_cap_fold", type=float, default=1e-3)
    ap.add_argument(
        "--project_safe",
        action="store_true",
        help="scale learned corrections toward classical if raw output fails",
    )
    ap.add_argument("--projection_iters", type=int, default=20)
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument(
        "--out",
        default="results/reconstruction",
        help="output prefix or directory",
    )
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64 if a.fp64 else torch.float32

    sample = load_sample(
        a.input,
        a.data,
        a.split,
        a.N,
        a.index,
    )
    obj = prepare_object(
        sample,
        a.m,
        device,
        dtype,
    )

    params = predict_params(
        obj,
        a.mode,
        a.ckpt,
        device,
        dtype,
        not a.no_learn_heights,
        a.c_bound,
    )

    reference_normal = classical_reference_normal(
        obj,
        a.n_u,
    )

    raw_geom = evaluate(
        obj,
        params,
        a.n_u,
        reference_normal,
    )
    raw_safety = safety_summary(
        raw_geom,
        obj,
        a.max_cap_fold,
    )

    final_params = params
    final_geom = raw_geom
    final_safety = raw_safety
    alpha = 1.0

    if a.project_safe and a.mode != "classical" and not raw_safety["safe"]:
        final_params, final_geom, final_safety, alpha = _project_params_to_sampled_safe(
            obj,
            params,
            a.n_u,
            reference_normal,
            a.max_cap_fold,
            iters=a.projection_iters,
        )

    report = {
        "mode": a.mode,
        "checkpoint": a.ckpt or None,
        "N": int(obj["R"].shape[0]),
        "m": int(obj["R"].shape[1]),
        "n_u": a.n_u,
        "max_cap_fold": a.max_cap_fold,
        "raw_safety": raw_safety,
        "projection": {
            "enabled": bool(a.project_safe),
            "applied": bool(alpha < 1.0),
            "alpha": float(alpha),
            "retained_correction_percent": float(100.0 * alpha),
        },
        "final_safety": final_safety,
    }

    add_gt_metrics(
        report,
        final_geom,
        sample,
        obj["norm"],
    )

    out_path = Path(a.out)
    if out_path.suffix:
        prefix = out_path.with_suffix("")
        prefix.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        tag = f"{a.split}_N{a.N}_{a.index}_{a.mode}"
        prefix = out_path / tag

    json_path = str(prefix) + ".json"
    npz_path = str(prefix) + ".npz"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    export_npz(
        npz_path,
        final_geom,
        obj,
        final_params,
    )

    print("raw safety:")
    print(
        f"  J-valid       : {raw_safety['jacobian_valid']}\n"
        f"  negative-J    : {100*raw_safety['negative_fraction']:.4f}%\n"
        f"  degenerate    : {100*raw_safety['degenerate_fraction']:.4f}%\n"
        f"  base cap fold : {raw_safety['base_cap_fold_max']:.6f}\n"
        f"  crown cap fold: {raw_safety['crown_cap_fold_max']:.6f}\n"
        f"  SAFE          : {raw_safety['safe']}"
    )

    if alpha < 1.0:
        print(
            f"safety projection: alpha={alpha:.6f} "
            f"({100*alpha:.1f}% correction retained)"
        )
        print(
            f"  final cap fold: {final_safety['cap_fold_max']:.6f}\n"
            f"  final SAFE    : {final_safety['safe']}"
        )

    print("wrote", json_path)
    print("wrote", npz_path)


if __name__ == "__main__":
    main()
