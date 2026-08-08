"""NSSR-V2 reconstruction / inference entry point.

This script reconstructs one object from contours using either:
- --mode classical: zero NSSR parameters;
- --mode net: a trained ParamNet checkpoint.

It runs the V2 geometry pipeline, including Jacobian, curvature and validation,
and exports normalized/world points plus diagnostics.

Repair is intentionally not exposed here yet: repair.py operates on explicit
resolved tangents, while evaluate_geometry() still builds tangents internally
from network parameters. Add a tangent override hook to geometry.py first.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nssr.geometry import evaluate_geometry, surface_points, zero_params
from nssr.metrics import evaluate_surface
from nssr.networks import ParamNet, contour_features
from nssr.preprocess import preprocess_object


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_sample(*, input_path, data_dir, split, n_slices, index):
    if input_path:
        obj = _load_pickle(input_path)
        if isinstance(obj, (list, tuple)):
            if not obj:
                raise ValueError(f"input file contains an empty sequence: {input_path}")
            if index < 0 or index >= len(obj):
                raise IndexError(f"--index {index} out of range for {len(obj)} samples")
            return obj[index]
        return obj

    if not data_dir:
        raise ValueError("provide either --input or --data")
    path = os.path.join(data_dir, f"{split}_N{n_slices}.pkl")
    samples = _load_pickle(path)
    if index < 0 or index >= len(samples):
        raise IndexError(f"--index {index} out of range for {len(samples)} samples in {path}")
    return samples[index]


def prepare_object(sample, *, m, device, dtype):
    pre = preprocess_object(
        sample["contours"],
        sample["Z"],
        m=m,
        base_circular=sample.get("base_circular", True),
        crown_circular=sample.get("crown_circular", True),
        closed_top=sample.get("closed_top", True),
    )

    def T(x):
        return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)

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


def denormalize_points(points: np.ndarray, norm: dict) -> np.ndarray:
    offset = np.asarray(
        [norm["center_xy"][0], norm["center_xy"][1], norm["zmid"]],
        dtype=points.dtype,
    )
    return points * float(norm["scale"]) + offset


def load_network(checkpoint, *, device, dtype, learn_heights, c_bound):
    if not checkpoint:
        raise ValueError("--ckpt is required in --mode net")
    net = ParamNet(learn_heights=learn_heights, c_bound=c_bound).to(
        device=device, dtype=dtype
    )
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict({k: v.to(dtype=dtype) for k, v in state.items()})
    net.eval()
    return net


def infer_params(obj, *, mode, checkpoint, device, dtype, learn_heights, c_bound):
    N, m = obj["R"].shape[:2]
    if mode == "classical":
        return zero_params(N, m, device=device, dtype=dtype)

    net = load_network(
        checkpoint or "",
        device=device,
        dtype=dtype,
        learn_heights=learn_heights,
        c_bound=c_bound,
    )
    with torch.no_grad():
        feats = contour_features(
            obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"]
        )
        return net(feats)


def classical_reference_normals(obj, *, n_u):
    N, m = obj["R"].shape[:2]
    params = zero_params(N, m, device=obj["R"].device, dtype=obj["R"].dtype)
    with torch.no_grad():
        ref = evaluate_geometry(
            obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"],
            params,
            n_u=n_u,
            closed_top=obj["closed_top"],
            base_circular=obj["base_circular"],
            crown_circular=obj["crown_circular"],
            compute_jacobian=False,
            compute_curvature=False,
            run_validation=False,
        )
    return ref.surface.normals.detach()


def reconstruct(obj, params, *, n_u, max_abs_curvature, max_tangent_magnitude):
    reference = classical_reference_normals(obj, n_u=n_u)
    with torch.no_grad():
        return evaluate_geometry(
            obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"],
            params,
            n_u=n_u,
            closed_top=obj["closed_top"],
            base_circular=obj["base_circular"],
            crown_circular=obj["crown_circular"],
            compute_jacobian=True,
            compute_curvature=True,
            run_validation=True,
            reference_normal=reference,
            max_abs_curvature=max_abs_curvature,
            max_tangent_magnitude=max_tangent_magnitude,
        )


def _tensor_float(x):
    return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)


def build_report(geometry, *, mode, checkpoint):
    report = {
        "mode": mode,
        "checkpoint": checkpoint or None,
        "surface_shape": list(geometry.surface.xyz.shape),
    }

    if geometry.validation is not None:
        v = geometry.validation
        report["validation"] = {
            "valid": bool(v.valid),
            "fold_count": int(v.fold_count),
            "negative_jacobian": int(v.negative_jacobian),
            "degenerate": int(v.degenerate),
            "curvature_violations": int(v.curvature_violations),
            "tangent_violations": int(v.tangent_violations),
        }

    if geometry.jacobian is not None:
        j = geometry.jacobian
        report["jacobian"] = {
            "negative_fraction": _tensor_float(j.negative_fraction),
            "degenerate_fraction": _tensor_float(j.degenerate_fraction),
            "minimum_signed": _tensor_float(j.minimum_signed),
        }

    if geometry.curvature is not None:
        k1 = geometry.curvature.principal_1
        k2 = geometry.curvature.principal_2
        mag = torch.maximum(torch.abs(k1), torch.abs(k2))
        report["curvature"] = {
            "mean_abs_principal": float(mag.mean().cpu().item()),
            "max_abs_principal": float(mag.amax().cpu().item()),
            "mean_curvature_mean": float(geometry.curvature.mean.mean().cpu().item()),
            "gaussian_curvature_mean": float(
                geometry.curvature.gaussian.mean().cpu().item()
            ),
        }

    if geometry.tangents.magnitude is not None:
        tm = geometry.tangents.magnitude
        report["tangents"] = {
            "mean_magnitude": float(tm.mean().cpu().item()),
            "max_magnitude": float(tm.amax().cpu().item()),
        }

    return report


def add_gt_metrics(report, *, geometry, sample, norm):
    if "gt_pts" not in sample:
        return

    pts = surface_points(geometry.surface.xyz).detach()
    nrms = geometry.surface.normals.reshape(-1, 3).detach()

    gt_np = np.asarray(sample["gt_pts"])
    offset = np.asarray(
        [norm["center_xy"][0], norm["center_xy"][1], norm["zmid"]],
        dtype=gt_np.dtype,
    )
    gt_normalized = (gt_np - offset) / float(norm["scale"])
    gt_pts = torch.as_tensor(gt_normalized, device=pts.device, dtype=pts.dtype)

    gt_normals = None
    if sample.get("gt_normals") is not None:
        gt_normals = torch.as_tensor(
            np.asarray(sample["gt_normals"]), device=pts.device, dtype=pts.dtype
        )

    metrics = evaluate_surface(pts, gt_pts, nrms, gt_normals)
    report["ground_truth_metrics"] = {k: float(v) for k, v in metrics.items()}


def export_npz(path, *, geometry, obj, params):
    S = geometry.surface.xyz.detach().cpu().numpy()
    normals = geometry.surface.normals.detach().cpu().numpy()
    points_norm = S.reshape(-1, 3)
    normals_flat = normals.reshape(-1, 3)
    points_world = denormalize_points(points_norm, obj["norm"])

    arrays = {
        "surface_normalized": S,
        "points_normalized": points_norm,
        "points_world": points_world,
        "normals": normals_flat,
        "tangent_radial": geometry.tangents.radial.detach().cpu().numpy(),
        "tangent_axial": geometry.tangents.axial.detach().cpu().numpy(),
    }

    if geometry.jacobian is not None:
        arrays.update({
            "jacobian_signed": geometry.jacobian.signed.detach().cpu().numpy(),
            "jacobian_area": geometry.jacobian.area_scale.detach().cpu().numpy(),
            "jacobian_valid": geometry.jacobian.valid_mask.detach().cpu().numpy(),
            "jacobian_flipped": geometry.jacobian.flipped_mask.detach().cpu().numpy(),
        })

    if geometry.curvature is not None:
        arrays.update({
            "curvature_mean": geometry.curvature.mean.detach().cpu().numpy(),
            "curvature_gaussian": geometry.curvature.gaussian.detach().cpu().numpy(),
            "curvature_k1": geometry.curvature.principal_1.detach().cpu().numpy(),
            "curvature_k2": geometry.curvature.principal_2.detach().cpu().numpy(),
        })

    for key, value in params.items():
        if torch.is_tensor(value):
            arrays[f"param_{key}"] = value.detach().cpu().numpy()

    np.savez_compressed(path, **arrays)


def export_xyz(path, points_world):
    np.savetxt(path, points_world, fmt="%.9g")


def build_parser():
    ap = argparse.ArgumentParser(description="Reconstruct one object with NSSR-V2.")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Pickle containing one sample or sample sequence.")
    source.add_argument("--data", help="Dataset directory containing <split>_N*.pkl.")

    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--N", type=int, default=7)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--mode", choices=("classical", "net"), default="net")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="reconstruction")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n_u", type=int, default=32)
    ap.add_argument("--c_bound", type=float, default=1.0)
    ap.add_argument("--no_learn_heights", action="store_true")
    ap.add_argument("--max_abs_curvature", type=float, default=100.0)
    ap.add_argument(
        "--max_tangent_magnitude", type=float, default=0.0,
        help="0 disables tangent-magnitude validation.",
    )
    ap.add_argument("--device", default="")
    ap.add_argument("--fp64", action="store_true")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.mode == "net" and not args.ckpt:
        ap.error("--ckpt is required when --mode net")
    if args.m < 3:
        ap.error("--m must be >= 3")
    if args.n_u < 2:
        ap.error("--n_u must be >= 2")
    if args.max_abs_curvature <= 0:
        ap.error("--max_abs_curvature must be > 0")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.fp64 else torch.float32

    sample = load_sample(
        input_path=args.input,
        data_dir=args.data,
        split=args.split,
        n_slices=args.N,
        index=args.index,
    )
    obj = prepare_object(sample, m=args.m, device=device, dtype=dtype)
    params = infer_params(
        obj,
        mode=args.mode,
        checkpoint=(args.ckpt or None),
        device=device,
        dtype=dtype,
        learn_heights=not args.no_learn_heights,
        c_bound=args.c_bound,
    )
    geometry = reconstruct(
        obj,
        params,
        n_u=args.n_u,
        max_abs_curvature=args.max_abs_curvature,
        max_tangent_magnitude=(
            args.max_tangent_magnitude if args.max_tangent_magnitude > 0 else None
        ),
    )

    report = build_report(
        geometry, mode=args.mode, checkpoint=(args.ckpt or None)
    )
    add_gt_metrics(report, geometry=geometry, sample=sample, norm=obj["norm"])

    out = Path(args.out)
    if out.suffix:
        out.parent.mkdir(parents=True, exist_ok=True)
        prefix = out.with_suffix("")
    else:
        out.mkdir(parents=True, exist_ok=True)
        prefix = out / "reconstruction"

    npz_path = str(prefix) + ".npz"
    xyz_path = str(prefix) + ".xyz"
    json_path = str(prefix) + ".json"

    export_npz(npz_path, geometry=geometry, obj=obj, params=params)
    points_world = denormalize_points(
        geometry.surface.xyz.detach().cpu().numpy().reshape(-1, 3), obj["norm"]
    )
    export_xyz(xyz_path, points_world)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print("NSSR-V2 reconstruction complete")
    print(f"  mode        : {args.mode}")
    print(f"  valid       : {report.get('validation', {}).get('valid', 'n/a')}")
    print(f"  NPZ         : {npz_path}")
    print(f"  XYZ         : {xyz_path}")
    print(f"  diagnostics : {json_path}")

    if "jacobian" in report:
        print(f"  negative J  : {100.0 * report['jacobian']['negative_fraction']:.3f}%")
    if "curvature" in report:
        print(f"  max |k|     : {report['curvature']['max_abs_principal']:.6g}")


if __name__ == "__main__":
    main()
