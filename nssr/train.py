"""NSSR-V2 training loop with safety-aware validation/checkpointing.

Active training safeguards
--------------------------
- normalized local orientation barrier (signed_J / area_scale)
- normalized cap radial/meridional turn-back barrier

Validation safety
-----------------
Every validation pass reports:
- J-valid rate
- cap-safe rate
- overall SAFE rate = J-valid AND cap-safe

Checkpoint policy
-----------------
- ``best_accuracy.pt``: lowest validation Chamfer.
- ``best_safe.pt``: lexicographically best safety tuple
      (SAFE rate, J-valid rate, cap-safe rate, -Chamfer).
- ``best.pt``: alias of ``best_safe.pt`` for downstream compatibility.
- ``last.pt``: final network state.

Epoch 0 is the classical zero-initialized baseline and is never saved as a
learned checkpoint.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch

from .geometry import (
    evaluate_geometry,
    hermite_surface,
    surface_normals,
    surface_points,
    tangent_field,
)
from .losses import (
    cap_radial_fold_max,
    intentional_pole_mask,
    total_loss,
)
from .metrics import c1_diagnostic, evaluate_surface
from .networks import ParamNet, contour_features
from .preprocess import preprocess_object


def to_torch(
    sample,
    m=256,
    device="cpu",
    dtype=torch.float32,
    gt_subsample=20000,
    seed=0,
):
    pre = preprocess_object(
        sample["contours"],
        sample["Z"],
        m=m,
        base_circular=sample.get("base_circular", True),
        crown_circular=sample.get("crown_circular", True),
        closed_top=sample.get("closed_top", True),
    )

    T = lambda x: torch.as_tensor(
        np.asarray(x), device=device, dtype=dtype
    )

    rng = np.random.default_rng(seed)
    q = sample["gt_pts"].shape[0]
    idx = rng.choice(q, size=min(gt_subsample, q), replace=False)

    nrm = pre["norm"]
    gt = (
        sample["gt_pts"][idx]
        - np.array([*nrm["center_xy"], nrm["zmid"]])
    ) / nrm["scale"]

    raw_normals = sample.get("gt_normals")
    gt_normals = None if raw_normals is None else T(raw_normals[idx])

    return {
        "R": T(pre["R"]),
        "Z": T(pre["Z"]),
        "RB": T(pre["RB"]),
        "RC": T(pre["RC"]),
        "Bh": T(pre["Bh"]),
        "Th": T(pre["Th"]),
        "gt_pts": T(gt),
        "gt_normals": gt_normals,
        "base_circular": pre["base_circular"],
        "crown_circular": pre["crown_circular"],
        "closed_top": pre["closed_top"],
    }


def cap_point_mask(S, closed_top=True):
    P, nu, m, _ = S.shape
    mask = torch.zeros(P, nu, m, dtype=torch.bool, device=S.device)
    mask[0] = True
    if closed_top:
        mask[-1] = True
    return mask.reshape(-1)


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


def _classical_reference_normal(obj, n_u):
    """Fixed pointwise normal field used for signed orientation."""
    with torch.no_grad():
        reference = evaluate_geometry(
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
        )
    return reference.surface.normals.detach()


def forward_object(
    net,
    obj,
    n_u=24,
    *,
    lam_jacobian=0.0,
    reference_normal=None,
):
    feats = contour_features(
        obj["R"], obj["Z"], obj["RB"], obj["RC"],
        obj["Bh"], obj["Th"]
    )
    params = net(feats)

    if lam_jacobian != 0.0:
        geometry = evaluate_geometry(
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
            reference_normal=reference_normal,
        )
        S = geometry.surface.xyz
        pts = surface_points(S)
        nrms = geometry.surface.normals.reshape(-1, 3)
    else:
        geometry = None
        S = hermite_surface(
            obj["R"], obj["Z"], obj["RB"], obj["RC"],
            obj["Bh"], obj["Th"],
            params,
            n_u=n_u,
            closed_top=obj.get("closed_top", True),
            base_circular=obj.get("base_circular", True),
            crown_circular=obj.get("crown_circular", True),
        )
        pts = surface_points(S)
        nrms = surface_normals(S).reshape(-1, 3)

    return S, pts, nrms, params, geometry


def _geometry_for_params(obj, params, n_u, reference_normal):
    """Geometry evaluation used by validation safety diagnostics."""
    return evaluate_geometry(
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
        reference_normal=reference_normal,
    )


def _safety_summary(geometry, obj, max_cap_fold):
    jac = geometry.jacobian
    if jac is None:
        raise RuntimeError("validation safety requires Jacobian evaluation")

    evaluable = ~intentional_pole_mask(
        geometry.surface.xyz,
        closed_top=obj.get("closed_top", True),
    )

    neg = jac.flipped_mask & evaluable
    deg = jac.degenerate_mask & evaluable
    denom = max(int(evaluable.sum().item()), 1)

    neg_count = int(neg.sum().item())
    deg_count = int(deg.sum().item())
    j_valid = (neg_count == 0 and deg_count == 0)

    cap_fold = float(
        cap_radial_fold_max(
            geometry.surface.xyz,
            obj["RB"],
            obj["RC"],
            closed_top=obj.get("closed_top", True),
        ).item()
    )
    cap_safe = cap_fold <= max_cap_fold

    return {
        "j_valid": bool(j_valid),
        "cap_safe": bool(cap_safe),
        "safe": bool(j_valid and cap_safe),
        "negative_fraction": neg_count / denom,
        "degenerate_fraction": deg_count / denom,
        "cap_fold": cap_fold,
    }


def _load_state_dict(path, device, dtype):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {k: v.to(dtype=dtype) for k, v in state.items()}


def _save_state(net, path):
    torch.save(net.state_dict(), path)


def train(
    samples,
    val_samples,
    out_dir="runs/exp1",
    epochs=100,
    lr=1e-3,
    m=256,
    n_u=24,
    device=None,
    dtype=torch.float32,
    lam_n=0.1,
    lam_r=1e-3,
    lam_s=1e-3,
    lam_jacobian=0.0,
    jacobian_margin=0.05,
    jacobian_power=2.0,
    lam_cap_fold=0.0,
    cap_fold_margin=1e-3,
    cap_fold_power=2.0,
    geometry_topk_fraction=0.05,
    max_cap_fold=1e-3,
    accum=8,
    seed=0,
    surf_sub=20000,
    gt_sub=20000,
    val_every=5,
    patience=0,
    val_subset=0,
    eval_n_u=None,
    init_ckpt=None,
    cap_weight=1.0,
    learn_heights=True,
    c_bound=1.0,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not (0.0 < geometry_topk_fraction <= 1.0):
        raise ValueError("geometry_topk_fraction must lie in (0, 1]")
    if jacobian_power <= 0:
        raise ValueError("jacobian_power must be > 0")
    if cap_fold_power <= 0:
        raise ValueError("cap_fold_power must be > 0")
    if cap_fold_margin <= 0:
        raise ValueError("cap_fold_margin must be > 0")
    if max_cap_fold <= 0:
        raise ValueError("max_cap_fold must be > 0")

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Preprocessing ...")
    train_objs = [
        to_torch(s, m, device, dtype, seed=i)
        for i, s in enumerate(samples)
    ]
    val_objs = [
        to_torch(s, m, device, dtype, seed=10_000 + i)
        for i, s in enumerate(val_samples)
    ]

    net = ParamNet(
        learn_heights=learn_heights,
        c_bound=c_bound,
    ).to(device=device, dtype=dtype)

    if init_ckpt:
        net.load_state_dict(_load_state_dict(init_ckpt, device, dtype))
        print(f"initialized from {init_ckpt} (fine-tuning)")

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=lr, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )

    # Training references are required only when Jacobian loss is active.
    train_reference_normals = None
    if lam_jacobian != 0.0:
        print("Preparing classical orientation references for Jacobian loss ...")
        train_reference_normals = [
            _classical_reference_normal(obj, n_u)
            for obj in train_objs
        ]

    # Validation safety is always measured, regardless of training loss weights.
    val_nu = eval_n_u or n_u
    print("Preparing validation safety references ...")
    val_reference_normals = [
        _classical_reference_normal(obj, val_nu)
        for obj in val_objs
    ]

    log_path = os.path.join(out_dir, "log.csv")
    logf = open(log_path, "w", newline="")
    logger = csv.writer(logf)
    logger.writerow([
        "epoch",
        "train_loss",
        "train_chamfer",
        "train_normal",
        "train_jacobian",
        "train_cap_fold",
        "val_chamfer_l2",
        "val_hausdorff",
        "c1_min",
        "j_valid_rate",
        "cap_safe_rate",
        "safe_rate",
        "mean_negative_j_fraction",
        "mean_cap_fold",
        "worst_cap_fold",
        "secs",
    ])

    best_accuracy = float("inf")
    best_safe_key = None
    best_safe_epoch = None
    best_accuracy_epoch = None
    bad = 0

    for epoch in range(epochs + 1):
        t0 = time.time()
        net.train()

        totals = {
            "loss": 0.0,
            "chamfer": 0.0,
            "normal": 0.0,
            "jacobian": 0.0,
            "cap_fold": 0.0,
        }
        nb = 0
        permutation = np.random.permutation(len(train_objs))
        optimizer.zero_grad()
        group_start = 0

        for step, k in enumerate(permutation):
            obj = train_objs[k]
            ref = (
                train_reference_normals[k]
                if train_reference_normals is not None
                else None
            )

            S, pts, nrms, params, geometry = forward_object(
                net,
                obj,
                n_u=n_u,
                lam_jacobian=lam_jacobian,
                reference_normal=ref,
            )

            cmask = (
                cap_point_mask(S, obj.get("closed_top", True))
                if cap_weight < 1.0
                else None
            )

            loss, parts = total_loss(
                pts,
                nrms,
                obj["gt_pts"],
                obj["gt_normals"],
                params,
                lam_n=lam_n,
                lam_r=lam_r,
                lam_s=lam_s,
                surf_sub=surf_sub,
                gt_sub=gt_sub,
                cap_mask=cmask,
                cap_weight=cap_weight,
                surface=S,
                RB=obj["RB"],
                RC=obj["RC"],
                closed_top=obj.get("closed_top", True),
                geometry=geometry,
                lam_jacobian=lam_jacobian,
                jacobian_margin=jacobian_margin,
                jacobian_power=jacobian_power,
                lam_cap_fold=lam_cap_fold,
                cap_fold_margin=cap_fold_margin,
                cap_fold_power=cap_fold_power,
                geometry_topk_fraction=geometry_topk_fraction,
            )

            if epoch > 0:
                group_end = min(group_start + accum, len(permutation))
                group_size = group_end - group_start
                (loss / group_size).backward()

                if step + 1 == group_end:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    group_start = group_end

            totals["loss"] += loss.item()
            totals["chamfer"] += parts["chamfer"]
            totals["normal"] += parts["normal"]
            totals["jacobian"] += parts["jacobian"]
            totals["cap_fold"] += parts["cap_fold"]
            nb += 1

        if epoch > 0:
            scheduler.step()

        do_val = epoch % val_every == 0 or epoch == epochs

        if do_val:
            net.eval()
            vobjs = val_objs if val_subset <= 0 else val_objs[:val_subset]

            cds, hs, c1s = [], [], []
            safety = []

            with torch.no_grad():
                for vi, obj in enumerate(vobjs):
                    feats = contour_features(
                        obj["R"], obj["Z"], obj["RB"], obj["RC"],
                        obj["Bh"], obj["Th"]
                    )
                    params = net(feats)

                    geom = _geometry_for_params(
                        obj,
                        params,
                        val_nu,
                        val_reference_normals[vi],
                    )

                    pts = surface_points(geom.surface.xyz)
                    nrms = geom.surface.normals.reshape(-1, 3)

                    metrics = evaluate_surface(
                        pts,
                        obj["gt_pts"],
                        nrms,
                        obj["gt_normals"],
                    )

                    gR, gZ = tangent_field(
                        obj["R"], obj["Z"], obj["RB"], obj["RC"],
                        obj["Bh"], obj["Th"], params,
                        closed_top=obj.get("closed_top", True),
                    )

                    cds.append(metrics["chamfer_l2"])
                    hs.append(metrics["hausdorff"])
                    c1s.append(c1_diagnostic(gR, gZ)["global_min"])
                    safety.append(
                        _safety_summary(geom, obj, max_cap_fold)
                    )

            v_cd = float(np.mean(cds))
            v_h = float(np.mean(hs))
            v_c1 = float(np.min(c1s))

            j_rate = float(np.mean([s["j_valid"] for s in safety]))
            cap_rate = float(np.mean([s["cap_safe"] for s in safety]))
            safe_rate = float(np.mean([s["safe"] for s in safety]))
            mean_neg = float(
                np.mean([s["negative_fraction"] for s in safety])
            )
            mean_cap = float(np.mean([s["cap_fold"] for s in safety]))
            worst_cap = float(np.max([s["cap_fold"] for s in safety]))

            elapsed = time.time() - t0
            row = [
                epoch,
                totals["loss"] / nb,
                totals["chamfer"] / nb,
                totals["normal"] / nb,
                totals["jacobian"] / nb,
                totals["cap_fold"] / nb,
                v_cd,
                v_h,
                v_c1,
                j_rate,
                cap_rate,
                safe_rate,
                mean_neg,
                mean_cap,
                worst_cap,
                elapsed,
            ]
            logger.writerow(row)
            logf.flush()

            tag = "  <-- CLASSICAL BASELINE" if epoch == 0 else ""
            print(
                f"ep {epoch:3d} "
                f"| loss {row[1]:.5f} "
                f"| CD {row[2]:.5f} "
                f"| J {row[4]:.3e} "
                f"| CAP {row[5]:.3e} "
                f"| val CD {v_cd:.6f} "
                f"| SAFE {100*safe_rate:5.1f}% "
                f"| Jv {100*j_rate:5.1f}% "
                f"| CAPv {100*cap_rate:5.1f}% "
                f"| worstCAP {worst_cap:.5f} "
                f"| C1min {v_c1:.3f} "
                f"| {elapsed:.1f}s{tag}"
            )

            if epoch > 0:
                # Accuracy checkpoint.
                if v_cd < best_accuracy:
                    best_accuracy = v_cd
                    best_accuracy_epoch = epoch
                    _save_state(
                        net,
                        os.path.join(out_dir, "best_accuracy.pt"),
                    )

                # Safety-first lexicographic checkpoint:
                # maximize safety, then component validity, then accuracy.
                safe_key = (
                    safe_rate,
                    j_rate,
                    cap_rate,
                    -v_cd,
                )
                if best_safe_key is None or safe_key > best_safe_key:
                    best_safe_key = safe_key
                    best_safe_epoch = epoch
                    _save_state(
                        net,
                        os.path.join(out_dir, "best_safe.pt"),
                    )
                    # Keep downstream scripts simple: best.pt means the
                    # recommended safety-aware checkpoint.
                    _save_state(
                        net,
                        os.path.join(out_dir, "best.pt"),
                    )

                # Early stopping remains tied to accuracy improvement.  Safety
                # checkpoints are saved independently and cannot be lost.
                if v_cd <= best_accuracy + 1e-15:
                    bad = 0
                else:
                    bad += val_every
                    if patience > 0 and bad >= patience:
                        print(
                            "early stop: no validation-CD improvement "
                            f"in {bad} epochs"
                        )
                        break
        else:
            print(
                f"ep {epoch:3d} "
                f"| loss {totals['loss']/nb:.5f} "
                f"| J {totals['jacobian']/nb:.3e} "
                f"| CAP {totals['cap_fold']/nb:.3e} "
                f"| {time.time()-t0:.1f}s (no val)"
            )

    _save_state(net, os.path.join(out_dir, "last.pt"))
    logf.close()

    print("\ncheckpoint summary")
    if best_safe_epoch is not None:
        print(
            f"  best safety-aware : epoch {best_safe_epoch} "
            f"-> {os.path.join(out_dir, 'best.pt')}"
        )
    if best_accuracy_epoch is not None:
        print(
            f"  best accuracy     : epoch {best_accuracy_epoch} "
            f"-> {os.path.join(out_dir, 'best_accuracy.pt')}"
        )
    print(f"  last              : {os.path.join(out_dir, 'last.pt')}")

    return net


__all__ = [
    "to_torch",
    "cap_point_mask",
    "forward_object",
    "train",
]
