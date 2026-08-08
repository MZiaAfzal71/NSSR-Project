"""NSSR-V2 training loop.

This version preserves the existing training behavior by default while adding
optional geometry-aware Jacobian and curvature regularization.

Key compatibility rule
----------------------
When ``lam_jacobian == 0`` and ``lam_curvature == 0``, the training path uses
the legacy-compatible ``hermite_surface`` function exactly as before.

When either geometry loss is enabled, the loop switches to
``evaluate_geometry`` so the same reconstructed surface is accompanied by
sampled derivatives, Jacobian diagnostics, and curvature.

The model architecture, checkpoint format, preprocessing path, optimizer,
scheduler, validation metrics, and epoch-0 classical baseline behavior remain
unchanged.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Optional

import numpy as np
import torch

from .geometry import (
    evaluate_geometry,
    hermite_surface,
    surface_normals,
    surface_points,
    tangent_field,
)
from .losses import total_loss
from .metrics import c1_diagnostic, evaluate_surface
from .networks import ParamNet, contour_features
from .preprocess import preprocess_object


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def to_torch(
    sample,
    m=256,
    device="cpu",
    dtype=torch.float32,
    gt_subsample=20000,
    seed=0,
):
    """Preprocess one object and convert arrays to torch tensors."""
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

    rng = np.random.default_rng(seed)

    q = sample["gt_pts"].shape[0]
    idx = rng.choice(
        q,
        size=min(gt_subsample, q),
        replace=False,
    )

    # Ground truth uses the same normalization as the input contours.
    nrm = pre["norm"]
    gt = (
        sample["gt_pts"][idx]
        - np.array([*nrm["center_xy"], nrm["zmid"]])
    ) / nrm["scale"]

    return {
        "R": T(pre["R"]),
        "Z": T(pre["Z"]),
        "RB": T(pre["RB"]),
        "RC": T(pre["RC"]),
        "Bh": T(pre["Bh"]),
        "Th": T(pre["Th"]),
        "gt_pts": T(gt),
        "gt_normals": T(sample["gt_normals"][idx]),
        "base_circular": pre["base_circular"],
        "crown_circular": pre["crown_circular"],
        "closed_top": pre["closed_top"],
    }


# ---------------------------------------------------------------------------
# Surface helpers
# ---------------------------------------------------------------------------

def cap_point_mask(S):
    """Boolean mask over flattened surface points marking cap patches."""
    P, nu, m, _ = S.shape

    mask = torch.zeros(
        P,
        nu,
        m,
        dtype=torch.bool,
        device=S.device,
    )

    mask[0] = True
    mask[-1] = True

    return mask.reshape(-1)


def _classical_reference_normal(obj, n_u):
    """Build a fixed orientation reference from the classical reconstruction.

    This should be computed without gradients.  The resulting normal field has
    the same sampled patch layout as the learned surface and is therefore a
    meaningful sign reference for the Jacobian barrier.
    """
    N, m = obj["R"].shape[:2]

    params = {
        "s_a": torch.zeros(N, m, device=obj["R"].device, dtype=obj["R"].dtype),
        "s_b": torch.zeros(N, m, device=obj["R"].device, dtype=obj["R"].dtype),
        "s_tau": torch.zeros(N, m, device=obj["R"].device, dtype=obj["R"].dtype),
        "s_fB": torch.zeros(m, device=obj["R"].device, dtype=obj["R"].dtype),
        "s_fC": torch.zeros(m, device=obj["R"].device, dtype=obj["R"].dtype),
        "s_bh": torch.zeros((), device=obj["R"].device, dtype=obj["R"].dtype),
        "s_th": torch.zeros((), device=obj["R"].device, dtype=obj["R"].dtype),
    }

    with torch.no_grad():
        reference = evaluate_geometry(
            obj["R"],
            obj["Z"],
            obj["RB"],
            obj["RC"],
            obj["Bh"],
            obj["Th"],
            params,
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
    lam_curvature=0.0,
    reference_normal=None,
    max_abs_curvature=100.0,
):
    """Forward one object through network and geometry pipeline.

    Returns
    -------
    S, pts, nrms, params, geometry

    ``geometry`` is None on the legacy/default path.
    """
    feats = contour_features(
        obj["R"],
        obj["Z"],
        obj["RB"],
        obj["RC"],
        obj["Bh"],
        obj["Th"],
    )

    params = net(feats)

    use_geometry = (
        lam_jacobian != 0.0
        or lam_curvature != 0.0
    )

    if use_geometry:
        geometry = evaluate_geometry(
            obj["R"],
            obj["Z"],
            obj["RB"],
            obj["RC"],
            obj["Bh"],
            obj["Th"],
            params,
            n_u=n_u,
            closed_top=obj.get("closed_top", True),
            base_circular=obj.get("base_circular", True),
            crown_circular=obj.get("crown_circular", True),
            compute_jacobian=True,
            compute_curvature=True,
            run_validation=False,
            reference_normal=reference_normal,
            max_abs_curvature=max_abs_curvature,
        )

        S = geometry.surface.xyz
        pts = surface_points(S)
        nrms = geometry.surface.normals.reshape(-1, 3)

    else:
        geometry = None

        S = hermite_surface(
            obj["R"],
            obj["Z"],
            obj["RB"],
            obj["RC"],
            obj["Bh"],
            obj["Th"],
            params,
            n_u=n_u,
            closed_top=obj.get("closed_top", True),
            base_circular=obj.get("base_circular", True),
            crown_circular=obj.get("crown_circular", True),
        )

        pts = surface_points(S)
        nrms = surface_normals(S).reshape(-1, 3)

    return S, pts, nrms, params, geometry


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    samples,
    val_samples,
    out_dir="runs/exp1",
    epochs=200,
    lr=1e-3,
    m=256,
    n_u=24,
    device=None,
    dtype=torch.float32,
    lam_n=0.1,
    lam_r=1e-3,
    lam_s=1e-3,
    lam_jacobian=0.0,
    lam_curvature=0.0,
    jacobian_margin=1e-4,
    max_abs_curvature=100.0,
    curvature_power=2.0,
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
    """Train NSSR with optional V2 geometric regularization.

    Geometry losses are disabled by default for exact migration compatibility.

    Recommended initial V2 experiment
    ---------------------------------
    Start with small values, for example:

        lam_jacobian = 1e-3
        lam_curvature = 1e-5

    and compare against the zero-weight baseline before increasing them.
    """
    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Preprocessing ...")

    train_objs = [
        to_torch(
            s,
            m,
            device,
            dtype,
            seed=i,
        )
        for i, s in enumerate(samples)
    ]

    val_objs = [
        to_torch(
            s,
            m,
            device,
            dtype,
            seed=10_000 + i,
        )
        for i, s in enumerate(val_samples)
    ]

    net = ParamNet(
        learn_heights=learn_heights,
        c_bound=c_bound,
    ).to(
        device=device,
        dtype=dtype,
    )

    if init_ckpt:
        state = torch.load(
            init_ckpt,
            map_location=device,
        )

        net.load_state_dict({
            k: v.to(dtype=dtype)
            for k, v in state.items()
        })

        print(f"initialized from {init_ckpt} (fine-tuning)")

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=lr,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    use_geometry_losses = (
        lam_jacobian != 0.0
        or lam_curvature != 0.0
    )

    # Precompute fixed classical normals for signed-Jacobian orientation.
    # This avoids allowing the learned prediction to define its own sign.
    train_reference_normals = None
    val_reference_normals = None

    if use_geometry_losses:
        print(
            "Preparing classical orientation references "
            "for geometry-aware losses ..."
        )

        train_reference_normals = [
            _classical_reference_normal(obj, n_u)
            for obj in train_objs
        ]

        val_nu = eval_n_u or n_u

        val_reference_normals = [
            _classical_reference_normal(obj, val_nu)
            for obj in val_objs
        ]

    log_path = os.path.join(
        out_dir,
        "log.csv",
    )

    logf = open(
        log_path,
        "w",
        newline="",
    )

    logger = csv.writer(logf)

    logger.writerow([
        "epoch",
        "train_loss",
        "train_chamfer",
        "train_normal",
        "train_jacobian",
        "train_curvature",
        "val_chamfer_l2",
        "val_hausdorff",
        "c1_min",
        "secs",
    ])

    best = float("inf")
    bad = 0

    for epoch in range(epochs + 1):
        t0 = time.time()

        net.train()

        total_loss_value = 0.0
        total_chamfer = 0.0
        total_normal = 0.0
        total_jacobian = 0.0
        total_curvature = 0.0
        nb = 0

        permutation = np.random.permutation(
            len(train_objs)
        )

        optimizer.zero_grad()

        for step, k in enumerate(permutation):
            obj = train_objs[k]

            reference_normal = (
                train_reference_normals[k]
                if train_reference_normals is not None
                else None
            )

            S, pts, nrms, params, geometry = forward_object(
                net,
                obj,
                n_u=n_u,
                lam_jacobian=lam_jacobian,
                lam_curvature=lam_curvature,
                reference_normal=reference_normal,
                max_abs_curvature=max_abs_curvature,
            )

            cmask = (
                cap_point_mask(S)
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
                geometry=geometry,
                lam_jacobian=lam_jacobian,
                lam_curvature=lam_curvature,
                jacobian_margin=jacobian_margin,
                max_abs_curvature=max_abs_curvature,
                curvature_power=curvature_power,
            )

            # Epoch zero is evaluation-only: with zero-initialized heads this
            # is the classical baseline.
            if epoch > 0:
                (loss / accum).backward()

                if (step + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(),
                        1.0,
                    )
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss_value += loss.item()
            total_chamfer += parts["chamfer"]
            total_normal += parts["normal"]
            total_jacobian += parts["jacobian"]
            total_curvature += parts["curvature"]
            nb += 1

        # Flush any gradient accumulation remainder.
        if (
            epoch > 0
            and len(permutation) % accum != 0
        ):
            torch.nn.utils.clip_grad_norm_(
                net.parameters(),
                1.0,
            )
            optimizer.step()
            optimizer.zero_grad()

        if epoch > 0:
            scheduler.step()

        do_val = (
            epoch % val_every == 0
            or epoch == epochs
        )

        eval_nu = eval_n_u or n_u

        if do_val:
            net.eval()

            vobjs = (
                val_objs
                if val_subset <= 0
                else val_objs[:val_subset]
            )

            vs = []
            hs = []
            c1s = []

            with torch.no_grad():
                for vi, obj in enumerate(vobjs):
                    reference_normal = (
                        val_reference_normals[vi]
                        if val_reference_normals is not None
                        else None
                    )

                    _, pts, nrms, params, _ = forward_object(
                        net,
                        obj,
                        n_u=eval_nu,
                        lam_jacobian=lam_jacobian,
                        lam_curvature=lam_curvature,
                        reference_normal=reference_normal,
                        max_abs_curvature=max_abs_curvature,
                    )

                    metrics = evaluate_surface(
                        pts,
                        obj["gt_pts"],
                        nrms,
                        obj["gt_normals"],
                    )

                    gR, gZ = tangent_field(
                        obj["R"],
                        obj["Z"],
                        obj["RB"],
                        obj["RC"],
                        obj["Bh"],
                        obj["Th"],
                        params,
                        closed_top=obj.get(
                            "closed_top",
                            True,
                        ),
                    )

                    vs.append(
                        metrics["chamfer_l2"]
                    )
                    hs.append(
                        metrics["hausdorff"]
                    )
                    c1s.append(
                        c1_diagnostic(
                            gR,
                            gZ,
                        )["global_min"]
                    )

            v_cd = float(np.mean(vs))
            v_h = float(np.mean(hs))
            v_c1 = float(np.min(c1s))

            row = [
                epoch,
                total_loss_value / nb,
                total_chamfer / nb,
                total_normal / nb,
                total_jacobian / nb,
                total_curvature / nb,
                v_cd,
                v_h,
                v_c1,
                time.time() - t0,
            ]

            logger.writerow(row)
            logf.flush()

            tag = (
                "  <-- CLASSICAL BASELINE"
                if epoch == 0
                else ""
            )

            print(
                f"ep {epoch:3d} "
                f"| loss {row[1]:.5f} "
                f"| CD {row[2]:.5f} "
                f"| J {row[4]:.3e} "
                f"| K {row[5]:.3e} "
                f"| val CD {v_cd:.6f} "
                f"| val H {v_h:.4f} "
                f"| C1min {v_c1:.3f} "
                f"| {row[9]:.1f}s"
                f"{tag}"
            )

            if epoch > 0 and v_cd < best:
                best = v_cd
                bad = 0

                torch.save(
                    net.state_dict(),
                    os.path.join(
                        out_dir,
                        "best.pt",
                    ),
                )

            elif epoch > 0:
                bad += val_every

                if (
                    patience > 0
                    and bad >= patience
                ):
                    print(
                        "early stop: no val improvement "
                        f"in {bad} epochs "
                        f"(best val CD {best:.6f})"
                    )
                    break

        else:
            print(
                f"ep {epoch:3d} "
                f"| loss {total_loss_value / nb:.5f} "
                f"| J {total_jacobian / nb:.3e} "
                f"| K {total_curvature / nb:.3e} "
                f"| {time.time() - t0:.1f}s "
                "(no val)"
            )

    logf.close()

    return net


__all__ = [
    "to_torch",
    "cap_point_mask",
    "forward_object",
    "train",
]
