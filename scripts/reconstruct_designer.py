"""NSSR-V2 designer-shape reconstruction compatibility script.

This is a V2-compatible port of the original designer workflow.
It intentionally preserves:
- preprocess_designer() reference/parity path,
- classical / net / TTO modes,
- freeze-cap behavior,
- parameter-space safe-render axis-clearance projection.

The newer core validation/repair stack is complementary to this script;
the old parameter-space safety projection is retained because it protects
axis clearance specifically.
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import matplotlib
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nssr.preprocess import preprocess_designer
from nssr.geometry import (
    hermite_surface,
    zero_params,
    surface_points,
    evaluate_geometry,
)
from nssr.networks import ParamNet, contour_features
from nssr.losses import (
    _nn_sqdist,
    cap_radial_fold_loss,
    cap_radial_fold_max,
    intentional_pole_mask,
)
from nssr.metrics import axis_clearance
from nssr.safety import (
    classical_geometry_and_reference,
    geometry_from_params,
    geometry_safety_summary,
    project_staged_to_safe,
    scaled_params,
)



def change_to_npy(file_path):
    # Path.with_suffix replaces the old extension with the new one
    return str(Path(file_path).with_suffix('.npy'))


def freeze_cap_params(params):
    """Hold ALL cap-controlling parameters at their classical values.

    Must cover BOTH families of cap parameter:
      s_fB / s_fC -- radial bulge of the cap surface
      s_bh / s_th -- the cap's AXIAL EXTENT (learned Bh/Th multipliers)

    An earlier version zeroed only s_fB/s_fC, which is why --freeze_caps
    appeared to do nothing: measured on the banana base cap, s_fB has NO
    effect on z-extent (0.0681 at every value), while s_bh spans
    0.0338-0.1371. The flatness the user was trying to suppress lives
    entirely in s_bh/s_th.
    """
    out = dict(params)
    for k in ("s_fB", "s_fC", "s_bh", "s_th"):
        if k in out:
            out[k] = torch.zeros_like(out[k])
    return out



def _scaled_params(params, alpha, keys=None):
    if keys is None:
        keys = tuple(params.keys())
    return {
        k: (v * alpha if k in keys else v)
        for k, v in params.items()
    }


def cap_fold_ratio(obj, S):
    """Maximum dimensionless radial/meridional turn-back in the caps."""
    return float(
        cap_radial_fold_max(
            S,
            obj["RB"],
            obj["RC"],
            closed_top=obj["closed_top"],
        ).detach()
    )


def _classical_reference_normal(obj, n_u):
    """Fixed classical pointwise normal field for signed-Jacobian checks."""
    N, m = obj["R"].shape[:2]
    params0 = zero_params(
        N,
        m,
        device=obj["R"].device,
        dtype=obj["R"].dtype,
    )
    with torch.no_grad():
        geom0 = evaluate_geometry(
            obj["R"], obj["Z"], obj["RB"], obj["RC"],
            obj["Bh"], obj["Th"],
            params0,
            n_u=n_u,
            closed_top=obj["closed_top"],
            base_circular=obj["base_circular"],
            crown_circular=obj["crown_circular"],
            compute_jacobian=False,
            compute_curvature=False,
            run_validation=False,
        )
    return geom0.surface.normals.detach()


def _jacobian_state(obj, params, n_u, reference_normal=None):
    """Sampled Jacobian validity outside the intentional exact cap poles."""
    if reference_normal is None:
        reference_normal = _classical_reference_normal(obj, n_u)

    geom = evaluate_geometry(
        obj["R"], obj["Z"], obj["RB"], obj["RC"],
        obj["Bh"], obj["Th"],
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

    jac = geom.jacobian
    evaluable = ~intentional_pole_mask(
        geom.surface.xyz,
        closed_top=obj["closed_top"],
    )
    neg = jac.flipped_mask & evaluable
    deg = jac.degenerate_mask & evaluable

    neg_count = int(neg.sum().item())
    deg_count = int(deg.sum().item())
    return {
        "valid": neg_count == 0 and deg_count == 0,
        "negative": neg_count,
        "degenerate": deg_count,
    }


def _safe_state(
    obj,
    params,
    n_u,
    surf_fn,
    min_ratio,
    max_cap_fold,
    reference_normal=None,
):
    """Combined designer safety state.

    Safety requires:
      1) requested axis clearance,
      2) sampled Jacobian validity outside intentional poles,
      3) cap turn-back <= max_cap_fold.
    """
    S = surf_fn(obj, params, n_u)
    ratio = float(axis_clearance(S, obj["R"])[1])
    fold = cap_fold_ratio(obj, S)
    jstate = _jacobian_state(
        obj,
        params,
        n_u,
        reference_normal=reference_normal,
    )
    ok = (
        ratio >= min_ratio
        and jstate["valid"]
        and fold <= max_cap_fold
    )
    return {
        "surface": S,
        "ratio": ratio,
        "fold": fold,
        "j_valid": jstate["valid"],
        "negative": jstate["negative"],
        "degenerate": jstate["degenerate"],
        "safe": ok,
    }


def project_to_safe(
    obj,
    params,
    n_u,
    surf_fn,
    min_ratio=0.15,
    max_cap_fold=1e-3,
    iters=18,
):
    """Designer projection using the shared NSSR J/cap projector.

    The shared failure-aware policy is authoritative for Jacobian/cap safety:
      - cap-only failure -> global scaling;
      - Jacobian failure -> tangent-only, then global fallback.

    Designer rendering adds one extra constraint: axis clearance.  If the
    shared J/cap result still violates ``min_ratio``, a final global line
    search is performed while rechecking BOTH axis clearance and the same
    shared J/cap safety definition.

    The original seven-value public return contract is preserved:
        params, alpha, r0, r1, f0, f1, stage
    """
    _, reference_normal = classical_geometry_and_reference(obj, n_u)

    raw_geom = geometry_from_params(
        obj, params, n_u, reference_normal
    )
    raw_safety = geometry_safety_summary(
        raw_geom, obj, max_cap_fold
    )
    raw_surface = surf_fn(obj, params, n_u)
    r0 = float(axis_clearance(raw_surface, obj["R"])[1])
    f0 = float(raw_safety["cap_fold_max"])

    # First enforce the canonical shared Jacobian/cap safety policy.
    (
        projected,
        alpha,
        stage,
        _,
        _,
        post_safety,
        _,
    ) = project_staged_to_safe(
        obj,
        params,
        n_u,
        reference_normal,
        max_cap_fold=max_cap_fold,
        max_iter=iters,
        with_metrics=False,
    )

    post_surface = surf_fn(obj, projected, n_u)
    r1 = float(axis_clearance(post_surface, obj["R"])[1])
    f1 = float(post_safety["cap_fold_max"])

    if r1 >= min_ratio:
        return projected, alpha, r0, r1, f0, f1, stage

    # Designer-only fallback: scale all original corrections toward the
    # classical endpoint until axis clearance AND shared J/cap safety pass.
    def designer_state(p):
        geom = geometry_from_params(obj, p, n_u, reference_normal)
        safety = geometry_safety_summary(geom, obj, max_cap_fold)
        S = surf_fn(obj, p, n_u)
        ratio = float(axis_clearance(S, obj["R"])[1])
        return safety, ratio

    classical = scaled_params(params, 0.0)
    c_safety, c_ratio = designer_state(classical)
    if not c_safety["safe"] or c_ratio < min_ratio:
        raise RuntimeError(
            "Classical endpoint is not safe under the requested "
            "axis/Jacobian/cap constraints."
        )

    lo, hi = 0.0, 1.0
    best = classical
    best_safety = c_safety
    best_ratio = c_ratio

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cand = scaled_params(params, mid)
        cand_safety, cand_ratio = designer_state(cand)
        if cand_safety["safe"] and cand_ratio >= min_ratio:
            lo = mid
            best = cand
            best_safety = cand_safety
            best_ratio = cand_ratio
        else:
            hi = mid

    return (
        best,
        lo,
        r0,
        best_ratio,
        f0,
        float(best_safety["cap_fold_max"]),
        "axis_all",
    )


def load_designer(ds, n1, device, dtype):
    pre = preprocess_designer(ds, n1=n1)
    T = lambda x: torch.as_tensor(np.asarray(x), device=device, dtype=dtype)
    obj = {k: T(pre[k]) for k in ("R", "Z", "RB", "RC", "Bh", "Th")}
    obj.update(base_circular=pre["base_circular"],
               crown_circular=pre["crown_circular"],
               closed_top=pre["closed_top"])
    return obj, pre


def surf(obj, params, n_u):
    return hermite_surface(obj["R"], obj["Z"], obj["RB"], obj["RC"],
                           obj["Bh"], obj["Th"], params, n_u=n_u,
                           closed_top=obj["closed_top"],
                           base_circular=obj["base_circular"],
                           crown_circular=obj["crown_circular"])


def render(S, path, title, hide_crown=False):
    """hide_crown drops the LAST patch from the plot only (matching the
    paper's Figure 1c convention for the vase); the underlying surface S
    always includes the full, accurately-computed crown patch -- nothing
    is removed from the reconstruction itself, only from this picture."""
    S = S.detach().cpu().numpy()
    np.save(change_to_npy(path), S)
    n_patches = S.shape[0] - 1 if hide_crown else S.shape[0]
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    for p in range(n_patches):
        ax.plot_wireframe(S[p, :, :, 0], S[p, :, :, 1], S[p, :, :, 2],
                          rcount=10, ccount=24, linewidth=0.5, color="k")
    ax.axis("equal"); ax.axis("off"); ax.view_init(elev=12, azim=15)
    fig.suptitle(title, fontsize=11, y=0.04)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", path)


def tto_leave_one_out(obj, n_u, iters=300, lr=1e-3, reg=0.0, lam_prox=1e-2,
                      lam_cap_fold=1e-2, init_net=None, c_bound=1.0,
                      device="cpu", dtype=torch.float64):
    """Test-time optimization, leave-one-slice-out, with NO ground truth:
    hold out each interior contour in turn, reconstruct from the rest, and
    require the surface to pass through the held-out ring.

    What is optimized: the WEIGHTS OF A ParamNet, i.e. a function mapping
    local geometry -> tangent corrections. NOT free per-row numbers.

    Why this matters (this was a real design flaw in an earlier version):
    with per-row parameters (s_a[i] etc.), every fold removes a row, so the
    surviving rows sit across LARGER vertical gaps than in the real object.
    The optimizer therefore tuned each row's tangent scaling to be correct
    for those widened gaps -- and the final render applies those same
    numbers to the FULL object, where the gaps are back to normal, so the
    tangents systematically overshoot. On the banana (7 rows, smooth,
    monotonic Z) the mismatch is mild and the result looked fine; on the
    apple and vase (non-monotonic Z, tightly-spaced rows, so removing a row
    changes local geometry a lot) it was severe -- exploded caps and
    displaced rings. A geometry-CONDITIONED function does not have this
    problem: it is evaluated on whatever configuration it is given, so the
    subset folds and the final full-object evaluation stay coherent. This
    is also exactly why '--mode net' transfers cleanly.

    Loss note: ONE-SIDED distance (held-out ring -> reconstructed surface).
    A two-sided Chamfer here also penalizes the whole surface for being far
    from the one small target ring, which collapses both caps inward.

    init_net: optional state_dict. Starting from the trained checkpoint
    makes this a per-object FINE-TUNE (usually best); starting from scratch
    (zero-initialized head => exactly classical) makes it fully
    training-data-free, which is the fair comparison against per-shape
    methods like OReX.
    """
    N, m = obj["R"].shape[0], obj["R"].shape[1]
    net = ParamNet(c_bound=c_bound).to(device=device, dtype=dtype)
    if init_net is not None:
        net.load_state_dict(init_net)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=reg)
    Z3 = lambda i: torch.cat([obj["R"][i],
                              obj["Z"][i].expand(m, 1)], dim=1)

    # Which folds are usable? Removing row i leaves rows i-1 and i+1
    # adjacent; if they sit at (nearly) the same height the resulting gap
    # patch has ~zero vertical extent and cannot pass through the held-out
    # ring at all -- an unsatisfiable fold that only injects large
    # gradients. The vase has exactly this case (rows 0 and 2 are both at
    # the same height, straddling its flat base ring), so skip such folds.
    Zc = obj["Z"].detach().cpu().numpy()
    span = float(abs(Zc.max() - Zc.min())) or 1.0
    interior = [i for i in range(1, N - 1)
                if abs(float(Zc[i + 1] - Zc[i - 1])) > 1e-3 * span]
    skipped = [i for i in range(1, N - 1) if i not in interior]
    if skipped:
        print(f"  tto: skipping degenerate fold(s) {skipped} "
              f"(rows either side sit at the same height)")
    if not interior:
        raise RuntimeError("no usable leave-one-out folds for this object")

    # ---- reference ("do not drift from here") parameters -----------------
    # Computed ONCE from the initial network, before any optimization.
    #   init_net=None  -> zero-initialized net -> reference == classical
    #   init_net=ckpt  -> reference == the trained model's own predictions
    # Everything below anchors to THIS reference rather than to zero. An
    # earlier version hard-zeroed the cap scalings and pulled all fields
    # toward zero; that is self-consistent when starting from classical
    # (the zero-init net already outputs ~0), which is why --tto_init
    # classical looked fine -- but it BROKE --tto_init net, because the
    # trained checkpoint predicts non-zero cap scalings and boundary-row
    # tangents that were co-adapted with them. Zeroing the caps while
    # keeping those tangents is an inconsistent pairing, and it reintroduced
    # the cap loops. It also meant the anchor fought the very checkpoint the
    # user asked to start from.
    net.eval()
    ref = {}
    with torch.no_grad():
        for i in interior:
            keep = [k for k in range(N) if k != i]
            f = contour_features(obj["R"][keep], obj["Z"][keep], obj["RB"],
                                 obj["RC"], obj["Bh"], obj["Th"])
            ref[i] = {k: v.detach().clone() for k, v in net(f).items()}
        f_full = contour_features(obj["R"], obj["Z"], obj["RB"], obj["RC"],
                                  obj["Bh"], obj["Th"])
        ref_full = {k: v.detach().clone() for k, v in net(f_full).items()}
    net.train()

    for it in range(iters):
        opt.zero_grad()
        loss = 0.0
        prox = 0.0
        for i in interior:
            keep = [k for k in range(N) if k != i]
            sub = {**obj, "R": obj["R"][keep], "Z": obj["Z"][keep]}
            feats = contour_features(sub["R"], sub["Z"], obj["RB"],
                                     obj["RC"], obj["Bh"], obj["Th"])
            params = net(feats)
            # The leave-one-out objective NEVER constrains the base or crown
            # cap (no fold targets them), so their scalings receive zero data
            # signal. Hold them at their REFERENCE values -- classical when
            # starting from classical, the trained model's learned caps when
            # starting from a checkpoint -- instead of letting them drift.
            params = {**params,
                      "s_fB": ref[i]["s_fB"],
                      "s_fC": ref[i]["s_fC"]}
            S = surf(sub, params, n_u)
            # Compare against the GAP PATCH ONLY, not the whole surface.
            # Patch order is [base cap, interior 1..Nsub-1, crown cap], and
            # interior patch k spans subset rows k-1,k; removing row i makes
            # original rows i-1,i+1 into subset rows i-1,i, so the patch
            # that must interpolate the held-out ring is at list index i.
            # Searching the WHOLE surface instead lets the optimizer cheat
            # by ballooning a CAP through the target ring (verified: for the
            # vase's i=7 fold the nearest patch is the crown cap, not the
            # gap patch) -- low loss, destroyed shape.
            gap_patch = S[i].reshape(-1, 3)
            d_t2p, _ = _nn_sqdist(Z3(i), gap_patch)
            loss = loss + d_t2p.mean()
            prox = prox + sum(((params[k] - ref[i][k]) ** 2).mean()
                              for k in ("s_a", "s_b", "s_tau"))
        # Proximity-to-REFERENCE anchor. Keeps the UNSUPERVISED regions
        # (both caps always; also interior(0,1) on the vase, whose only fold
        # is the degenerate skipped one) at wherever the initialization put
        # them, instead of drifting wherever the shared network weights
        # happen to go -- the cause of the exploding cap cones. Supervised
        # regions still move freely, since the data term dominates there.
        loss = loss + lam_prox * prox

        # The leave-one-out data term does not directly supervise the cap.
        # Evaluate the geometry-conditioned network on the FULL object and
        # explicitly prevent radial cap turn-back during TTO.
        if lam_cap_fold != 0.0:
            f_full_now = contour_features(
                obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"]
            )
            p_full_now = net(f_full_now)
            p_full_now = {
                **p_full_now,
                "s_fB": ref_full["s_fB"],
                "s_fC": ref_full["s_fC"],
            }
            S_full_now = surf(obj, p_full_now, n_u)
            loss = loss + lam_cap_fold * cap_radial_fold_loss(
                S_full_now,
                obj["RB"],
                obj["RC"],
                closed_top=obj["closed_top"],
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if it % 50 == 0:
            print(f"  tto iter {it}: loss {loss.item():.6f}")
    # final parameters: evaluate the tuned function on the FULL object
    net.eval()
    with torch.no_grad():
        feats = contour_features(obj["R"], obj["Z"], obj["RB"], obj["RC"],
                                 obj["Bh"], obj["Th"])
        params = {k: v.detach() for k, v in net(feats).items()}
        # caps unsupervised by this objective -> hold at the reference
        # (classical if init from classical; the trained model's own
        # learned caps if init from a checkpoint)
        params["s_fB"] = ref_full["s_fB"]
        params["s_fC"] = ref_full["s_fC"]
    # Diagnostics: s is bounded to +-2 (=> tangent weights within e^{+-2}).
    # We report BOTH the absolute magnitude and the drift from the
    # reference, since with --tto_init net a large |s| may simply be what
    # the trained model already predicted, not something tto introduced.
    print(f"  tto per-row |s| (bound {c_bound:g})  and  |s - reference| drift:")
    for k in ("s_a", "s_b", "s_tau"):
        mag = params[k].abs().max(dim=1).values.cpu().numpy()
        drift = (params[k] - ref_full[k]).abs().max(dim=1).values.cpu().numpy()
        print(f"    {k:6s} |s|   " + " ".join(f"{v:.2f}" for v in mag))
        print(f"    {'':6s} drift " + " ".join(f"{v:.2f}" for v in drift))
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="banana",
                    choices=["banana", "apple", "vase"])
    ap.add_argument("--mode", default="classical",
                    choices=["classical", "net", "tto"])
    ap.add_argument("--ckpt", default="runs/exp1/best.pt")
    ap.add_argument("--n1", type=int, default=25)
    ap.add_argument("--n_u", type=int, default=40)
    ap.add_argument("--no_safe_render", action="store_true",
                    help="disable the axis-clearance projection (on by "
                         "default). The projection scales the predicted "
                         "TANGENT parameters by the largest factor that "
                         "keeps the surface off the object's axis; it is a "
                         "no-op when the prediction is already safe.")
    ap.add_argument("--min_clearance", type=float, default=0.15,
                    help="clearance target for the projection, as a "
                         "fraction of the narrowest input contour")
    ap.add_argument("--freeze_caps", action="store_true",
                    help="hold the boundary scalings s_fB/s_fC at the "
                         "CLASSICAL values while still using the learned "
                         "body tangents. The cap's height is fixed by Bh/Th "
                         "(preprocessing) and is NOT learnable, so s_fB only "
                         "controls radial bulge, while s_bh/s_th control "
                         "axial EXTENT (both are frozen by this flag); "
                         "because the caps carry ~8x "
                         "the per-point error of the body, Chamfer training "
                         "tends to shrink that bulge, which reads as a flat "
                         "pole. Use this to keep classical cap appearance "
                         "with an otherwise learned surface.")
    ap.add_argument("--tto_init", default="net", choices=["net", "classical"],
                    help="'net' = per-object fine-tune of the trained "
                         "checkpoint (usually best); 'classical' = start "
                         "from the classical pipeline, fully "
                         "training-data-free (fair vs per-shape methods)")
    ap.add_argument("--tto_iters", type=int, default=300)
    ap.add_argument("--tto_lr", type=float, default=1e-3)
    ap.add_argument("--tto_prox", type=float, default=1e-2,
                    help="anchor toward the classical solution; raise if "
                         "unsupervised regions (caps) distort")
    ap.add_argument("--tto_cap_fold", type=float, default=1e-2,
                    help="cap radial-fold penalty used during TTO")
    ap.add_argument("--max_cap_fold", type=float, default=1e-3,
                    help="maximum normalized cap radial turn-back allowed by "
                         "the inference safety projection")
    ap.add_argument("--out", default="results/designer")
    ap.add_argument("--c_bound", type=float, default=1.0,
                    help="must match the value the checkpoint was trained with")
    ap.add_argument("--no_learn_heights", action="store_true",
                    help="build ParamNet without the cap-height head; must "
                         "match how the checkpoint was TRAINED")
    ap.add_argument("--tag", default="",
                    help="extra suffix for the output filename, e.g. the "
                         "checkpoint name, so runs are not overwritten")
    ap.add_argument("--show_crown", action="store_true",
                    help="also draw the vase's crown patch (hidden by "
                         "default to match the paper's Figure 1c; the "
                         "underlying surface always includes it)")
    a = ap.parse_args()
    print(f"ParamNet c_bound: {a.c_bound:g}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float64
    os.makedirs(a.out, exist_ok=True)

    obj, pre = load_designer(a.ds, a.n1, dev, dt)
    N, m = obj["R"].shape[0], obj["R"].shape[1]

    if a.mode == "classical":
        params = zero_params(N, m, device=dev, dtype=dt)
    elif a.mode == "net":
        net = ParamNet(learn_heights=not a.no_learn_heights,
                       c_bound=a.c_bound).to(device=dev, dtype=dt)
        sd = torch.load(a.ckpt, map_location=dev)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        net.load_state_dict(sd)
        net.eval()
        with torch.no_grad():
            params = net(contour_features(obj["R"], obj["Z"], obj["RB"],
                                          obj["RC"], obj["Bh"], obj["Th"]))
            if a.freeze_caps:
                params = freeze_cap_params(params)
    else:
        init_sd = None
        if a.tto_init == "net":
            sd = torch.load(a.ckpt, map_location=dev)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            init_sd = {k: v.to(dtype=dt) for k, v in sd.items()}
        params = tto_leave_one_out(obj, a.n_u, iters=a.tto_iters,
                                   lr=a.tto_lr, lam_prox=a.tto_prox,
                                   lam_cap_fold=a.tto_cap_fold,
                                   init_net=init_sd, device=dev, dtype=dt, c_bound=a.c_bound)

    if a.freeze_caps:
        params = freeze_cap_params(params)
    S = surf(obj, params, a.n_u)
    hide_crown = pre.get("hide_crown_render", False) and not a.show_crown

    # Descriptive filename: the mode alone is ambiguous, since tto has two
    # initializations and --freeze_caps is orthogonal to both.
    tag = a.mode
    if a.mode == "tto":
        tag += f"_init-{a.tto_init}"
    if a.freeze_caps:
        tag += "_freezecaps"
    if a.tag:
        tag += f"_{a.tag}"
    label = f"{a.ds} — NSSR ({a.mode}"
    if a.mode == "tto":
        label += f", init={a.tto_init}"
    if a.freeze_caps:
        label += ", caps frozen"
    label += ")"
    if not a.no_safe_render and a.mode != "classical":
        params, alpha, r0, r1, f0, f1, stage = project_to_safe(
            obj, params, a.n_u, surf,
            min_ratio=a.min_clearance,
            max_cap_fold=a.max_cap_fold)
        if alpha < 1.0:
            print(
                f"safety projection ({stage}): axis {r0:.3f} -> {r1:.3f}, "
                f"cap-turnback {f0:.5f} -> {f1:.5f} "
                f"(alpha={alpha:.3f}, retained {alpha*100:.0f}% correction)"
            )
            S = surf(obj, params, a.n_u)
    clr, ratio = axis_clearance(S, obj["R"])
    print(f"axis clearance: {clr:.4f} ({ratio*100:.1f}% of the narrowest "
          f"input contour)")
    _, final_ref = classical_geometry_and_reference(obj, a.n_u)
    final_geom = geometry_from_params(obj, params, a.n_u, final_ref)
    final_safety = geometry_safety_summary(
        final_geom, obj, a.max_cap_fold
    )
    print(f"cap turn-back max: {final_safety['cap_fold_max']:.6f}")
    print(
        f"Jacobian validity: "
        f"{'OK' if final_safety['jacobian_valid'] else 'FAIL'} "
        f"(negative={final_safety['negative_jacobian']}, "
        f"degenerate={final_safety['degenerate']})"
    )
    if ratio < 0.10:
        print("  WARNING: the surface is collapsing onto the object axis. "
              "This produces a pinch-to-a-point and a flared 'cone' in the "
              "wireframe. Lower --c_bound (1.0 is safe on all three designer "
              "shapes; 2.0 is not) and retrain, or raise --reg.")
    out_png = os.path.join(a.out, f"{a.ds}_{tag}.png")
    render(S, out_png, label, hide_crown=hide_crown)


if __name__ == "__main__":
    main()
