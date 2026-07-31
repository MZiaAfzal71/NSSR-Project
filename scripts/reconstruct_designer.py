"""Run NSSR on the CiSE designer shapes (banana / apple / vase).

Modes
-----
--mode classical   render the classical (params = 0) reconstruction through
                   the NSSR torch pipeline (should reproduce your paper's
                   Figure 1, since parity is verified).
--mode net         load a trained ParamNet checkpoint (--ckpt) and render
                   the learned reconstruction (generalization test:
                   trained on synthetic, applied to designer shapes).
--mode tto         test-time optimization, leave-one-slice-out: optimize
                   the s-fields directly (no network, no training data)
                   so the surface reconstructed WITHOUT slice i passes
                   through slice i, for each interior i.  This is the
                   training-data-free operating mode (METHOD.md sec. 7.5).

Usage:  python scripts/reconstruct_designer.py --ds banana --mode classical
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nssr.preprocess import preprocess_designer
from nssr.geometry import hermite_surface, zero_params, surface_points
from nssr.networks import ParamNet, contour_features
from nssr.losses import chamfer, _nn_sqdist


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
                      init_net=None, device="cpu", dtype=torch.float64):
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
    net = ParamNet().to(device=device, dtype=dtype)
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
    print("  tto per-row |s| (bound 2.0)  and  |s - reference| drift:")
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
    ap.add_argument("--out", default="results/designer")
    ap.add_argument("--show_crown", action="store_true",
                    help="also draw the vase's crown patch (hidden by "
                         "default to match the paper's Figure 1c; the "
                         "underlying surface always includes it)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float64
    os.makedirs(a.out, exist_ok=True)

    obj, pre = load_designer(a.ds, a.n1, dev, dt)
    N, m = obj["R"].shape[0], obj["R"].shape[1]

    if a.mode == "classical":
        params = zero_params(N, m, device=dev, dtype=dt)
    elif a.mode == "net":
        net = ParamNet().to(device=dev, dtype=dt)
        net.load_state_dict(torch.load(a.ckpt, map_location=dev))
        net.eval()
        with torch.no_grad():
            params = net(contour_features(obj["R"], obj["Z"], obj["RB"],
                                          obj["RC"], obj["Bh"], obj["Th"]))
    else:
        init_sd = None
        if a.tto_init == "net":
            sd = torch.load(a.ckpt, map_location=dev)
            init_sd = {k: v.to(dtype=dt) for k, v in sd.items()}
        params = tto_leave_one_out(obj, a.n_u, iters=a.tto_iters,
                                   lr=a.tto_lr, lam_prox=a.tto_prox,
                                   init_net=init_sd, device=dev, dtype=dt)

    S = surf(obj, params, a.n_u)
    hide_crown = pre.get("hide_crown_render", False) and not a.show_crown
    render(S, os.path.join(a.out, f"{a.ds}_{a.mode}.png"),
           f"{a.ds} — NSSR ({a.mode})", hide_crown=hide_crown)


if __name__ == "__main__":
    main()
