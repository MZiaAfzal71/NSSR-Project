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


def tto_leave_one_out(obj, n_u, iters=300, lr=1e-3, reg=0.0,
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
    interior = list(range(1, N - 1))
    Z3 = lambda i: torch.cat([obj["R"][i],
                              obj["Z"][i].expand(m, 1)], dim=1)
    for it in range(iters):
        opt.zero_grad()
        loss = 0.0
        for i in interior:
            keep = [k for k in range(N) if k != i]
            sub = {**obj, "R": obj["R"][keep], "Z": obj["Z"][keep]}
            feats = contour_features(sub["R"], sub["Z"], obj["RB"],
                                     obj["RC"], obj["Bh"], obj["Th"])
            params = net(feats)
            S = surf(sub, params, n_u)
            d_t2p, _ = _nn_sqdist(Z3(i), surface_points(S))
            loss = loss + d_t2p.mean()
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
                                   lr=a.tto_lr, init_net=init_sd,
                                   device=dev, dtype=dt)

    S = surf(obj, params, a.n_u)
    hide_crown = pre.get("hide_crown_render", False) and not a.show_crown
    render(S, os.path.join(a.out, f"{a.ds}_{a.mode}.png"),
           f"{a.ds} — NSSR ({a.mode})", hide_crown=hide_crown)


if __name__ == "__main__":
    main()
