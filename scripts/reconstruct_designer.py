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
from nssr.losses import chamfer


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


def render(S, path, title):
    S = S.detach().cpu().numpy()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    for p in range(S.shape[0]):
        ax.plot_wireframe(S[p, :, :, 0], S[p, :, :, 1], S[p, :, :, 2],
                          rcount=10, ccount=24, linewidth=0.5, color="k")
    ax.axis("equal"); ax.axis("off"); ax.view_init(elev=12, azim=15)
    fig.suptitle(title, fontsize=11, y=0.04)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", path)


def tto_leave_one_out(obj, n_u, iters=400, lr=3e-2, reg=1e-3):
    """Optimize s-fields so that, with each interior slice held out in turn,
    the surface built from the remaining slices passes through it."""
    N, m = obj["R"].shape[0], obj["R"].shape[1]
    dev, dt = obj["R"].device, obj["R"].dtype
    raw = {k: torch.zeros(N, m, device=dev, dtype=dt, requires_grad=True)
           for k in ("s_a", "s_b", "s_tau")}
    rawB = torch.zeros(m, device=dev, dtype=dt, requires_grad=True)
    rawC = torch.zeros(m, device=dev, dtype=dt, requires_grad=True)
    opt = torch.optim.Adam(list(raw.values()) + [rawB, rawC], lr=lr)
    interior = list(range(1, N - 1))
    Z3 = lambda i: torch.cat([obj["R"][i],
                              obj["Z"][i].expand(m, 1)], dim=1)
    for it in range(iters):
        opt.zero_grad()
        loss = 0.0
        for i in interior:
            keep = [k for k in range(N) if k != i]
            sub = {**obj,
                   "R": obj["R"][keep], "Z": obj["Z"][keep]}
            params = {"s_a": 2*torch.tanh(raw["s_a"][keep]),
                      "s_b": 2*torch.tanh(raw["s_b"][keep]),
                      "s_tau": 2*torch.tanh(raw["s_tau"][keep]),
                      "s_fB": 2*torch.tanh(rawB),
                      "s_fC": 2*torch.tanh(rawC)}
            S = surf(sub, params, n_u)
            loss = loss + chamfer(surface_points(S), Z3(i))
        for v in list(raw.values()) + [rawB, rawC]:
            loss = loss + reg * (v ** 2).mean()
        loss.backward(); opt.step()
        if it % 50 == 0:
            print(f"  tto iter {it}: loss {loss.item():.6f}")
    params = {"s_a": 2*torch.tanh(raw["s_a"]).detach(),
              "s_b": 2*torch.tanh(raw["s_b"]).detach(),
              "s_tau": 2*torch.tanh(raw["s_tau"]).detach(),
              "s_fB": 2*torch.tanh(rawB).detach(),
              "s_fC": 2*torch.tanh(rawC).detach()}
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
    ap.add_argument("--out", default="results/designer")
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
        params = tto_leave_one_out(obj, a.n_u)

    S = surf(obj, params, a.n_u)
    render(S, os.path.join(a.out, f"{a.ds}_{a.mode}.png"),
           f"{a.ds} — NSSR ({a.mode})")


if __name__ == "__main__":
    main()
