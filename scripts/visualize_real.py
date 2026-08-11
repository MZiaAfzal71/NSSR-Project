"""Publication-quality real-mesh visualization for NSSR-V2.

Panels:
  GT mesh points | input contours | classical | learned/projected NSSR

Important rendering detail:
  ``close_periodic_surface`` appends the first circumferential sample as the
  final display column.  This closes the visible seam in plot_surface/export
  WITHOUT adding a duplicate point to training, metrics, or Jacobian checks.
"""
from __future__ import annotations
import argparse, glob, os, pickle, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MESH_EXTS = (".obj", ".ply", ".stl", ".off")


def close_periodic_surface(S):
    """Append first circumferential point only for display/export."""
    A = np.asarray(S)
    if A.ndim < 3 or A.shape[-1] != 3:
        raise ValueError("surface must end in (..., m, 3)")
    return np.concatenate([A, A[..., :1, :]], axis=-2)


def _equal_aspect(ax, P):
    P = np.asarray(P).reshape(-1, 3)
    c = (P.max(0) + P.min(0)) / 2
    r = max(float((P.max(0) - P.min(0)).max() / 2), 1e-8)
    ax.set_xlim(c[0]-r, c[0]+r)
    ax.set_ylim(c[1]-r, c[1]+r)
    ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")


def panel_points(fig, pos, P, title, c=None, elev=14, azim=20):
    ax = fig.add_subplot(*pos, projection="3d")
    kw = {}
    if c is not None:
        kw["c"] = c
        kw["cmap"] = "magma"
    else:
        kw["c"] = "0.35"
    ax.scatter(P[:,0], P[:,1], P[:,2], s=1.0, linewidths=0, alpha=.85, **kw)
    _equal_aspect(ax, P)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)
    return ax


def panel_surface(fig, pos, S, title, elev=14, azim=20):
    ax = fig.add_subplot(*pos, projection="3d")
    # Flatten patch/u into one meridional display axis, then repeat seam.
    G = np.asarray(S).reshape(-1, S.shape[-2], 3)
    G = close_periodic_surface(G)
    ax.plot_surface(
        G[:,:,0], G[:,:,1], G[:,:,2],
        rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=.92
    )
    _equal_aspect(ax, G)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)
    return ax


def panel_contours(fig, pos, R, Z, title, elev=14, azim=20):
    ax = fig.add_subplot(*pos, projection="3d")
    allp = []
    for C, z in zip(R, Z):
        C = np.asarray(C)
        loop = np.concatenate([C, C[:1]], axis=0)
        P = np.column_stack([loop, np.full(len(loop), z)])
        ax.plot(P[:,0], P[:,1], P[:,2], lw=1.0)
        allp.append(P)
    _equal_aspect(ax, np.vstack(allp))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)
    return ax


def nn_dist(A, B):
    from scipy.spatial import cKDTree
    return cKDTree(B).query(A)[0]


def load_state(path, device):
    import torch
    s = torch.load(path, map_location=device)
    return s["state_dict"] if isinstance(s, dict) and "state_dict" in s else s


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--data", default="")
    ap.add_argument("--meshes", default="")
    ap.add_argument("--axis_select", choices=["longest","search"], default="search")
    ap.add_argument("--N", type=int, default=15)
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--n_u", type=int, default=16)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="results/real_figs")
    ap.add_argument("--elev", type=float, default=14)
    ap.add_argument("--azim", type=float, default=20)
    ap.add_argument("--pole_zoom", action="store_true")
    ap.add_argument("--pole_frac", type=float, default=.18)
    ap.add_argument("--surface_render", action="store_true")
    ap.add_argument("--project_safe", action="store_true")
    ap.add_argument("--max_cap_fold", type=float, default=1e-3)
    ap.add_argument("--c_bound", type=float, default=1.0)
    a = ap.parse_args()

    import torch
    from nssr.train import to_torch
    from nssr.geometry import surface_points, zero_params
    from nssr.networks import ParamNet
    from nssr.safety import (
        classical_geometry_and_reference,
        geometry_from_params,
        params_from_net,
        project_staged_to_safe,
    )

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32

    samples = []
    if a.data:
        p = os.path.join(a.data, f"test_N{a.N}.pkl")
        with open(p, "rb") as f:
            samples = pickle.load(f)[:a.n]
    elif a.meshes:
        from nssr.slicing import make_sample_from_mesh
        paths = []
        for ext in MESH_EXTS:
            paths += glob.glob(os.path.join(a.meshes, "**", f"*{ext}"), recursive=True)
        for p in sorted(paths):
            if len(samples) >= a.n:
                break
            s, reason = make_sample_from_mesh(p, N=a.N, axis_select=a.axis_select)
            if s is not None:
                samples.append(s)
    else:
        raise SystemExit("give --data or --meshes")

    net = None
    if a.ckpt:
        net = ParamNet(c_bound=a.c_bound).to(device=dev, dtype=dt)
        net.load_state_dict(load_state(a.ckpt, dev))
        net.eval()

    for i, s in enumerate(samples):
        obj = to_torch(s, m=a.m, device=dev, dtype=dt,
                       gt_subsample=20000, seed=40000+i)
        gt = obj["gt_pts"].cpu().numpy()

        with torch.no_grad():
            classical, ref = classical_geometry_and_reference(obj, a.n_u)
            learned = None
            stage = "none"
            if net is not None:
                params = params_from_net(net, obj)
                if a.project_safe:
                    params, _, stage, *_ = project_staged_to_safe(
                        obj, params, a.n_u, ref,
                        max_cap_fold=a.max_cap_fold,
                        max_iter=40, with_metrics=False
                    )
                learned = geometry_from_params(obj, params, a.n_u, ref)

        S0 = classical.surface.xyz.cpu().numpy()
        P0 = surface_points(classical.surface.xyz).cpu().numpy()
        P1 = None if learned is None else surface_points(learned.surface.xyz).cpu().numpy()
        S1 = None if learned is None else learned.surface.xyz.cpu().numpy()

        rng = np.random.default_rng(0)
        gsub = gt[rng.choice(len(gt), min(9000,len(gt)), replace=False)]
        p0sub = P0[rng.choice(len(P0), min(9000,len(P0)), replace=False)]
        d0 = nn_dist(p0sub, gt)

        ncols = 4 if P1 is not None else 3
        fig = plt.figure(figsize=(4.15*ncols, 4.6))
        panel_points(fig, (1,ncols,1), gsub, "(a) ground truth", elev=a.elev, azim=a.azim)

        Rn, Zn = obj["R"].cpu().numpy(), obj["Z"].cpu().numpy()
        panel_contours(fig, (1,ncols,2), Rn, Zn, f"(b) input: {len(Zn)} slices",
                       elev=a.elev, azim=a.azim)

        if a.surface_render:
            panel_surface(fig, (1,ncols,3), S0, "(c) classical",
                          elev=a.elev, azim=a.azim)
        else:
            panel_points(fig, (1,ncols,3), p0sub,
                         f"(c) classical err={d0.mean():.4f}", c=d0,
                         elev=a.elev, azim=a.azim)

        if P1 is not None:
            p1sub = P1[rng.choice(len(P1), min(9000,len(P1)), replace=False)]
            d1 = nn_dist(p1sub, gt)
            imp = 100*(d0.mean()-d1.mean())/max(d0.mean(),1e-12)
            ttl = f"(d) NSSR {stage} {imp:+.1f}%"
            if a.surface_render:
                panel_surface(fig, (1,ncols,4), S1, ttl,
                              elev=a.elev, azim=a.azim)
            else:
                panel_points(fig, (1,ncols,4), p1sub, ttl, c=d1,
                             elev=a.elev, azim=a.azim)

        name = os.path.basename(s.get("path", f"object_{i}")).split(".")[0]
        fig.suptitle(name, fontsize=11)
        fig.tight_layout()
        pfig = os.path.join(a.out, f"real_{i:02d}_{name}.png")
        fig.savefig(pfig, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print("wrote", pfig)

        if a.pole_zoom:
            zlo, zhi = gt[:,2].min(), gt[:,2].max()
            band = a.pole_frac*(zhi-zlo)
            for which, pred in (
                ("base", lambda P: P[:,2] <= zlo+band),
                ("crown", lambda P: P[:,2] >= zhi-band),
            ):
                sets = [("GT", gt[pred(gt)]), ("classical", P0[pred(P0)])]
                if P1 is not None:
                    sets.append(("NSSR", P1[pred(P1)]))
                fz = plt.figure(figsize=(4.1*len(sets),4.2))
                for k,(lbl,Q) in enumerate(sets):
                    panel_points(fz, (1,len(sets),k+1), Q, lbl,
                                 elev=6, azim=a.azim)
                fz.suptitle(f"{name} - {which} pole", fontsize=11)
                fz.tight_layout()
                pp = os.path.join(a.out, f"pole_{which}_{i:02d}_{name}.png")
                fz.savefig(pp, dpi=220, bbox_inches="tight")
                plt.close(fz)
                print("wrote", pp)


if __name__ == "__main__":
    main()
