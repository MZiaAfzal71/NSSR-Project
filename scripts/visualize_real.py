"""Visualize real-mesh results: GROUND TRUTH vs NSSR reconstruction.

Produces, per mesh, a figure with up to four panels:
   1. ground-truth mesh surface
   2. the input slice contours (what the method actually sees)
   3. classical reconstruction
   4. learned reconstruction (with --ckpt), coloured by distance to GT

This is the figure the paper needs for its real-data section: it shows the
true object, how little information the slices carry, and what each method
recovers from them.

Usage:
    # after building a real dataset
    python scripts/visualize_real.py --data data/real --N 9 \
        --ckpt runs/real_N9/best.pt --n 6 --out results/real_figs

    # or straight from mesh files
    python scripts/visualize_real.py --meshes data/meshes --N 9 \
        --ckpt runs/real_N9/best.pt --n 6 --out results/real_figs
"""
import sys, os, argparse, pickle, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401

MESH_EXTS = (".obj", ".ply", ".stl", ".off")


def _equal_aspect(ax, P):
    c = (P.max(0) + P.min(0)) / 2
    r = (P.max(0) - P.min(0)).max() / 2 or 1.0
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")


def panel_points(fig, pos, P, title, c=None, cmap="viridis", s=1.0,
                 elev=14, azim=20, cbar_label=None):
    ax = fig.add_subplot(*pos, projection="3d")
    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=s,
                    c=(c if c is not None else "0.35"), cmap=cmap,
                    linewidths=0, alpha=0.9)
    _equal_aspect(ax, P)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10, pad=0)
    if c is not None and cbar_label:
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    return ax


def panel_contours(fig, pos, contours, Z, title, elev=14, azim=20):
    ax = fig.add_subplot(*pos, projection="3d")
    allp = []
    for C, z in zip(contours, Z):
        C = np.asarray(C)
        loop = np.vstack([C, C[:1]])
        ax.plot(loop[:, 0], loop[:, 1], np.full(len(loop), z),
                lw=1.1, color="#c1121f")
        allp.append(np.column_stack([loop, np.full(len(loop), z)]))
    _equal_aspect(ax, np.vstack(allp))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10, pad=0)


def nn_dist(A, B, chunk=2048):
    from scipy.spatial import cKDTree
    return cKDTree(B).query(A[::1])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="", help="dataset dir (test_N*.pkl)")
    ap.add_argument("--meshes", default="", help="or a mesh directory")
    ap.add_argument("--N", type=int, default=9)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n_u", type=int, default=28)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--n", type=int, default=6, help="how many objects")
    ap.add_argument("--out", default="results/real_figs")
    ap.add_argument("--elev", type=float, default=14)
    ap.add_argument("--azim", type=float, default=20)
    ap.add_argument("--pole_zoom", action="store_true",
                    help="add a close-up of the base and crown poles, "
                         "classical vs learned, coloured by distance to GT. "
                         "The 2x2 cap ablation found the LEARNED cap is ~43%% "
                         "more accurate than classical even though it reads "
                         "as visually flatter -- this panel shows that "
                         "directly, which is the evidence a reader needs "
                         "for a counter-intuitive result.")
    ap.add_argument("--pole_frac", type=float, default=0.18,
                    help="fraction of the object's height included in the "
                         "pole close-up")
    a = ap.parse_args()

    import torch
    from nssr.train import to_torch, forward_object
    from nssr.geometry import hermite_surface, zero_params, surface_points
    from nssr.networks import ParamNet

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32

    # ---- gather samples ---------------------------------------------------
    samples = []
    if a.data:
        p = os.path.join(a.data, f"test_N{a.N}.pkl")
        if not os.path.exists(p):
            p = os.path.join(a.data, f"train_N{a.N}.pkl")
        with open(p, "rb") as f:
            samples = pickle.load(f)[:a.n]
        print(f"loaded {len(samples)} samples from {p}")
    elif a.meshes:
        from nssr.slicing import make_sample_from_mesh
        paths = []
        for ext in MESH_EXTS:
            paths.extend(glob.glob(os.path.join(a.meshes, "**", f"*{ext}"),
                                   recursive=True))
        for p in sorted(paths):
            if len(samples) >= a.n:
                break
            s, reason = make_sample_from_mesh(p, N=a.N)
            if s is None:
                print(f"  skip {os.path.basename(p)}: {reason}")
                continue
            samples.append(s)
        print(f"sliced {len(samples)} meshes from {a.meshes}")
    else:
        print("give --data or --meshes")
        return 1
    if not samples:
        print("no usable samples")
        return 1

    net = None
    if a.ckpt and os.path.exists(a.ckpt):
        net = ParamNet().to(device=dev, dtype=dt)
        net.load_state_dict(torch.load(a.ckpt, map_location=dev))
        net.eval()
    else:
        print("(no --ckpt: classical only)")

    ncols = 4 if net is not None else 3
    for i, s in enumerate(samples):
        obj = to_torch(s, a.m, dev, dt, seed=i)
        gt = obj["gt_pts"].cpu().numpy()

        with torch.no_grad():
            p0 = zero_params(obj["R"].shape[0], a.m, device=dev, dtype=dt)
            S0 = hermite_surface(obj["R"], obj["Z"], obj["RB"], obj["RC"],
                                 obj["Bh"], obj["Th"], p0, n_u=a.n_u,
                                 closed_top=obj.get("closed_top", True),
                                 base_circular=obj.get("base_circular", True),
                                 crown_circular=obj.get("crown_circular", True))
            P0 = surface_points(S0).cpu().numpy()
            P1 = None
            if net is not None:
                _, pts, _, _ = forward_object(net, obj, n_u=a.n_u)
                P1 = pts.cpu().numpy()

        # subsample for plotting
        rng = np.random.default_rng(0)
        gsub = gt[rng.choice(len(gt), min(9000, len(gt)), replace=False)]
        p0sub = P0[rng.choice(len(P0), min(9000, len(P0)), replace=False)]
        d0 = nn_dist(p0sub, gt)

        fig = plt.figure(figsize=(4.1 * ncols, 4.6))
        panel_points(fig, (1, ncols, 1), gsub, "(a) ground-truth mesh",
                     elev=a.elev, azim=a.azim)
        # contours in the SAME normalized frame as everything else
        Rn = obj["R"].cpu().numpy(); Zn = obj["Z"].cpu().numpy()
        panel_contours(fig, (1, ncols, 2), list(Rn), Zn,
                       f"(b) input: {len(Zn)} slices",
                       elev=a.elev, azim=a.azim)
        panel_points(fig, (1, ncols, 3), p0sub,
                     f"(c) classical  (mean err {d0.mean():.4f})",
                     c=d0, cmap="magma", elev=a.elev, azim=a.azim,
                     cbar_label="distance to GT")
        if P1 is not None:
            p1sub = P1[rng.choice(len(P1), min(9000, len(P1)), replace=False)]
            d1 = nn_dist(p1sub, gt)
            imp = 100 * (d0.mean() - d1.mean()) / max(d0.mean(), 1e-12)
            panel_points(fig, (1, ncols, 4), p1sub,
                         f"(d) NSSR learned  (mean err {d1.mean():.4f}, "
                         f"{imp:+.1f}%)",
                         c=d1, cmap="magma", elev=a.elev, azim=a.azim,
                         cbar_label="distance to GT")
            # share the colour scale so panels are comparable
            vmax = max(d0.max(), d1.max())
            for ax in fig.axes:
                for coll in ax.collections:
                    if coll.get_array() is not None:
                        coll.set_clim(0, vmax)

        if a.pole_zoom:
            zlo, zhi = gt[:, 2].min(), gt[:, 2].max()
            band = a.pole_frac * (zhi - zlo)
            for which, sel_z in (("base", lambda P: P[:, 2] <= zlo + band),
                                 ("crown", lambda P: P[:, 2] >= zhi - band)):
                sets = [("ground truth", gt[sel_z(gt)], None),
                        ("classical", P0[sel_z(P0)], None)]
                if P1 is not None:
                    sets.append(("NSSR learned", P1[sel_z(P1)], None))
                fz = plt.figure(figsize=(4.1 * len(sets), 4.4))
                vmaxp = 0.0
                dists = []
                for k, (lbl, Q, _) in enumerate(sets):
                    d = nn_dist(Q, gt) if lbl != "ground truth" else None
                    dists.append(d)
                    if d is not None and len(d):
                        vmaxp = max(vmaxp, float(d.max()))
                for k, ((lbl, Q, _), d) in enumerate(zip(sets, dists)):
                    ttl = lbl if d is None else f"{lbl}  (mean {d.mean():.4f})"
                    panel_points(fz, (1, len(sets), k + 1), Q, ttl,
                                 c=d, cmap="magma", s=3.0,
                                 elev=6, azim=a.azim,
                                 cbar_label=("distance to GT"
                                             if d is not None else None))
                for ax in fz.axes:
                    for coll in ax.collections:
                        if coll.get_array() is not None and vmaxp > 0:
                            coll.set_clim(0, vmaxp)
                nm = os.path.basename(s.get("path", f"object_{i}")).split(".")[0]
                fz.suptitle(f"{nm} — {which} pole close-up", fontsize=11)
                fz.tight_layout()
                pp = os.path.join(a.out, f"pole_{which}_{i:02d}_{nm}.png")
                fz.savefig(pp, dpi=160, bbox_inches="tight")
                plt.close(fz)
                print(f"wrote {pp}")

        name = os.path.basename(s.get("path", f"object_{i}")).split(".")[0]
        fig.suptitle(f"{name}", fontsize=11)
        fig.tight_layout()
        outp = os.path.join(a.out, f"real_{i:02d}_{name}.png")
        fig.savefig(outp, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {outp}")

    print(f"\n{len(samples)} figures in {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
