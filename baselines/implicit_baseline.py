"""Minimal OReX-style neural-implicit baseline: per-object occupancy MLP
with Fourier features, trained on in/out labels sampled on the slice
planes, surface extracted by dense grid + marching cubes (skimage).

This is the SIMPLE baseline for the ablation table. For the headline
comparison also run the official OReX code (github.com/haimsaw/OReX)
on the same inputs.

Usage: python baselines/implicit_baseline.py --sample data/synthetic/test_N7.pkl --idx 0
"""
import sys, os, argparse, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch, torch.nn as nn
from matplotlib.path import Path as MplPath

class FourierMLP(nn.Module):
    def __init__(self, n_freq=6, hidden=128, layers=5):
        super().__init__()
        self.B = torch.randn(3, n_freq * 3) * 4.0
        d = n_freq * 6
        seq = [nn.Linear(d, hidden), nn.SiLU()]
        for _ in range(layers - 2):
            seq += [nn.Linear(hidden, hidden), nn.SiLU()]
        seq += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*seq)
    def forward(self, x):
        proj = x @ self.B.to(x.device, x.dtype)
        h = torch.cat([torch.sin(proj), torch.cos(proj)], -1)
        return self.net(h).squeeze(-1)

def plane_samples(contours, Z, n_per_plane=4000, pad=1.3, rng=None):
    rng = rng or np.random.default_rng(0)
    X, Y = [], []
    for C, z in zip(contours, Z):
        lo, hi = C.min(0), C.max(0)
        c, half = (lo+hi)/2, (hi-lo)/2 * pad
        pts = c + (rng.uniform(-1, 1, (n_per_plane, 2)) * half)
        inside = MplPath(C).contains_points(pts)
        # add boundary-hugging samples (crucial, as in OReX)
        bidx = rng.integers(0, len(C), n_per_plane // 2)
        bpts = C[bidx] + rng.normal(0, 0.02*half.mean(), (n_per_plane//2, 2))
        binside = MplPath(C).contains_points(bpts)
        pts = np.vstack([pts, bpts]); inside = np.concatenate([inside, binside])
        X.append(np.column_stack([pts, np.full(len(pts), z)]))
        Y.append(inside.astype(np.float64))
    return np.vstack(X), np.concatenate(Y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--grid", type=int, default=192)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(a.sample, "rb") as f:
        s = pickle.load(f)[a.idx]
    X, Y = plane_samples(s["contours"], s["Z"])
    X = torch.tensor(X, dtype=torch.float32, device=dev)
    Y = torch.tensor(Y, dtype=torch.float32, device=dev)
    net = FourierMLP().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    bce = nn.BCEWithLogitsLoss()
    for it in range(a.iters):
        idx = torch.randint(0, len(X), (4096,), device=dev)
        loss = bce(net(X[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"iter {it}: bce {loss.item():.4f}")
    from skimage.measure import marching_cubes
    g = a.grid
    lo = X.cpu().numpy()[:, :3].min(0) - 0.1
    hi = X.cpu().numpy()[:, :3].max(0) + 0.1
    axes = [np.linspace(lo[d], hi[d], g) for d in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    occ = []
    with torch.no_grad():
        for i in range(0, len(G), 65536):
            occ.append(torch.sigmoid(net(torch.tensor(
                G[i:i+65536], dtype=torch.float32, device=dev))).cpu().numpy())
    occ = np.concatenate(occ).reshape(g, g, g)
    verts, faces, _, _ = marching_cubes(occ, 0.5, spacing=[
        (hi[d]-lo[d])/(g-1) for d in range(3)])
    verts += lo
    from nssr.metrics import evaluate_surface
    P = torch.tensor(verts, dtype=torch.float32)
    Q = torch.tensor(s["gt_pts"], dtype=torch.float32)
    print(evaluate_surface(P, Q))

if __name__ == "__main__":
    main()
