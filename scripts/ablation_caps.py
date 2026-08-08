"""2x2 ablation: cap weighting x learnable cap heights (V2-compatible)."""
import sys, os, argparse, pickle, csv, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from nssr.train import to_torch
from nssr.geometry import hermite_surface, zero_params, surface_points
from nssr.networks import ParamNet, contour_features
from nssr.losses import _nn_sqdist

ARMS = [
    ("A_w1.00_hOFF", 1.00, False),
    ("B_w1.00_hON", 1.00, True),
    ("C_w0.25_hOFF", 0.25, False),
    ("D_w0.25_hON", 0.25, True),
]

def build_surface(obj, params, n_u):
    return hermite_surface(
        obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"], params,
        n_u=n_u,
        closed_top=obj.get("closed_top", True),
        base_circular=obj.get("base_circular", True),
        crown_circular=obj.get("crown_circular", True),
    )

def cap_point_mask_local(S, closed_top=True):
    P, nu, m, _ = S.shape
    mask = torch.zeros(P, nu, m, dtype=torch.bool, device=S.device)
    mask[0] = True
    if closed_top:
        mask[-1] = True
    return mask.reshape(-1)

@torch.no_grad()
def split_errors(S, gt_pts, closed_top=True):
    pts = surface_points(S)
    mask = cap_point_mask_local(S, closed_top=closed_top)
    d, _ = _nn_sqdist(pts, gt_pts)
    d = d.sqrt()
    cap = d[mask]
    body = d[~mask]
    return (
        cap.mean().item() if cap.numel() else float("nan"),
        body.mean().item() if body.numel() else float("nan"),
        d.mean().item(),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--N", type=int, default=9)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n_u", type=int, default=28)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--runs_dir", default="runs/caps_ablation")
    ap.add_argument("--out", default="results/ablation_caps.csv")
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--c_bound", type=float, default=1.0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32
    os.makedirs(a.runs_dir, exist_ok=True)

    if not a.skip_train:
        for name, w, h in ARMS:
            outdir = os.path.join(a.runs_dir, name)
            if os.path.exists(os.path.join(outdir, "best.pt")):
                print(f"[skip] {name} already trained")
                continue
            cmd = [
                sys.executable, "scripts/train_model.py",
                "--data", a.data, "--N", str(a.N), "--m", str(a.m),
                "--epochs", str(a.epochs),
                "--surf_sub", "8000", "--gt_sub", "8000",
                "--val_every", "5", "--val_subset", "20", "--patience", "20",
                "--cap_weight", str(w), "--c_bound", str(a.c_bound), "--out", outdir,
            ]
            if not h:
                cmd.append("--no_learn_heights")
            print("\n>>> " + " ".join(cmd))
            subprocess.run(cmd, check=True)

    with open(os.path.join(a.data, f"test_N{a.N}.pkl"), "rb") as f:
        test = pickle.load(f)[:a.n_test]
    objs = [to_torch(s, a.m, dev, dt, seed=7000 + i) for i, s in enumerate(test)]

    rows = []
    cap_c, body_c, all_c = [], [], []
    for obj in objs:
        p0 = zero_params(obj["R"].shape[0], a.m, device=dev, dtype=dt)
        S = build_surface(obj, p0, a.n_u)
        c, b, t = split_errors(S, obj["gt_pts"], obj.get("closed_top", True))
        cap_c.append(c); body_c.append(b); all_c.append(t)
    rows.append({
        "arm": "classical", "cap_weight": "-", "heights": "-",
        "cap_err": np.mean(cap_c), "body_err": np.mean(body_c), "all_err": np.mean(all_c),
    })

    for name, w, h in ARMS:
        ck = os.path.join(a.runs_dir, name, "best.pt")
        if not os.path.exists(ck):
            print(f"[missing] {ck} -- skipping {name}")
            continue
        net = ParamNet(learn_heights=h, c_bound=a.c_bound).to(device=dev, dtype=dt)
        net.load_state_dict(torch.load(ck, map_location=dev))
        net.eval()
        caps, bodies, alls = [], [], []
        with torch.no_grad():
            for obj in objs:
                feats = contour_features(obj["R"], obj["Z"], obj["RB"], obj["RC"], obj["Bh"], obj["Th"])
                S = build_surface(obj, net(feats), a.n_u)
                c, b, t = split_errors(S, obj["gt_pts"], obj.get("closed_top", True))
                caps.append(c); bodies.append(b); alls.append(t)
        rows.append({
            "arm": name, "cap_weight": w, "heights": h,
            "cap_err": np.mean(caps), "body_err": np.mean(bodies), "all_err": np.mean(alls),
        })

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)

    for r in rows:
        print(f"{r['arm']:20s} cap={float(r['cap_err']):.6f} body={float(r['body_err']):.6f} all={float(r['all_err']):.6f}")
    print("wrote", a.out)

if __name__ == "__main__":
    main()
