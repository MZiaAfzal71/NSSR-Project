"""2x2 ablation: cap_weight on/off  x  learnable cap heights on/off.

Reports CAP-REGION error separately from BODY error. This separation is the
point: caps are ~17% of surface points but carry ~8x the per-point error,
so a single Chamfer number is dominated by the body and cannot show whether
a cap intervention did anything. Every previous attempt to judge these
changes from renders or from one aggregate number was inconclusive for
exactly this reason.

Trains four models, then evaluates each on the test split:

    A  cap_weight=1.00, heights OFF   (closest to the original model)
    B  cap_weight=1.00, heights ON
    C  cap_weight=0.25, heights OFF
    D  cap_weight=0.25, heights ON    (both remedies)

plus the classical baseline for reference.

Usage:
    python scripts/ablation_caps.py --data data/synthetic --N 9 \
        --epochs 60 --out results/ablation_caps.csv

    # skip training and just score existing runs
    python scripts/ablation_caps.py --data data/synthetic --N 9 \
        --skip_train --runs_dir runs/caps_ablation
"""
import sys, os, argparse, pickle, csv, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

from nssr.train import to_torch, cap_point_mask
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
    return hermite_surface(obj["R"], obj["Z"], obj["RB"], obj["RC"],
                           obj["Bh"], obj["Th"], params, n_u=n_u,
                           closed_top=obj.get("closed_top", True),
                           base_circular=obj.get("base_circular", True),
                           crown_circular=obj.get("crown_circular", True))


@torch.no_grad()
def split_errors(S, gt_pts):
    """Mean distance to GT, reported separately for cap and body points."""
    pts = surface_points(S)
    mask = cap_point_mask(S)
    d, _ = _nn_sqdist(pts, gt_pts)
    d = d.sqrt()
    cap = d[mask]
    body = d[~mask]
    return (cap.mean().item() if cap.numel() else float("nan"),
            body.mean().item() if body.numel() else float("nan"),
            d.mean().item())


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
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32
    os.makedirs(a.runs_dir, exist_ok=True)

    # ---- train the four arms ---------------------------------------------
    if not a.skip_train:
        for name, w, h in ARMS:
            outdir = os.path.join(a.runs_dir, name)
            if os.path.exists(os.path.join(outdir, "best.pt")):
                print(f"[skip] {name} already trained")
                continue
            cmd = [sys.executable, "scripts/train_model.py",
                   "--data", a.data, "--N", str(a.N), "--m", str(a.m),
                   "--epochs", str(a.epochs),
                   "--surf_sub", "8000", "--gt_sub", "8000",
                   "--val_every", "5", "--val_subset", "20",
                   "--patience", "20",
                   "--cap_weight", str(w), "--out", outdir]
            if not h:
                cmd.append("--no_learn_heights")
            print("\n>>> " + " ".join(cmd))
            subprocess.run(cmd, check=True)

    # ---- evaluate ---------------------------------------------------------
    with open(os.path.join(a.data, f"test_N{a.N}.pkl"), "rb") as f:
        test = pickle.load(f)[:a.n_test]
    objs = [to_torch(s, a.m, dev, dt, seed=7000 + i)
            for i, s in enumerate(test)]

    rows = []
    # classical reference
    cap_c, body_c, all_c = [], [], []
    for obj in objs:
        p0 = zero_params(obj["R"].shape[0], a.m, device=dev, dtype=dt)
        S = build_surface(obj, p0, a.n_u)
        c, b, t = split_errors(S, obj["gt_pts"])
        cap_c.append(c); body_c.append(b); all_c.append(t)
    rows.append({"arm": "classical", "cap_weight": "-", "heights": "-",
                 "cap_err": np.mean(cap_c), "body_err": np.mean(body_c),
                 "all_err": np.mean(all_c)})

    for name, w, h in ARMS:
        ck = os.path.join(a.runs_dir, name, "best.pt")
        if not os.path.exists(ck):
            print(f"[missing] {ck} -- skipping {name}")
            continue
        net = ParamNet(learn_heights=h).to(device=dev, dtype=dt)
        net.load_state_dict(torch.load(ck, map_location=dev))
        net.eval()
        caps, bodies, alls = [], [], []
        with torch.no_grad():
            for obj in objs:
                feats = contour_features(obj["R"], obj["Z"], obj["RB"],
                                         obj["RC"], obj["Bh"], obj["Th"])
                S = build_surface(obj, net(feats), a.n_u)
                c, b, t = split_errors(S, obj["gt_pts"])
                caps.append(c); bodies.append(b); alls.append(t)
        rows.append({"arm": name, "cap_weight": w, "heights": h,
                     "cap_err": np.mean(caps), "body_err": np.mean(bodies),
                     "all_err": np.mean(alls)})

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader(); wri.writerows(rows)

    base = rows[0]
    print(f"\n{'arm':16s} {'w':>5s} {'heights':>8s} "
          f"{'CAP err':>10s} {'vs cls':>8s} "
          f"{'BODY err':>10s} {'vs cls':>8s} {'ALL err':>10s}")
    for r in rows:
        dc = (100 * (base["cap_err"] - r["cap_err"]) / base["cap_err"]
              if r["arm"] != "classical" else 0.0)
        db = (100 * (base["body_err"] - r["body_err"]) / base["body_err"]
              if r["arm"] != "classical" else 0.0)
        print(f"{r['arm']:16s} {str(r['cap_weight']):>5s} "
              f"{str(r['heights']):>8s} {r['cap_err']:>10.5f} "
              f"{dc:>7.1f}% {r['body_err']:>10.5f} {db:>7.1f}% "
              f"{r['all_err']:>10.5f}")
    print(f"\nwrote {a.out}")
    print("\nRead the CAP column, not ALL: caps are ~17% of points, so an "
          "aggregate number hides what these interventions do.")


if __name__ == "__main__":
    main()
