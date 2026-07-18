"""Train NSSR.

Usage:  python scripts/train_model.py --data data/synthetic --N 7 \
            --epochs 200 --out runs/exp1 [--free_residual]
"""
import sys, os, argparse, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from nssr.train import train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--N", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n_u", type=int, default=24)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--free_residual", action="store_true")
    ap.add_argument("--fp64", action="store_true")
    a = ap.parse_args()
    with open(os.path.join(a.data, f"train_N{a.N}.pkl"), "rb") as f:
        tr = pickle.load(f)
    with open(os.path.join(a.data, f"val_N{a.N}.pkl"), "rb") as f:
        va = pickle.load(f)
    train(tr, va, out_dir=a.out, epochs=a.epochs, lr=a.lr, m=a.m,
          n_u=a.n_u, lam_r=a.reg, free_residual=a.free_residual,
          dtype=torch.float64 if a.fp64 else torch.float32)

if __name__ == "__main__":
    main()
