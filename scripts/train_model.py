"""Train NSSR.

Usage:  python scripts/train_model.py --data data/synthetic --N 7 \
            --epochs 200 --out runs/exp1
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
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--c_bound", type=float, default=1.0,
                    help="bound on the learned log-multipliers: tangent "
                         "weights stay within e^{+-c_bound}. c=2 permits "
                         "7.4x amplification, which pushes narrow features "
                         "THROUGH the central axis (measured: the vase's "
                         "0.19-radius neck collapses to 0.002 at s_tau=2). "
                         "c=1 (2.7x) is safe on all three designer shapes. "
                         "MUST match the value a checkpoint was trained with.")
    ap.add_argument("--no_learn_heights", action="store_true",
                    help="OFF arm of the learnable-cap-height ablation: the "
                         "network never emits s_bh/s_th, so Bh/Th stay at "
                         "their classical (preprocessing) values")
    ap.add_argument("--cap_weight", type=float, default=1.0,
                    help="<1 down-weights CAP-patch points in the Chamfer "
                         "term (e.g. 0.25). Caps are ~17%% of surface points "
                         "but carry ~8x the per-point error, so they "
                         "dominate the gradient and the model shrinks their "
                         "radial bulge, flattening the poles.")
    ap.add_argument("--init_ckpt", default="",
                    help="initialize from this checkpoint (e.g. a "
                         "synthetic-trained model) instead of from the "
                         "classical solution -- use for fine-tuning")
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=0,
                    help="early stop after this many epochs w/o val gain (0=off)")
    ap.add_argument("--val_subset", type=int, default=0,
                    help="validate on first K objects only (0=all)")
    ap.add_argument("--eval_n_u", type=int, default=0,
                    help="n_u for validation (0=same as train n_u)")
    ap.add_argument("--n_u", type=int, default=24)
    ap.add_argument("--surf_sub", type=int, default=8000)
    ap.add_argument("--gt_sub", type=int, default=8000)
    a = ap.parse_args()
    with open(os.path.join(a.data, f"train_N{a.N}.pkl"), "rb") as f:
        tr = pickle.load(f)
    with open(os.path.join(a.data, f"val_N{a.N}.pkl"), "rb") as f:
        va = pickle.load(f)
    train(tr, va, out_dir=a.out, epochs=a.epochs, lr=a.lr, m=a.m,
          n_u=a.n_u, lam_r=a.reg,
          surf_sub=a.surf_sub, gt_sub=a.gt_sub,
          val_every=a.val_every, patience=a.patience,
          val_subset=a.val_subset, eval_n_u=(a.eval_n_u or None),
          init_ckpt=(a.init_ckpt or None), cap_weight=a.cap_weight,
          learn_heights=not a.no_learn_heights, c_bound=a.c_bound,
          dtype=torch.float64 if a.fp64 else torch.float32)

if __name__ == "__main__":
    main()
