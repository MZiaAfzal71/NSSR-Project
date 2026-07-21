"""Multi-GPU training for Kaggle's 2x T4 (uses BOTH GPUs).

Object-level data parallelism: each epoch's objects are sharded across GPUs,
every rank computes gradients on its shard, gradients are all-reduced
(averaged), and each rank steps an identical copy of the model.  This suits
NSSR perfectly because objects are independent and N varies per object
(so batching is per-object anyway).

Why not split one object across GPUs: a single object is tiny; the win is
running twice as many objects per wall-clock second, which this does.

Usage (inside a Kaggle notebook cell, 2x T4 selected):
    !python scripts/train_multigpu.py --data data/synthetic --N 7 \
        --epochs 200 --m 256 --out runs/exp1

Falls back to single-GPU automatically if only one device is visible.
"""
import sys, os, argparse, pickle, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nssr.train import to_torch, forward_object
from nssr.geometry import tangent_field
from nssr.networks import ParamNet
from nssr.losses import total_loss
from nssr.metrics import evaluate_surface, c1_diagnostic


def _run(rank, world, args, ret):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dtype = torch.float64 if args.fp64 else torch.float32
    torch.manual_seed(args.seed)

    with open(os.path.join(args.data, f"train_N{args.N}.pkl"), "rb") as f:
        train_s = pickle.load(f)
    with open(os.path.join(args.data, f"val_N{args.N}.pkl"), "rb") as f:
        val_s = pickle.load(f)

    if rank == 0:
        print(f"[{world} GPU] preprocessing "
              f"{len(train_s)} train / {len(val_s)} val objects ...")
    train_objs = [to_torch(s, args.m, dev, dtype, seed=i)
                  for i, s in enumerate(train_s)]
    # validation only on rank 0 (cheap, avoids duplicate work)
    val_objs = ([to_torch(s, args.m, dev, dtype, seed=10_000 + i)
                 for i, s in enumerate(val_s)] if rank == 0 else [])

    net = ParamNet(free_residual=args.free_residual).to(dev, dtype)
    # broadcast rank-0 init to all ranks so models start identical
    for p in net.parameters():
        dist.broadcast(p.data, src=0)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        logf = open(os.path.join(args.out, "log.csv"), "w", newline="")
        logger = csv.writer(logf)
        logger.writerow(["epoch", "train_loss", "train_chamfer",
                         "val_chamfer_l2", "val_hausdorff",
                         "val_normal", "c1_min", "secs"])
        best = float("inf")

    idx_all = np.arange(len(train_objs))
    for epoch in range(args.epochs + 1):
        t0 = time.time()
        # deterministic shuffle shared across ranks, then shard
        g = np.random.default_rng(1234 + epoch)
        perm = g.permutation(idx_all)
        my = perm[rank::world]                     # this rank's objects
        net.train()
        tot = torch.zeros(1, device=dev)
        totc = torch.zeros(1, device=dev)
        nb = torch.zeros(1, device=dev)
        opt.zero_grad()
        for step, k in enumerate(my):
            obj = train_objs[k]
            _, pts, nrms, params = forward_object(net, obj, n_u=args.n_u)
            loss, parts = total_loss(pts, nrms, obj["gt_pts"],
                                     obj["gt_normals"], params,
                                     args.lam_n, args.reg, args.lam_s,
                                     surf_sub=args.surf_sub,
                                     gt_sub=args.gt_sub)
            if epoch > 0:
                (loss / max(len(my), 1)).backward()
            tot += loss.detach(); totc += parts["chamfer"]; nb += 1

        if epoch > 0:
            # average gradients across ranks (DDP-style manual all-reduce)
            for p in net.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()

        for t in (tot, totc, nb):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        do_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
        if rank == 0 and do_val:
            net.eval()
            e_nu = args.eval_n_u or args.n_u
            vobjs = val_objs if args.val_subset <= 0 else val_objs[:args.val_subset]
            vs, hs, ns, c1s = [], [], [], []
            with torch.no_grad():
                for obj in vobjs:
                    _, pts, nrms, params = forward_object(net, obj, n_u=e_nu)
                    mets = evaluate_surface(pts, obj["gt_pts"], nrms,
                                            obj["gt_normals"])
                    gR, gZ = tangent_field(obj["R"], obj["Z"], obj["RB"],
                                           obj["RC"], obj["Bh"], obj["Th"],
                                           params,
                                           closed_top=obj.get("closed_top", True))
                    vs.append(mets["chamfer_l2"]); hs.append(mets["hausdorff"])
                    ns.append(mets.get("normal_consistency", float("nan")))
                    c1s.append(c1_diagnostic(gR, gZ)["global_min"])
            row = [epoch, (tot / nb).item(), (totc / nb).item(),
                   float(np.mean(vs)), float(np.mean(hs)),
                   float(np.nanmean(ns)), float(np.min(c1s)), time.time() - t0]
            logger.writerow(row); logf.flush()
            tag = "  <-- CLASSICAL BASELINE" if epoch == 0 else ""
            print(f"ep {epoch:3d} | loss {row[1]:.5f} | val CD {row[3]:.6f} "
                  f"| val H {row[4]:.4f} | NC {row[5]:.3f} "
                  f"| C1min {row[6]:.3f} | {row[7]:.1f}s{tag}")
            if epoch > 0 and row[3] < best:
                best = row[3]
                torch.save(net.state_dict(),
                           os.path.join(args.out, "best.pt"))
        elif rank == 0:
            print(f"ep {epoch:3d} | loss {(tot/nb).item():.5f} | "
                  f"{time.time()-t0:.1f}s (no val)")
        dist.barrier()

    if rank == 0:
        logf.close()
        ret["best"] = best
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--N", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n_u", type=int, default=24)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--lam_n", type=float, default=0.1)
    ap.add_argument("--lam_s", type=float, default=1e-3)
    ap.add_argument("--surf_sub", type=int, default=8000)
    ap.add_argument("--gt_sub", type=int, default=8000)
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--free_residual", action="store_true")
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--val_subset", type=int, default=0)
    ap.add_argument("--eval_n_u", type=int, default=0)
    args = ap.parse_args()

    world = torch.cuda.device_count()
    if world < 2:
        print(f"Only {world} GPU visible -> single-GPU training.")
        from nssr.train import train
        with open(os.path.join(args.data, f"train_N{args.N}.pkl"), "rb") as f:
            tr = pickle.load(f)
        with open(os.path.join(args.data, f"val_N{args.N}.pkl"), "rb") as f:
            va = pickle.load(f)
        train(tr, va, out_dir=args.out, epochs=args.epochs, lr=args.lr,
              m=args.m, n_u=args.n_u, lam_r=args.reg, lam_n=args.lam_n,
              lam_s=args.lam_s, free_residual=args.free_residual,
              surf_sub=args.surf_sub, gt_sub=args.gt_sub,
              dtype=torch.float64 if args.fp64 else torch.float32)
        return

    print(f"Spawning {world} processes (one per T4).")
    mgr = mp.Manager(); ret = mgr.dict()
    mp.spawn(_run, args=(world, args, ret), nprocs=world, join=True)
    print("done. best val chamfer:", ret.get("best"))


if __name__ == "__main__":
    main()
