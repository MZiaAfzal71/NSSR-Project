"""Training loop.  Batch size is 1 object (N varies); gradient accumulation
over `accum` objects emulates larger batches.  Epoch 0 logs the classical
(theta = 0) baseline automatically, since the net is zero-initialized.
"""
from __future__ import annotations
import csv, os, time
import numpy as np
import torch

from .preprocess import preprocess_object
from .geometry import hermite_surface, tangent_field, surface_points, surface_normals
from .networks import ParamNet, contour_features
from .losses import total_loss
from .metrics import evaluate_surface, c1_diagnostic


def to_torch(sample, m=256, device="cpu", dtype=torch.float32,
             gt_subsample=20000, seed=0):
    pre = preprocess_object(sample["contours"], sample["Z"], m=m)
    T = lambda x: torch.as_tensor(np.asarray(x), device=device, dtype=dtype)
    g = np.random.default_rng(seed)
    q = sample["gt_pts"].shape[0]
    idx = g.choice(q, size=min(gt_subsample, q), replace=False)
    # GT must be normalized with the SAME transform as the contours
    nrm = pre["norm"]
    gt = (sample["gt_pts"][idx] - np.array([*nrm["center_xy"], nrm["zmid"]])) \
        / nrm["scale"]
    return {"R": T(pre["R"]), "Z": T(pre["Z"]), "RB": T(pre["RB"]),
            "RC": T(pre["RC"]), "Bh": T(pre["Bh"]), "Th": T(pre["Th"]),
            "gt_pts": T(gt), "gt_normals": T(sample["gt_normals"][idx]),
            "base_circular": pre.get("base_circular", True),
            "crown_circular": pre.get("crown_circular", True),
            "closed_top": pre.get("closed_top", True)}


def forward_object(net, obj, n_u=24):
    feats = contour_features(obj["R"], obj["Z"], obj["RB"], obj["RC"],
                             obj["Bh"], obj["Th"])
    params = net(feats)
    S = hermite_surface(obj["R"], obj["Z"], obj["RB"], obj["RC"],
                        obj["Bh"], obj["Th"], params, n_u=n_u,
                        closed_top=obj.get("closed_top", True),
                        base_circular=obj.get("base_circular", True),
                        crown_circular=obj.get("crown_circular", True))
    pts = surface_points(S)
    nrms = surface_normals(S).reshape(-1, 3)
    return S, pts, nrms, params


def train(samples, val_samples, out_dir="runs/exp1", epochs=200, lr=1e-3,
          m=256, n_u=24, device=None, dtype=torch.float32,
          lam_n=0.1, lam_r=1e-3, lam_s=1e-3, accum=8,
          free_residual=False, seed=0):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    print("Preprocessing ...")
    train_objs = [to_torch(s, m, device, dtype, seed=i)
                  for i, s in enumerate(samples)]
    val_objs = [to_torch(s, m, device, dtype, seed=10_000 + i)
                for i, s in enumerate(val_samples)]

    net = ParamNet(free_residual=free_residual).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    logf = open(os.path.join(out_dir, "log.csv"), "w", newline="")
    logger = csv.writer(logf)
    logger.writerow(["epoch", "train_loss", "train_chamfer",
                     "val_chamfer_l2", "val_hausdorff", "c1_min", "secs"])

    best = float("inf")
    for epoch in range(epochs + 1):                     # epoch 0 = classical
        t0 = time.time()
        net.train()
        tot, totc, nb = 0.0, 0.0, 0
        perm = np.random.permutation(len(train_objs))
        opt.zero_grad()
        for step, k in enumerate(perm):
            obj = train_objs[k]
            _, pts, nrms, params = forward_object(net, obj, n_u=n_u)
            loss, parts = total_loss(pts, nrms, obj["gt_pts"],
                                     obj["gt_normals"], params,
                                     lam_n, lam_r, lam_s)
            if epoch > 0:                                # epoch 0: eval only
                (loss / accum).backward()
                if (step + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step(); opt.zero_grad()
            tot += loss.item(); totc += parts["chamfer"]; nb += 1
        if epoch > 0:
            sched.step()

        # validation
        net.eval()
        vs, hs, c1s = [], [], []
        with torch.no_grad():
            for obj in val_objs:
                _, pts, nrms, params = forward_object(net, obj, n_u=n_u)
                mets = evaluate_surface(pts, obj["gt_pts"], nrms,
                                        obj["gt_normals"])
                gR, gZ = tangent_field(obj["R"], obj["Z"], obj["RB"],
                                       obj["RC"], obj["Bh"], obj["Th"], params)
                vs.append(mets["chamfer_l2"]); hs.append(mets["hausdorff"])
                c1s.append(c1_diagnostic(gR, gZ)["global_min"])
        row = [epoch, tot / nb, totc / nb, float(np.mean(vs)),
               float(np.mean(hs)), float(np.min(c1s)), time.time() - t0]
        logger.writerow(row); logf.flush()
        tag = "  <-- CLASSICAL BASELINE" if epoch == 0 else ""
        print(f"ep {epoch:3d} | loss {row[1]:.5f} | val CD {row[3]:.6f} "
              f"| val H {row[4]:.4f} | C1min {row[5]:.3f}{tag}")
        if epoch > 0 and row[3] < best:
            best = row[3]
            torch.save(net.state_dict(), os.path.join(out_dir, "best.pt"))
    logf.close()
    return net
