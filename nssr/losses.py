"""Losses (METHOD.md sec. 5) and shared distance utilities."""
from __future__ import annotations
import torch


def _pairwise_min(a: torch.Tensor, b: torch.Tensor, chunk=4096):
    """For each point in a, squared distance to nearest point in b.
    Chunked cdist to bound memory.  Returns (dists (Na,), idx (Na,))."""
    mins, idxs = [], []
    for i in range(0, a.shape[0], chunk):
        d = torch.cdist(a[i:i + chunk], b)            # (c, Nb)
        mn, ix = d.min(dim=1)
        mins.append(mn ** 2); idxs.append(ix)
    return torch.cat(mins), torch.cat(idxs)


def chamfer(pred: torch.Tensor, gt: torch.Tensor):
    """Two-sided mean squared Chamfer.  pred (P,3), gt (Q,3)."""
    d_pg, _ = _pairwise_min(pred, gt)
    d_gp, _ = _pairwise_min(gt, pred)
    return d_pg.mean() + d_gp.mean()


def normal_loss(pred_pts, pred_normals, gt_pts, gt_normals):
    """1 - |cos| between GT normals and normals of nearest predicted point."""
    _, idx = _pairwise_min(gt_pts, pred_pts)
    cos = (gt_normals * pred_normals[idx]).sum(-1).abs()
    return (1.0 - cos).mean()


def total_loss(pred_pts, pred_normals, gt_pts, gt_normals, params,
               lam_n=0.1, lam_r=1e-3, lam_s=1e-3):
    from .networks import param_l2, param_smoothness
    l_cd = chamfer(pred_pts, gt_pts)
    l_n = normal_loss(pred_pts, pred_normals, gt_pts, gt_normals) \
        if gt_normals is not None else torch.zeros((), device=pred_pts.device)
    l_r = param_l2(params)
    l_s = param_smoothness(params)
    loss = l_cd + lam_n * l_n + lam_r * l_r + lam_s * l_s
    return loss, {"chamfer": l_cd.item(), "normal": float(l_n),
                  "reg": float(l_r), "smooth": float(l_s)}
