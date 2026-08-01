"""Losses (METHOD.md sec. 5) and shared distance utilities.

Memory model: nearest-neighbour search is double-chunked (both the query
and the target sides are tiled), so peak memory is bounded by
chunk_q * chunk_t regardless of cloud size.  This is what lets large m
(256+) fit in 15 GB.  For the training loss we additionally subsample the
predicted surface (surf_sub) -- the gradient is unbiased and memory drops
further.
"""
from __future__ import annotations
import torch


def _nn_sqdist(a, b, chunk_q=4096, chunk_t=16384):
    """For each point in a: squared distance to nearest point in b, and
    that neighbour's index.  Double-chunked -> peak mem ~ chunk_q*chunk_t."""
    Na = a.shape[0]
    out_d = torch.empty(Na, device=a.device, dtype=a.dtype)
    out_i = torch.empty(Na, device=a.device, dtype=torch.long)
    for i in range(0, Na, chunk_q):
        aq = a[i:i + chunk_q]
        best_d = torch.full((aq.shape[0],), float("inf"),
                            device=a.device, dtype=a.dtype)
        best_i = torch.zeros(aq.shape[0], device=a.device, dtype=torch.long)
        for j in range(0, b.shape[0], chunk_t):
            bt = b[j:j + chunk_t]
            d = torch.cdist(aq, bt)
            dmin, imin = d.min(dim=1)
            upd = dmin < best_d
            best_d = torch.where(upd, dmin, best_d)
            best_i = torch.where(upd, imin + j, best_i)
        out_d[i:i + chunk_q] = best_d ** 2
        out_i[i:i + chunk_q] = best_i
    return out_d, out_i


def chamfer(pred, gt, surf_sub=None, gt_sub=None):
    """Two-sided mean squared Chamfer.  Optionally subsample either side
    (unbiased estimate) to cap memory / cost during training."""
    if surf_sub is not None and pred.shape[0] > surf_sub:
        idx = torch.randperm(pred.shape[0], device=pred.device)[:surf_sub]
        pred = pred[idx]
    if gt_sub is not None and gt.shape[0] > gt_sub:
        idx = torch.randperm(gt.shape[0], device=gt.device)[:gt_sub]
        gt = gt[idx]
    d_pg, _ = _nn_sqdist(pred, gt)
    d_gp, _ = _nn_sqdist(gt, pred)
    return d_pg.mean() + d_gp.mean()


def normal_loss(pred_pts, pred_normals, gt_pts, gt_normals, gt_sub=None):
    if gt_sub is not None and gt_pts.shape[0] > gt_sub:
        idx = torch.randperm(gt_pts.shape[0], device=gt_pts.device)[:gt_sub]
        gt_pts, gt_normals = gt_pts[idx], gt_normals[idx]
    _, idx = _nn_sqdist(gt_pts, pred_pts)
    cos = (gt_normals * pred_normals[idx]).sum(-1).abs()
    return (1.0 - cos).mean()


def chamfer_weighted(pred, gt, pred_w=None, surf_sub=None, gt_sub=None):
    """Two-sided Chamfer with an optional PER-PREDICTED-POINT weight.

    Both directions are weighted, which is the point:
      pred->gt : each predicted point's distance is scaled by its weight.
      gt->pred : each GT point is scaled by the weight of the predicted
                 point it matched to.

    Weighting only the pred->gt side (as an earlier version did, by
    subsampling cap points) does almost nothing, because the gt->pred term
    still requires a predicted point near every GT pole sample -- and that
    is the term that actually penalizes a cap for being the wrong size.
    Subsampling one side is in fact counterproductive: it removes predicted
    cap points while keeping the GT pole points they must cover.
    """
    if surf_sub is not None and pred.shape[0] > surf_sub:
        idx = torch.randperm(pred.shape[0], device=pred.device)[:surf_sub]
        pred = pred[idx]
        if pred_w is not None:
            pred_w = pred_w[idx]
    if gt_sub is not None and gt.shape[0] > gt_sub:
        idx = torch.randperm(gt.shape[0], device=gt.device)[:gt_sub]
        gt = gt[idx]
    d_pg, _ = _nn_sqdist(pred, gt)
    d_gp, idx_gp = _nn_sqdist(gt, pred)
    if pred_w is None:
        return d_pg.mean() + d_gp.mean()
    w_pg = pred_w
    w_gp = pred_w[idx_gp]
    return ((d_pg * w_pg).sum() / w_pg.sum().clamp_min(1e-8)
            + (d_gp * w_gp).sum() / w_gp.sum().clamp_min(1e-8))


def total_loss(pred_pts, pred_normals, gt_pts, gt_normals, params,
               lam_n=0.1, lam_r=1e-3, lam_s=1e-3,
               surf_sub=20000, gt_sub=20000, cap_mask=None, cap_weight=1.0):
    """cap_mask: bool tensor over pred_pts marking CAP-patch points, given
    weight `cap_weight` (<1) in BOTH Chamfer directions.

    Rationale: cap patches converge to a single point, so their sampling
    density and per-point error are not comparable to the body (measured:
    caps are ~17% of surface points but carry ~8x the per-point error).
    Chamfer averages over points, so the poles dominate the gradient."""
    from .networks import param_l2, param_smoothness
    w = None
    if cap_mask is not None and cap_weight != 1.0:
        w = torch.ones(pred_pts.shape[0], device=pred_pts.device,
                       dtype=pred_pts.dtype)
        w = torch.where(cap_mask, torch.full_like(w, cap_weight), w)
    l_cd = chamfer_weighted(pred_pts, gt_pts, pred_w=w,
                            surf_sub=surf_sub, gt_sub=gt_sub)
    if gt_normals is not None:
        l_n = normal_loss(pred_pts, pred_normals, gt_pts, gt_normals,
                          gt_sub=gt_sub)
    else:
        l_n = torch.zeros((), device=pred_pts.device)
    l_r = param_l2(params)
    l_s = param_smoothness(params)
    loss = l_cd + lam_n * l_n + lam_r * l_r + lam_s * l_s
    return loss, {"chamfer": l_cd.detach().item(),
                  "normal": float(l_n.detach()) if torch.is_tensor(l_n) else float(l_n),
                  "reg": float(l_r.detach()), "smooth": float(l_s.detach())}
