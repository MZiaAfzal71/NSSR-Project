"""NumPy mirror of the classical pipeline (params = 0) — verifies the math
without needing PyTorch.  Reconstructs (a) a unit sphere from 7 circles and
(b) a random synthetic object, and reports Chamfer error against dense GT.

Run:  python tests/numpy_sanity_check.py
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nssr.preprocess import preprocess_object            # noqa: E402
from nssr.synthetic import make_sample                   # noqa: E402

EPS = 1e-12


# ---- NumPy transliteration of nssr/geometry.py (classical path) --------
def tangent_field_np(R, Z, RB, RC, Bh, Th, closed_top=True):
    N, m, _ = R.shape
    dR = np.empty((N + 1, m, 2)); dZ = np.empty(N + 1)
    dR[1:N] = R[1:] - R[:-1]; dR[0] = R[0] - RB
    dR[N] = (RC - R[-1]) if closed_top else dR[N - 1]
    dZ[1:N] = Z[1:] - Z[:-1]; dZ[0] = Z[0] - Bh
    dZ[N] = (Th - Z[-1]) if closed_top else dZ[N - 1]
    nrm = np.sqrt((dR ** 2).sum(-1) + EPS)
    a = nrm[1:].copy(); b = nrm[:-1].copy()
    a[0] *= 2.0
    if closed_top:
        b[N - 1] *= 2.0
    denom = a * abs(dZ[:-1])[:, None] + b * abs(dZ[1:])[:, None] + EPS
    gR = (a[..., None] * dR[:-1] + b[..., None] * dR[1:]) / denom[..., None]
    gZ = (a * dZ[:-1][:, None] + b * dZ[1:][:, None]) / denom
    return gR, gZ


def hermite_surface_np(R, Z, RB, RC, Bh, Th, n_u=32, closed_top=True):
    N, m, _ = R.shape
    u = np.linspace(0, 1, n_u)
    L0 = 1 - 3 * u**2 + 2 * u**3; L1 = 3 * u**2 - 2 * u**3
    H0 = u - 2 * u**2 + u**3;     H1 = -u**2 + u**3
    gR, gZ = tangent_field_np(R, Z, RB, RC, Bh, Th, closed_top)
    P = []
    # base cap, circular case (Eqs. 31-32) without the fB*gRB radial term
    # (direction term verified in torch; sphere test uses radial symmetry
    # so we include it via the analytic radial direction):
    dZ0 = abs(Z[0] - Bh)
    rB = R[0] - RB
    nB = np.sqrt((rB ** 2).sum(-1) + EPS)
    gB = np.sqrt((gR[0] ** 2).sum(-1) + EPS)
    fB = np.sqrt(2) * nB * np.sqrt(1 + dZ0 * gB / nB)
    eB = rB / nB[..., None]                       # radial boundary direction
    FR = (L0[:, None, None] * RB + L1[:, None, None] * R[0]
          + H0[:, None, None] * (fB[..., None] * eB)
          + 2 * dZ0 * H1[:, None, None] * gR[0])
    Fz = ((1 - u**2) * Bh + u**2 * Z[0])[:, None] * np.ones((1, m))
    P.append(np.concatenate([FR, Fz[..., None]], -1))
    # interior
    for i in range(1, N):
        dZi = abs(Z[i] - Z[i - 1])
        FR = (L0[:, None, None] * R[i - 1] + L1[:, None, None] * R[i]
              + dZi * (H0[:, None, None] * gR[i - 1]
                       + H1[:, None, None] * gR[i]))
        Fz = (L0[:, None] * Z[i - 1] + L1[:, None] * Z[i]
              + dZi * (H0[:, None] * gZ[i - 1] + H1[:, None] * gZ[i]))
        P.append(np.concatenate([FR, Fz[..., None]], -1))
    if closed_top:
        dZN = abs(Th - Z[-1])
        rC = R[-1] - RC
        nC = np.sqrt((rC ** 2).sum(-1) + EPS)
        gC = np.sqrt((gR[-1] ** 2).sum(-1) + EPS)
        fC = np.sqrt(2) * nC * np.sqrt(1 + dZN * gC / nC)
        eC = rC / nC[..., None]
        FR = (L0[:, None, None] * R[-1] + L1[:, None, None] * RC
              + H1[:, None, None] * (fC[..., None] * eC)
              + 2 * dZN * H0[:, None, None] * gR[-1])
        Fz = ((1 - u)**2 * Z[-1] + u * (2 - u) * Th)[:, None] * np.ones((1, m))
        P.append(np.concatenate([FR, Fz[..., None]], -1))
    return np.stack(P)                                   # (P, n_u, m, 3)


def _oneside(A, B, chunk=512):
    """Mean over A of distance to nearest point in FULL set B (chunked)."""
    mins = []
    for i in range(0, len(A), chunk):
        d = np.linalg.norm(A[i:i+chunk, None, :] - B[None, :, :], axis=-1)
        mins.append(d.min(1))
    return np.concatenate(mins).mean()


def chamfer_np(A, B, sub=6000, rng=None):
    """Accuracy (pred->full GT) + coverage (GT->full pred); query sides
    subsampled, target sides FULL so the value measures surface error,
    not sampling density."""
    rng = rng or np.random.default_rng(0)
    Aq = A[rng.choice(len(A), min(sub, len(A)), replace=False)]
    Bq = B[rng.choice(len(B), min(sub, len(B)), replace=False)]
    return 0.5 * (_oneside(Aq, B) + _oneside(Bq, A))


def run_case(name, contours, Z, gt_pts, m=128):
    pre = preprocess_object(contours, Z, m=m)
    nrm = pre["norm"]
    gt = (gt_pts - np.array([*nrm["center_xy"], nrm["zmid"]])) / nrm["scale"]
    S = hermite_surface_np(pre["R"], pre["Z"], pre["RB"], pre["RC"],
                           pre["Bh"], pre["Th"])
    pred = S.reshape(-1, 3)
    cd = chamfer_np(pred, gt)
    diag = np.sqrt(gt.max(0) - gt.min(0)).sum()  # rough scale ref
    print(f"[{name:9s}] contours N={len(contours)}, m={m} | "
          f"surface pts {pred.shape[0]:6d} | chamfer-L1 (two-sided) = {cd:.4f}")
    return cd


def main():
    ok = True
    # --- sphere from circles -------------------------------------------
    N = 7
    t = np.linspace(0.12, 0.88, N)
    Z = np.cos(np.pi * (1 - t))                     # z in (-1, 1)
    contours = []
    for z in Z:
        r = np.sqrt(1 - z**2)
        th = np.linspace(0, 2*np.pi, 90, endpoint=False)
        contours.append(np.stack([r*np.cos(th), r*np.sin(th)], 1))
    th = np.linspace(0, 2*np.pi, 120, endpoint=False)
    ph = np.linspace(0.02, np.pi - 0.02, 120)
    TH, PH = np.meshgrid(th, ph)
    gt = np.stack([np.sin(PH)*np.cos(TH), np.sin(PH)*np.sin(TH),
                   np.cos(PH)], -1).reshape(-1, 3)
    cd = run_case("sphere", contours, Z, gt)
    ok &= cd < 0.02

    # --- random synthetic objects --------------------------------------
    for seed in (1, 2, 3):
        s = make_sample(seed, N=9)
        cd = run_case(f"synth-{seed}", s["contours"], s["Z"], s["gt_pts"])
        ok &= cd < 0.05
    print("\nSANITY CHECK:", "PASS" if ok else "FAIL",
          "(classical pipeline reconstructs test objects with small error)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
