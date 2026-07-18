"""Sanity check of the classical pipeline (nssr.geometry_np) on analytic
test objects — verifies the corrected, reference-parity math end to end
without torch.  Run:  python tests/numpy_sanity_check.py
(See tests/parity_check.py for elementwise parity vs the reference code.)
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nssr.preprocess import preprocess_object            # noqa: E402
from nssr.synthetic import make_sample                   # noqa: E402
from nssr.geometry_np import hermite_surface_np          # noqa: E402


def _oneside(A, B, chunk=512):
    mins = []
    for i in range(0, len(A), chunk):
        d = np.linalg.norm(A[i:i+chunk, None, :] - B[None, :, :], axis=-1)
        mins.append(d.min(1))
    return np.concatenate(mins).mean()


def chamfer_np(A, B, sub=6000, rng=None):
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
    print(f"[{name:9s}] N={len(contours)}, m={m} | surface pts "
          f"{pred.shape[0]:6d} | chamfer-L1 = {cd:.4f}")
    return cd


def main():
    ok = True
    N = 7
    t = np.linspace(0.12, 0.88, N)
    Z = np.cos(np.pi * (1 - t))
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
    ok &= run_case("sphere", contours, Z, gt) < 0.02
    for seed in (1, 2, 3):
        s = make_sample(seed, N=9)
        ok &= run_case(f"synth-{seed}", s["contours"], s["Z"],
                       s["gt_pts"]) < 0.05
    print("\nSANITY CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
