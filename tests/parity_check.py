"""PARITY CHECK: NSSR classical path (params = 0) vs the reference CiSE
implementation, on the actual banana / apple / vase datasets.

Runs the reference pipeline (curve_goodman -> match_parameters ->
base_crown_pt/ht -> surf_tangent -> surf_pts) and then evaluates
nssr.geometry_np on the SAME R, Z, RB, RC, Bh, T.  Compares gR, gz, gRB,
gRC, fb, fc, FR, Fz elementwise.

No torch required.  Run:  python tests/parity_check.py
"""
import sys, os, types
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

# shapes_3D_data imports torch at module level; stub it if unavailable.
try:
    import torch  # noqa: F401
except ImportError:
    sys.modules["torch"] = types.ModuleType("torch")

from reference.shapes_3D_data import data_3d_shape                # noqa: E402
from reference.curves_python_loops import curve_goodman           # noqa: E402
from reference.surfaces_python_loops import (                     # noqa: E402
    t_no_pts, match_parameters, base_crown_pt, base_crown_ht,
    surf_tangent, surf_pts)
from nssr import geometry_np as G                                 # noqa: E402


def run_reference(ds, n1=25, n2=40, dt=np.float64):
    I, Z, Null_Hts = data_3d_shape(ds, dtype=dt)
    tot_pts, seg_pts = t_no_pts(I, n1)
    N = len(seg_pts); M = 4; step = tot_pts // M
    r = np.stack([curve_goodman(I[k], seg_pts[k]) for k in range(len(I))])
    R = match_parameters(r, N, tot_pts, M)               # (N, tot_pts+1, 2)
    RB, RC = base_crown_pt(R, N, tot_pts, M, step)
    if ds == "apple":
        B, T = Null_Hts[0], Null_Hts[1]
        bt = ct = "n"
    else:
        B, T = base_crown_ht(R, N, tot_pts, M, step, Z, Null_Hts, dtype=dt)
        bt = ct = "y"
    gR, gz, gRB, gRC, fb, fc = surf_tangent(
        R, N, tot_pts, Z, Null_Hts, RB, RC, B, T, bt, ct, dtype=dt)
    FR, Fz = surf_pts(R, N, tot_pts, Z, RB, RC, B, T, gRB, gRC, fb, fc,
                      gR, gz, bt, ct, n2, dtype=dt)
    return dict(R=R, Z=Z, RB=RB, RC=RC, B=B, T=T, N=N, gR=gR, gz=gz,
                gRB=gRB, gRC=gRC, fb=fb, fc=fc, FR=FR, Fz=Fz,
                circular=(bt == "y"), n2=n2)


def compare(name, mine, ref, tol, report):
    err = np.max(np.abs(np.asarray(mine) - np.asarray(ref)))
    ok = err <= tol
    report.append((name, err, ok))
    return ok


def check_shape(ds):
    ref = run_reference(ds)
    R, Z = ref["R"], np.asarray(ref["Z"], dtype=np.float64)
    RB, RC = np.asarray(ref["RB"]), np.asarray(ref["RC"])
    Bh, Th = float(ref["B"]), float(ref["T"])
    N = ref["N"]

    gR, gZ = G.tangent_field_np(R, Z, RB, RC, Bh, Th)
    fB, fC = G.boundary_scalings_np(R, RB, RC, Z, Bh, Th, gR)
    S = G.hermite_surface_np(R, Z, RB, RC, Bh, Th, n_u=ref["n2"] + 1,
                             base_circular=ref["circular"],
                             crown_circular=ref["circular"])
    rep, ok = [], True
    ok &= compare("gR", gR, ref["gR"], 1e-9, rep)
    ok &= compare("gz", gZ, ref["gz"], 1e-9, rep)
    ok &= compare("fb", fB, ref["fb"], 1e-9, rep)
    ok &= compare("fc", fC, ref["fc"], 1e-9, rep)
    if ref["circular"]:
        gRB = G.boundary_directions_np(R, Z, RB, RC, Bh, Th, at_base=True)
        gRC = G.boundary_directions_np(R, Z, RB, RC, Bh, Th, at_base=False)
        ok &= compare("gRB", gRB, ref["gRB"], 1e-9, rep)
        ok &= compare("gRC", gRC, ref["gRC"], 1e-9, rep)
    # reference FR/Fz layout: (N+1, M, n2+1, [2]); mine: (N+1, n2+1, M, 3)
    FR_mine = S[..., :2].transpose(0, 2, 1, 3)
    Fz_mine = S[..., 2].transpose(0, 2, 1)
    ok &= compare("FR", FR_mine, ref["FR"], 1e-8, rep)
    ok &= compare("Fz", Fz_mine, ref["Fz"], 1e-8, rep)

    print(f"\n[{ds}]  (N={N}, M={R.shape[1]}, circular caps: "
          f"{ref['circular']})")
    for name, err, o in rep:
        print(f"   {name:4s} max|diff| = {err:9.2e}   "
              f"{'OK' if o else '** MISMATCH **'}")
    return ok


def main():
    ok = all(check_shape(ds) for ds in ("banana", "apple", "vase"))
    print("\nPARITY CHECK:", "PASS — NSSR classical path exactly matches "
          "the reference implementation" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
