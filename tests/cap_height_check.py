"""Verify learnable cap heights (s_bh / s_th).

Checks:
  1. s = 0 reproduces the classical Bh/Th EXACTLY (so the classical
     pipeline remains the model's initialization).
  2. The learned multiplier PRESERVES THE SIGN of the classical gap, for
     both outward caps (banana) and inward-dipping poles (apple, vase).
     A sign flip would invert the base/crown patch and send |Delta Z|
     through zero, which appears in tangent denominators.
  3. The cap's z-EXTENT now actually responds to the parameter -- the whole
     point, since s_fB/s_fC only change radial bulge.
  4. Bounds hold: the gap scales within e^{+-max_log} and never collapses
     to zero or runs away.

Run:  python tests/cap_height_check.py     (no torch needed: pure NumPy
      mirror of geometry.apply_cap_heights)
"""
import sys, os, types
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from nssr.preprocess import preprocess_designer
from nssr.geometry_np import hermite_surface_np, zero_params_np

MAX_LOG = 0.7


def apply_np(Z, Bh, Th, s_bh, s_th, max_log=MAX_LOG):
    """NumPy mirror of geometry.apply_cap_heights."""
    sb = max_log * np.tanh(s_bh)
    st = max_log * np.tanh(s_th)
    return (Z[0] - np.exp(sb) * (Z[0] - Bh),
            Z[-1] + np.exp(st) * (Th - Z[-1]))


def main():
    ok = True
    for ds in ("banana", "apple", "vase"):
        pre = preprocess_designer(ds, n1=25)
        R, Z, Bh, Th = pre["R"], pre["Z"], pre["Bh"], pre["Th"]
        g0, gN = Z[0] - Bh, Th - Z[-1]
        kind_b = "outward" if g0 > 0 else "INWARD (dimple)"
        kind_c = "outward" if gN > 0 else "INWARD (dimple)"
        print(f"\n=== {ds} ===")
        print(f"  classical base gap {g0:+.4f} ({kind_b}), "
              f"crown gap {gN:+.4f} ({kind_c})")

        # 1. s = 0 -> exact classical
        b0, t0 = apply_np(Z, Bh, Th, 0.0, 0.0)
        exact = (abs(b0 - Bh) < 1e-12) and (abs(t0 - Th) < 1e-12)
        ok &= exact
        print(f"  s=0 reproduces classical exactly: {exact}")

        # 2/4. sign preserved and bounded across the parameter range
        signs_ok, bounds_ok = True, True
        for s in (-6, -2, -1, 0, 1, 2, 6):
            b, t = apply_np(Z, Bh, Th, s, s)
            gb, gt = Z[0] - b, t - Z[-1]
            signs_ok &= (np.sign(gb) == np.sign(g0)) and (np.sign(gt) == np.sign(gN))
            r_b, r_t = abs(gb / g0), abs(gt / gN)
            bounds_ok &= (np.exp(-MAX_LOG) - 1e-9 <= r_b <= np.exp(MAX_LOG) + 1e-9)
            bounds_ok &= (np.exp(-MAX_LOG) - 1e-9 <= r_t <= np.exp(MAX_LOG) + 1e-9)
        ok &= signs_ok and bounds_ok
        print(f"  gap sign preserved over s in [-6, 6]: {signs_ok}")
        print(f"  gap scale within e^+-{MAX_LOG} ({np.exp(-MAX_LOG):.2f}x"
              f"-{np.exp(MAX_LOG):.2f}x): {bounds_ok}")

        # 3. the cap's z-extent actually responds
        exts = []
        N, m, _ = R.shape
        for s in (-3.0, 0.0, 3.0):
            b, t = apply_np(Z, Bh, Th, s, s)
            p = zero_params_np(N, m)
            S = hermite_surface_np(R, Z, pre["RB"], pre["RC"], b, t,
                                   params=p, n_u=24,
                                   closed_top=pre["closed_top"],
                                   base_circular=pre["base_circular"],
                                   crown_circular=pre["crown_circular"])
            cap = S[0]
            exts.append(cap[..., 2].max() - cap[..., 2].min())
        responds = (max(exts) - min(exts)) > 1e-6
        ok &= responds
        print(f"  base-cap z-extent at s=-3/0/+3: "
              f"{exts[0]:.4f} / {exts[1]:.4f} / {exts[2]:.4f}  "
              f"-> responds: {responds}")

    print("\nCAP HEIGHT CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
