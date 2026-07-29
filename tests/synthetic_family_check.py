"""Regression test for the three synthetic geometric families (standard /
dimpled / open_top -- see nssr/synthetic.py).  Verifies:
  1. family mix roughly matches FAMILY_PROBS,
  2. dimpled objects never violate true-endpoint monotonicity (which
     default_null_hts and cap_heights rely on) while still regularly
     producing a genuine INTERIOR Z reversal (the apple-like stress case),
  3. open_top objects carry closed_top=False end-to-end,
  4. every family survives preprocess_object + hermite_surface_np with
     no NaN/Inf and the expected patch count (confirms closed_top=False
     genuinely removes the crown patch, not just a rendering choice).

Run:  python tests/synthetic_family_check.py
"""
import sys, os, types
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from nssr.synthetic import make_sample, FAMILY_PROBS
from nssr.preprocess import preprocess_object
from nssr.geometry_np import hermite_surface_np, zero_params_np

N_SAMPLES = 600


def main():
    ok = True
    counts = {k: 0 for k in FAMILY_PROBS}
    endpoint_violations = 0
    dimple_reversals = 0
    n_dimpled = 0
    geometry_failures = []

    for seed in range(N_SAMPLES):
        s = make_sample(seed, N=9)
        counts[s["family"]] += 1

        if s["family"] == "dimpled":
            n_dimpled += 1
            d = np.diff(s["Z"])
            if d[0] <= 0 or d[-1] <= 0:
                endpoint_violations += 1
            elif len(d) > 2 and (d[1:-1] < 0).any():
                dimple_reversals += 1

        if s["family"] == "open_top" and s["closed_top"] is not False:
            geometry_failures.append((seed, "open_top but closed_top != False"))
            continue

        try:
            pre = preprocess_object(s["contours"], s["Z"], m=64,
                                    base_circular=s["base_circular"],
                                    crown_circular=s["crown_circular"],
                                    closed_top=s["closed_top"])
            p0 = zero_params_np(pre["R"].shape[0], pre["R"].shape[1])
            S = hermite_surface_np(pre["R"], pre["Z"], pre["RB"], pre["RC"],
                                   pre["Bh"], pre["Th"], params=p0, n_u=8,
                                   closed_top=pre["closed_top"],
                                   base_circular=pre["base_circular"],
                                   crown_circular=pre["crown_circular"])
            if np.isnan(S).any() or np.isinf(S).any():
                geometry_failures.append((seed, "NaN/Inf in surface"))
                continue
            expected = (pre["R"].shape[0] - 1) + 1 + (1 if pre["closed_top"] else 0)
            if S.shape[0] != expected:
                geometry_failures.append(
                    (seed, f"patch count {S.shape[0]} != expected {expected}"))
        except Exception as e:                          # noqa: BLE001
            geometry_failures.append((seed, repr(e)))

    print(f"family counts (n={N_SAMPLES}): {counts}  "
          f"(target ratios: {FAMILY_PROBS})")
    for fam, target in FAMILY_PROBS.items():
        frac = counts[fam] / N_SAMPLES
        close = abs(frac - target) < 0.06
        ok &= close
        print(f"  {fam:10s} {frac:.3f} vs target {target:.3f}  "
              f"{'OK' if close else '** OFF TARGET **'}")

    print(f"\ndimpled endpoint-monotonicity violations: {endpoint_violations} "
          f"/ {n_dimpled}  {'OK' if endpoint_violations == 0 else '** FAIL **'}")
    ok &= (endpoint_violations == 0)
    print(f"dimpled samples with genuine interior Z reversal: "
          f"{dimple_reversals} / {n_dimpled}  "
          f"{'OK' if dimple_reversals > n_dimpled * 0.2 else '** TOO FEW **'}")
    ok &= (dimple_reversals > n_dimpled * 0.2)

    print(f"\ngeometry failures: {len(geometry_failures)}")
    for f in geometry_failures[:10]:
        print("  ", f)
    ok &= (len(geometry_failures) == 0)

    print("\nSYNTHETIC FAMILY CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
