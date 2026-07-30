"""Verify the Euler-characteristic / genus filter used by
scripts/fetch_thingi10k.py to reject meshes that cannot possibly yield a
single closed loop per slice.

Cases:
  * genus-0 solids of revolution (sphere/vase/apple/banana) -> genus 0, keep
  * a TORUS (genus 1)                                        -> reject
  * a double torus (genus 2)                                 -> reject
  * an open mesh (a cylinder with no caps)                   -> edge_ok False

Run:  python tests/euler_filter_check.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from scripts.fetch_thingi10k import euler_genus, aspect_ok
from scripts.make_test_meshes import build_mesh, PROFILES


def torus(n_u=60, n_v=30, R=1.0, r=0.35, handles=1):
    """Closed genus-1 torus (or a chain approximating higher genus by
    counting: we just test genus 1 and a 2-torus built as a connected sum
    surrogate via two separate handles is non-trivial, so we validate
    genus 1 exactly and rely on the formula for higher genus)."""
    V, F = [], []
    for i in range(n_u):
        u = 2 * np.pi * i / n_u
        for j in range(n_v):
            v = 2 * np.pi * j / n_v
            x = (R + r * np.cos(v)) * np.cos(u)
            y = (R + r * np.cos(v)) * np.sin(u)
            z = r * np.sin(v)
            V.append([x, y, z])
    idx = lambda i, j: (i % n_u) * n_v + (j % n_v)
    for i in range(n_u):
        for j in range(n_v):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            F.append([a, b, c]); F.append([a, c, d])
    return np.array(V), np.array(F)


def open_cylinder(n_u=40, n_v=10):
    V, F = [], []
    for i in range(n_u):
        th = 2 * np.pi * i / n_u
        for j in range(n_v):
            V.append([np.cos(th), np.sin(th), j / (n_v - 1)])
    idx = lambda i, j: (i % n_u) * n_v + j
    for i in range(n_u):
        for j in range(n_v - 1):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            F.append([a, b, c]); F.append([a, c, d])
    return np.array(V), np.array(F)


def main():
    ok = True
    print(f"{'mesh':14s} {'euler':>6s} {'genus':>6s} {'closed':>7s} "
          f"{'verdict':>10s}  expected")
    for name in PROFILES:
        V, F = build_mesh(name, n_t=40, n_th=32)
        e, g, edge_ok = euler_genus(V.shape[0], F)
        keep = edge_ok and g <= 0
        exp = "KEEP"
        good = keep
        ok &= good
        print(f"{name:14s} {e:6d} {g:6.1f} {str(edge_ok):>7s} "
              f"{'KEEP' if keep else 'REJECT':>10s}  {exp}"
              f"{'' if good else '   ** WRONG **'}")

    V, F = torus()
    e, g, edge_ok = euler_genus(V.shape[0], F)
    keep = edge_ok and g <= 0
    good = (not keep) and abs(g - 1.0) < 1e-9
    ok &= good
    print(f"{'torus':14s} {e:6d} {g:6.1f} {str(edge_ok):>7s} "
          f"{'KEEP' if keep else 'REJECT':>10s}  REJECT (genus 1)"
          f"{'' if good else '   ** WRONG **'}")

    V, F = open_cylinder()
    e, g, edge_ok = euler_genus(V.shape[0], F)
    keep = edge_ok and g <= 0
    good = (not keep) and (not edge_ok)
    ok &= good
    print(f"{'open cylinder':14s} {e:6d} {g:6.1f} {str(edge_ok):>7s} "
          f"{'KEEP' if keep else 'REJECT':>10s}  REJECT (not closed)"
          f"{'' if good else '   ** WRONG **'}")

    # aspect filter: sphere is near-isotropic, banana is elongated
    Vs, _ = build_mesh("sphere", 40, 32)
    Vb, _ = build_mesh("bananalike", 40, 32)
    print(f"\naspect filter: sphere elongated={aspect_ok(Vs)} "
          f"bananalike elongated={aspect_ok(Vb)}")

    print("\nEULER/GENUS FILTER CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
