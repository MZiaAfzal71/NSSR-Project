"""Write known-good watertight OBJ meshes (pure NumPy, no downloads).

Purpose: validate the real-mesh pipeline (slicing -> dataset -> training)
end to end BEFORE spending time scraping Thingi10k / ShapeNet. These are
solids of revolution with analytic profiles chosen to mirror the three
designer shapes:

    sphere   -- trivial sanity case
    vaselike -- pinched neck + flared rim
    applelike-- dimpled poles (non-monotone radius near both ends)
    bananalike-- bent axis, tapered ends

Each mesh is a closed quad grid with triangle fans at both poles, so it is
watertight by construction (verified here: every edge shared by exactly 2
faces, and Euler characteristic V - E + F = 2).

Usage:
    python scripts/make_test_meshes.py --out data/meshes_test
    python scripts/check_mesh_pipeline.py --meshes data/meshes_test
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np


# ---------------------------------------------------------------- profiles
def prof_sphere(t):
    return np.sin(np.pi * t), np.cos(np.pi * t) * -1.0


def prof_vaselike(t):
    z = 2.2 * t - 1.1
    r = (0.45 + 0.42 * np.exp(-((t - 0.22) ** 2) / 0.02)
         - 0.22 * np.exp(-((t - 0.62) ** 2) / 0.02)
         + 0.30 * np.exp(-((t - 0.97) ** 2) / 0.004))
    r = r * np.sin(np.pi * np.clip(t, 0, 1)) ** 0.25
    return np.clip(r, 1e-3, None), z


def prof_applelike(t):
    z = 2.0 * t - 1.0
    r = np.sqrt(np.clip(1.0 - z ** 2, 0, None)) ** 0.85
    r = r * (1.0 - 0.30 * np.exp(-((t - 0.02) ** 2) / 0.004)
                - 0.30 * np.exp(-((t - 0.98) ** 2) / 0.004))
    return np.clip(r, 1e-3, None), z


def prof_bananalike(t):
    z = 2.4 * t - 1.2
    r = 0.30 * np.sin(np.pi * np.clip(t, 0, 1)) ** 0.6
    return np.clip(r, 1e-3, None), z


BEND = {"bananalike": 0.55}
PROFILES = {"sphere": prof_sphere, "vaselike": prof_vaselike,
            "applelike": prof_applelike, "bananalike": prof_bananalike}


# ------------------------------------------------------------------ meshing
def build_mesh(name, n_t=96, n_th=64):
    """Solid of revolution -> (vertices (V,3), faces (F,3)) watertight."""
    prof = PROFILES[name]
    bend = BEND.get(name, 0.0)
    t = np.linspace(0.0, 1.0, n_t)
    th = np.linspace(0.0, 2 * np.pi, n_th, endpoint=False)
    r, z = prof(t)

    verts = [np.array([bend * np.sin(np.pi * t[0]), 0.0, z[0]])]   # south pole
    for i in range(1, n_t - 1):
        x0 = bend * np.sin(np.pi * t[i])
        ring = np.stack([x0 + r[i] * np.cos(th),
                         r[i] * np.sin(th),
                         np.full(n_th, z[i])], axis=1)
        verts.append(ring)
    verts.append(np.array([bend * np.sin(np.pi * t[-1]), 0.0, z[-1]]))  # north
    V = np.vstack([verts[0][None, :]] + verts[1:-1] + [verts[-1][None, :]])

    faces = []
    n_rings = n_t - 2
    # south fan
    for j in range(n_th):
        faces.append([0, 1 + j, 1 + (j + 1) % n_th])
    # quad band -> 2 triangles
    for i in range(n_rings - 1):
        a0 = 1 + i * n_th
        b0 = 1 + (i + 1) * n_th
        for j in range(n_th):
            j1 = (j + 1) % n_th
            faces.append([a0 + j, b0 + j, b0 + j1])
            faces.append([a0 + j, b0 + j1, a0 + j1])
    # north fan
    top = V.shape[0] - 1
    a0 = 1 + (n_rings - 1) * n_th
    for j in range(n_th):
        j1 = (j + 1) % n_th
        faces.append([a0 + j, top, a0 + j1])
    return V, np.asarray(faces, dtype=np.int64)


def check_watertight(V, F):
    """Every edge in exactly 2 faces, and V - E + F == 2 (genus 0)."""
    from collections import Counter
    c = Counter()
    for f in F:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            c[(min(a, b), max(a, b))] += 1
    counts = Counter(c.values())
    E = len(c)
    euler = V.shape[0] - E + F.shape[0]
    return counts, euler, all(v == 2 for v in c.values()) and euler == 2


def write_obj(path, V, F):
    with open(path, "w") as fh:
        for v in V:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in F:
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/meshes_test")
    ap.add_argument("--n_t", type=int, default=96)
    ap.add_argument("--n_th", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ok_all = True
    for name in PROFILES:
        V, F = build_mesh(name, a.n_t, a.n_th)
        counts, euler, ok = check_watertight(V, F)
        ok_all &= ok
        p = os.path.join(a.out, f"{name}.obj")
        write_obj(p, V, F)
        print(f"{name:11s} V={V.shape[0]:5d} F={F.shape[0]:5d} "
              f"edge-degree={dict(counts)} euler={euler} "
              f"{'watertight OK' if ok else '** NOT WATERTIGHT **'} -> {p}")
    print("\nAll meshes watertight:", ok_all)
    print(f"Next: python scripts/check_mesh_pipeline.py --meshes {a.out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
