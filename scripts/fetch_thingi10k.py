"""Fetch suitable meshes from Thingi10K and write them as OBJ for NSSR.

Why this script exists: raw Thingi10K is mostly UNUSABLE for cross-sectional
reconstruction. The dataset's own published statistics: 50% non-solid, 45%
self-intersecting, 31% coplanar self-intersecting, 26% multiple components,
22% non-manifold, 11% topologically open. Feeding that directly into
slice-based reconstruction would reject almost everything. Two features of
the official `thingi10k` package solve this:

 1. **The `tetwild` variant.** Surface meshes remeshed with TetWild --
    high-quality, closed triangle meshes. Use this, not `raw`.
 2. **Built-in filters + CLIP semantic search.** We can ask for
    `closed=True, num_components=1, self_intersecting=False, solid=True`
    AND semantically query for the object classes that actually suit a
    star-shaped, single-loop-per-slice pipeline ("a vase", "a bottle", "a
    jar", ...). That is far better than downloading 10k models and
    discarding 95%.

We additionally pre-filter on GENUS via the Euler characteristic
(V - E + F = 2 - 2g), keeping only genus-0 meshes, since a handle guarantees
some slice will produce more than one loop.

Setup:
    pip install 'thingi10k[clip]' trimesh shapely
    # (drop [clip] if you don't want semantic search; then use --no_query)

Usage:
    python scripts/fetch_thingi10k.py --out data/meshes --limit 150
    python scripts/fetch_thingi10k.py --out data/meshes --limit 150 \
        --queries "a vase" "a bottle" "a jar" "a cup" "an egg"

Then:
    python scripts/check_mesh_pipeline.py --meshes data/meshes --N 9 --limit 20
    python scripts/make_mesh_dataset.py --meshes data/meshes --N 5 7 9 15 \
        --out data/real

NOTE: this script talks to a third-party package and network, and could not
be executed in the environment where it was written. The OBJ writing and the
genus/Euler filter ARE tested (tests/euler_filter_check.py). If the
thingi10k API has changed, `help(thingi10k.dataset)` lists current options.
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

DEFAULT_QUERIES = ["a vase", "a bottle", "a jar", "a cup", "a bowl",
                   "an egg", "a pear", "a simple round pot", "a flower pot",
                   "a drinking glass", "a bud vase", "an urn", "a goblet",
                   "a rounded container", "a smooth organic shape",
                   "a piece of fruit"]


def euler_genus(V_count, F):
    """(euler, genus, edge_ok) for a triangle mesh.
    Closed orientable manifold: V - E + F = 2 - 2g, so g = (2 - euler)/2.
    edge_ok is False if any edge is not shared by exactly 2 faces (i.e. the
    mesh is not a closed manifold, so the genus formula does not apply)."""
    from collections import Counter
    c = Counter()
    for f in F:
        a, b, d = int(f[0]), int(f[1]), int(f[2])
        for u, v in ((a, b), (b, d), (d, a)):
            c[(min(u, v), max(u, v))] += 1
    E = len(c)
    edge_ok = all(v == 2 for v in c.values())
    euler = V_count - E + len(F)
    genus = (2 - euler) / 2.0
    return euler, genus, edge_ok


def write_obj(path, V, F):
    with open(path, "w") as fh:
        for v in V:
            fh.write(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n")
        for f in F:
            fh.write(f"f {int(f[0])+1} {int(f[1])+1} {int(f[2])+1}\n")


def aspect_ok(V, min_ratio=1.15):
    """Require some elongation/structure: a near-perfect sphere gives a
    trivially easy reconstruction. Ratio of longest to shortest extent."""
    ext = V.max(0) - V.min(0)
    ext = np.sort(ext)
    if ext[0] <= 0:
        return False
    return (ext[2] / ext[0]) >= min_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/meshes")
    ap.add_argument("--limit", type=int, default=150,
                    help="max meshes to write")
    ap.add_argument("--variant", default="tetwild",
                    choices=["tetwild", "npz", "raw"],
                    help="tetwild = TetWild-remeshed, closed, high quality "
                         "(strongly recommended)")
    ap.add_argument("--queries", nargs="*", default=None,
                    help=f"CLIP semantic queries; default {DEFAULT_QUERIES}")
    ap.add_argument("--no_query", action="store_true",
                    help="skip CLIP semantic search (geometric filters only)")
    ap.add_argument("--min_vertices", type=int, default=500)
    ap.add_argument("--max_vertices", type=int, default=200000)
    ap.add_argument("--max_genus", type=int, default=0)
    ap.add_argument("--per_query", type=int, default=40)
    ap.add_argument("--cache_dir", default=None)
    a = ap.parse_args()

    try:
        import thingi10k
    except ImportError:
        print("ERROR: pip install 'thingi10k[clip]'   (or plain thingi10k "
              "plus --no_query)")
        return 1

    os.makedirs(a.out, exist_ok=True)
    print(f"initializing thingi10k (variant={a.variant}) -- first run "
          f"downloads the dataset and may take a while ...")
    kw = {"variant": a.variant}
    if a.cache_dir:
        kw["cache_dir"] = a.cache_dir
    thingi10k.init(**kw)

    # Filter set determined empirically with scripts/diagnose_thingi10k.py
    # on the tetwild variant (9976 entries total):
    #   closed=True             -> 9007   (usable)
    #   num_components=1        -> 7754   (usable)
    #   self_intersecting=False -> 9976   (no-op: TetWild removed them all)
    #   euler=2                 -> 2911   (usable; == GENUS 0 for a closed
    #                                      orientable manifold, exactly the
    #                                      property we need)
    #   solid=True              ->    0   <-- column unpopulated for this
    #   vertex_manifold=True    ->    0       variant; ALL entries are False
    #   edge_manifold=True      ->    0       so these filters wipe everything
    #   genus=0                 ->    0   <-- column unpopulated; use euler=2
    # Requiring solid=True is what made the first run write zero meshes.
    base_filters = dict(closed=True, num_components=1,
                        euler=2,
                        num_vertices=(a.min_vertices, a.max_vertices))

    def count_of(**kw):
        """Size of a filtered dataset, tolerating unsupported kwargs."""
        try:
            ds = thingi10k.dataset(**kw)
        except TypeError:
            return None
        try:
            return len(ds)
        except TypeError:
            n = 0
            for _ in ds:
                n += 1
            return n

    # ---- progressively relax the filters until something survives -------
    # A silently-empty result set is the most likely failure mode here
    # (metadata fields differ between variants, and CLIP query + strict
    # filters can intersect to nothing), so probe BEFORE downloading meshes
    # and report exactly what was dropped.
    relax_order = ["num_vertices", "euler", "num_components", "closed"]
    active = dict(base_filters)
    n = count_of(**active)
    if n in (None, 0):
        print(f"full filter set yields {n} entries -- relaxing:")
        for key in relax_order:
            if key not in active:
                continue
            dropped = active.pop(key)
            n = count_of(**active)
            print(f"  dropped {key}={dropped} -> {n} entries")
            if n:
                break
    if not n:
        print("ERROR: no entries even with all filters relaxed. Run:\n"
              "  python scripts/diagnose_thingi10k.py --variant "
              f"{a.variant}\nand send me the output.")
        return 1
    print(f"using filters {active} ({n} candidate entries)")

    queries = ([] if a.no_query
               else (a.queries if a.queries is not None else DEFAULT_QUERIES))
    if queries:
        probe = count_of(query=queries[0], **active)
        if probe in (None, 0):
            print(f"CLIP query {queries[0]!r} + filters yields {probe} "
                  f"entries -> disabling semantic search, using geometric "
                  f"filters only")
            queries = []
    plans = ([{"query": q, **active} for q in queries] if queries
             else [dict(active)])

    written, seen = 0, set()
    stats = {"euler_reject": 0, "genus_reject": 0, "aspect_reject": 0,
             "load_fail": 0, "ok": 0}
    for plan in plans:
        label = plan.get("query", "geometric-only")
        got = 0
        print(f"\n--- querying: {label}")
        try:
            it = thingi10k.dataset(**plan)
        except TypeError as e:
            print(f"  filter not supported by this thingi10k version ({e}); "
                  f"retrying with geometric filters only")
            it = thingi10k.dataset(**active)
        for entry in it:
            if written >= a.limit or got >= a.per_query:
                break
            fid = entry.get("file_id")
            if fid in seen:
                continue
            seen.add(fid)
            try:
                V, F = thingi10k.load_file(entry["file_path"])
                V = np.asarray(V, dtype=np.float64)
                F = np.asarray(F, dtype=np.int64)
                if F.shape[1] != 3:
                    stats["load_fail"] += 1
                    continue
            except Exception:                          # noqa: BLE001
                stats["load_fail"] += 1
                continue
            euler, genus, edge_ok = euler_genus(V.shape[0], F)
            if not edge_ok:
                stats["euler_reject"] += 1
                continue
            if genus > a.max_genus:
                stats["genus_reject"] += 1
                continue
            if not aspect_ok(V):
                stats["aspect_reject"] += 1
                continue
            write_obj(os.path.join(a.out, f"thingi_{fid}.obj"), V, F)
            written += 1
            got += 1
            stats["ok"] += 1
        print(f"  wrote {got} from this query (total {written})")
        if written >= a.limit:
            break

    # CLIP queries return only ~20 hits each, so top up from the geometric
    # pool if we are still short of --limit.
    if written < a.limit and queries:
        print(f"\n--- topping up with geometric filters only "
              f"({written}/{a.limit} so far)")
        try:
            for entry in thingi10k.dataset(**active):
                if written >= a.limit:
                    break
                fid = entry.get("file_id")
                if fid in seen:
                    continue
                seen.add(fid)
                try:
                    V, F = thingi10k.load_file(entry["file_path"])
                    V = np.asarray(V, dtype=np.float64)
                    F = np.asarray(F, dtype=np.int64)
                    if F.shape[1] != 3:
                        stats["load_fail"] += 1
                        continue
                except Exception:                      # noqa: BLE001
                    stats["load_fail"] += 1
                    continue
                euler, genus, edge_ok = euler_genus(V.shape[0], F)
                if not edge_ok:
                    stats["euler_reject"] += 1
                    continue
                if genus > a.max_genus:
                    stats["genus_reject"] += 1
                    continue
                if not aspect_ok(V):
                    stats["aspect_reject"] += 1
                    continue
                write_obj(os.path.join(a.out, f"thingi_{fid}.obj"), V, F)
                written += 1
                stats["ok"] += 1
        except Exception as e:                         # noqa: BLE001
            print(f"  top-up pass failed: {type(e).__name__}: {e}")
        print(f"  after top-up: {written}")

    print(f"\nwrote {written} meshes to {a.out}")
    print("filter outcomes:", stats)
    if written == 0:
        print("\nNOTHING WAS WRITTEN. All-zero counters above mean the "
              "dataset query returned no entries at all (not that meshes "
              "were rejected). Run:\n"
              f"  python scripts/diagnose_thingi10k.py --variant {a.variant}\n"
              "and send me the output.")
        return 1
    print("\nNext:")
    print(f"  python scripts/check_mesh_pipeline.py --meshes {a.out} "
          f"--N 9 --limit 20")
    print(f"  python scripts/make_mesh_dataset.py --meshes {a.out} "
          f"--N 5 7 9 15 --out data/real")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
