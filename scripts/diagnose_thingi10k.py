"""Diagnose why a thingi10k query returns nothing.

`fetch_thingi10k.py` combines several filters plus an optional CLIP query.
If the combination yields zero entries, this script finds out WHICH filter
is responsible, instead of guessing. It:

  1. prints the keys/values of one real entry (so we see what metadata the
     chosen variant actually exposes),
  2. counts entries for each filter INDIVIDUALLY,
  3. counts entries for filters applied CUMULATIVELY,
  4. tests a CLIP query alone, and combined with the geometric filters.

Run this, paste the output, and the fix is then obvious.

Usage:
    python scripts/diagnose_thingi10k.py --variant tetwild
    python scripts/diagnose_thingi10k.py --variant npz --query "a vase"
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def count(it, cap=100000):
    """Count entries in whatever dataset() returns (iterator or Dataset)."""
    try:
        return len(it)
    except TypeError:
        pass
    n = 0
    for _ in it:
        n += 1
        if n >= cap:
            break
    return n


def try_filter(thingi10k, label, **kw):
    try:
        ds = thingi10k.dataset(**kw)
        n = count(ds)
        print(f"  {label:52s} -> {n:6d} entries"
              + ("   *** ZERO ***" if n == 0 else ""))
        return n
    except TypeError as e:
        print(f"  {label:52s} -> UNSUPPORTED ({e})")
        return None
    except Exception as e:                             # noqa: BLE001
        print(f"  {label:52s} -> ERROR {type(e).__name__}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tetwild",
                    choices=["tetwild", "npz", "raw"])
    ap.add_argument("--query", default="a vase")
    ap.add_argument("--cache_dir", default=None)
    a = ap.parse_args()

    import thingi10k
    kw = {"variant": a.variant}
    if a.cache_dir:
        kw["cache_dir"] = a.cache_dir
    thingi10k.init(**kw)

    print(f"\n=== variant: {a.variant} ===")
    print("\n--- 1. unfiltered ---")
    total = try_filter(thingi10k, "dataset()")

    print("\n--- 2. one sample entry: what metadata exists? ---")
    try:
        for entry in thingi10k.dataset():
            for k in sorted(entry.keys()):
                v = entry[k]
                sv = str(v)
                if len(sv) > 60:
                    sv = sv[:60] + "..."
                print(f"    {k:26s} = {sv}")
            break
    except Exception as e:                             # noqa: BLE001
        print(f"    could not read an entry: {e}")

    print("\n--- 3. each filter INDIVIDUALLY ---")
    singles = {
        "closed=True": dict(closed=True),
        "num_components=1": dict(num_components=1),
        "self_intersecting=False": dict(self_intersecting=False),
        "solid=True": dict(solid=True),
        "vertex_manifold=True": dict(vertex_manifold=True),
        "edge_manifold=True": dict(edge_manifold=True),
        "num_vertices=(500,200000)": dict(num_vertices=(500, 200000)),
        "euler=2": dict(euler=2),
        "genus=0": dict(genus=0),
    }
    for label, kwf in singles.items():
        try_filter(thingi10k, label, **kwf)

    print("\n--- 4. CUMULATIVE (the combination fetch_thingi10k.py uses) ---")
    cum = {}
    for label, kwf in [("closed=True", dict(closed=True)),
                       ("+ num_components=1", dict(num_components=1)),
                       ("+ self_intersecting=False",
                        dict(self_intersecting=False)),
                       ("+ solid=True", dict(solid=True)),
                       ("+ num_vertices=(500,200000)",
                        dict(num_vertices=(500, 200000)))]:
        cum.update(kwf)
        try_filter(thingi10k, label, **dict(cum))

    print("\n--- 5. CLIP query ---")
    try_filter(thingi10k, f"query={a.query!r} alone", query=a.query)
    try_filter(thingi10k, f"query={a.query!r} + closed=True",
               query=a.query, closed=True)
    try_filter(thingi10k, f"query={a.query!r} + full filter set",
               query=a.query, **dict(cum))

    print("\nInterpretation:")
    print("  * If a single filter shows ZERO, that field is unavailable or")
    print("    always-false for this variant -- drop it.")
    print("  * If the cumulative row drops to zero at a specific step, that")
    print("    step is the culprit.")
    print("  * If 'query alone' is zero, CLIP search is the problem: rerun")
    print("    fetch_thingi10k.py with --no_query.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
