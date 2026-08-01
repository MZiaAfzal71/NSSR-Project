"""Verify that this copy of NSSR contains every accumulated fix.

Staleness (an old file left in place, or a cached import in a notebook
kernel) has repeatedly produced confusing results in this project. Run this
first whenever a result looks like a regression -- it checks the actual
source on disk, so it cannot be fooled by a stale in-memory import.

Run:  python scripts/verify_fixes.py
"""
import sys, os, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")


def read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


CHECKS = [
    # (file, must-contain, must-NOT-contain, description)
    ("nssr/preprocess.py", 'out["closed_top"] = True', None,
     "vase computes its full crown patch (parity with reference code); "
     "hiding it is render-only"),
    ("nssr/preprocess.py", 'out["hide_crown_render"]', None,
     "crown hiding is a rendering flag, not a geometry change"),
    ("nssr/geometry.py", "denom.abs() < EPS", None,
     "boundary-direction denominator guarded (vase duplicate-height NaN)"),
    ("nssr/geometry_np.py", "abs(denom) < tiny", None,
     "same guard in the NumPy twin"),
    ("nssr/geometry.py", "nB / torch.sqrt(1.0 +", None,
     "fb/fc DIVIDE by the sqrt term, matching the reference implementation"),
    ("nssr/losses.py", "_nn_sqdist", "_pairwise_min",
     "memory-flat double-chunked nearest-neighbour search"),
    ("nssr/metrics.py", "max_pts", None,
     "evaluation caps point counts for memory"),
    ("scripts/evaluate.py", "HIGHER_IS_BETTER", None,
     "normal-consistency improvement sign handled correctly"),
    ("nssr/train.py", "init_ckpt", None,
     "training can initialize from a checkpoint (synthetic->real finetune)"),
    ("nssr/synthetic.py", "FAMILY_PROBS", None,
     "three synthetic geometric families (standard/dimpled/open_top)"),
    ("nssr/preprocess.py", "def pole_is_dimpled", None,
     "inward-dipping poles detected from Z reversal (apple-style dimples "
     "on real meshes)"),
    ("nssr/preprocess.py", 'out["base_dimpled"]', None,
     "dimple flags recorded and used to pick the non-circular cap formula"),
    ("scripts/visualize_real.py", "panel_contours", None,
     "GT vs input-slices vs reconstruction visualization"),
    ("nssr/slicing.py", "np.linalg.det(M) < 0", None,
     "mesh axis alignment is a rotation, never a reflection"),
    ("nssr/slicing.py", "to_2D=T", None,
     "explicit section->2D frame so slices share a common frame"),
    ("nssr/slicing.py", "poly.interiors", None,
     "slices with holes are rejected"),
    ("scripts/fetch_thingi10k.py", "euler=2", "@ACTIVE_FILTER:solid",
     "Thingi10K filter uses euler=2 (genus 0), not the unpopulated "
     "solid/manifold columns"),
    # --- the six TTO fixes, in order ---
    ("scripts/reconstruct_designer.py", "d_t2p, _ = _nn_sqdist(Z3(i)",
     "chamfer(surface_points(S), Z3(i))",
     "TTO #1: one-sided ring->surface loss (no cap-collapse)"),
    ("scripts/reconstruct_designer.py", "net = ParamNet().to(device=device",
     None,
     "TTO #3: optimizes a geometry-conditioned FUNCTION, not per-row numbers"),
    ("scripts/reconstruct_designer.py", "gap_patch = S[i].reshape(-1, 3)",
     None,
     "TTO #4a: loss targets only the gap patch (no cap-cheating)"),
    ("scripts/reconstruct_designer.py", "1e-3 * span", None,
     "TTO #4b: geometrically degenerate folds are skipped"),
    ("scripts/reconstruct_designer.py", 'ref[i]["s_fB"]',
     'torch.zeros_like(params["s_fB"])',
     "TTO #5+#6: caps held at the REFERENCE (classical or the trained "
     "checkpoint), never hard-zeroed"),
    ("scripts/reconstruct_designer.py", "(params[k] - ref[i][k]) ** 2", None,
     "TTO #6: proximity anchor is relative to the initialization"),
    ("scripts/reconstruct_designer.py", "ref_full[k]", None,
     "TTO #6: drift diagnostic vs the reference"),
]


def main():
    ok = True
    print(f"{'status':8s} {'file':34s} check")
    print("-" * 100)
    for rel, must, must_not, desc in CHECKS:
        try:
            src = read(rel)
        except FileNotFoundError:
            print(f"{'MISSING':8s} {rel:34s} file not found")
            ok = False
            continue
        if must_not == "@ACTIVE_FILTER:solid":
            # `solid=True` legitimately appears in the comment explaining why
            # that filter returns 0 entries; only the ACTIVE base_filters
            # assignment matters, so inspect just that statement.
            i = src.find("base_filters = dict(")
            block = src[i:src.find(")", i)] if i >= 0 else ""
            good = (must in src) and ("solid" not in block)
        else:
            good = (must in src) and (must_not is None or must_not not in src)
        ok &= good
        print(f"{'ok' if good else '** FAIL':8s} {rel:34s} {desc}")
        if not good:
            if must not in src:
                print(f"{'':8s} {'':34s}   expected to find: {must!r}")
            elif must_not == "@ACTIVE_FILTER:solid":
                print(f"{'':8s} {'':34s}   base_filters still contains "
                      f"'solid' (the unpopulated column)")
            elif must_not and must_not in src:
                print(f"{'':8s} {'':34s}   stale code present: {must_not!r}")

    print("\nfile checksums (compare against the ones quoted in chat):")
    for rel in ("scripts/reconstruct_designer.py", "nssr/geometry.py",
                "nssr/preprocess.py", "nssr/synthetic.py",
                "scripts/fetch_thingi10k.py"):
        try:
            h = hashlib.md5(read(rel).encode()).hexdigest()
            print(f"  {h}  {rel}")
        except FileNotFoundError:
            print(f"  {'-'*32}  {rel} (missing)")

    print("\nVERIFY FIXES:", "PASS -- this copy is current"
          if ok else "FAIL -- you are running an OUTDATED copy; re-extract "
                     "the latest package")
    if ok:
        print("\nIf a result still looks like a regression after this passes,")
        print("restart your notebook kernel: Python caches imported modules,")
        print("so updated files on disk are ignored by an already-running")
        print("session (invoking scripts via `!python ...` avoids this).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
