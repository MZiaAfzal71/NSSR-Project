"""Build mixed synthetic+real training/validation pickles for NSSR-V2.

The mixed model is trained once on domain-balanced train/val data.  Test sets
remain separate: use scripts/validate.py and scripts/evaluate_projection.py
against data/synthetic and data/real independently.

This is preferable to mixing the test sets because it preserves clear
in-domain and cross-domain paper statistics.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
import numpy as np


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def balanced_mix(a, b, rng, max_per_domain=0):
    """Return a shuffled, domain-balanced concatenation."""
    na, nb = len(a), len(b)
    n = min(na, nb)
    if max_per_domain > 0:
        n = min(n, max_per_domain)
    if n < 1:
        raise RuntimeError("one domain has no samples")
    ia = rng.choice(na, n, replace=False)
    ib = rng.choice(nb, n, replace=False)
    out = [a[i] for i in ia] + [b[i] for i in ib]
    rng.shuffle(out)
    return out, n


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--synthetic", default="data/synthetic")
    ap.add_argument("--real", default="data/real")
    ap.add_argument("--out", default="data/mixed")
    ap.add_argument("--Ns", type=int, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max_train_per_domain", type=int, default=0,
        help="0 uses min(real,synthetic); otherwise cap each domain",
    )
    ap.add_argument(
        "--max_val_per_domain", type=int, default=0,
        help="0 uses min(real,synthetic); otherwise cap each domain",
    )
    a = ap.parse_args()

    if a.Ns:
        Ns = sorted(set(a.Ns))
    else:
        Ns = []
        for p in Path(a.real).glob("train_N*.pkl"):
            try:
                N = int(p.stem.split("N")[-1])
            except ValueError:
                continue
            if (Path(a.synthetic) / p.name).exists():
                Ns.append(N)
        Ns = sorted(set(Ns))

    if not Ns:
        raise SystemExit("no common N datasets found")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    manifest = []

    for N in Ns:
        rng = np.random.default_rng(a.seed + N)
        syn_tr = load(os.path.join(a.synthetic, f"train_N{N}.pkl"))
        real_tr = load(os.path.join(a.real, f"train_N{N}.pkl"))
        syn_va = load(os.path.join(a.synthetic, f"val_N{N}.pkl"))
        real_va = load(os.path.join(a.real, f"val_N{N}.pkl"))

        train, ntr = balanced_mix(
            syn_tr, real_tr, rng, a.max_train_per_domain
        )
        val, nva = balanced_mix(
            syn_va, real_va, rng, a.max_val_per_domain
        )

        save(os.path.join(a.out, f"train_N{N}.pkl"), train)
        save(os.path.join(a.out, f"val_N{N}.pkl"), val)

        # A test pickle is needed only so run_full_sweep's discovery sees a
        # complete dataset. It is NOT the paper test set. We use a balanced
        # diagnostic mixture and later evaluate the checkpoint separately on
        # the untouched synthetic and real test sets.
        syn_te = load(os.path.join(a.synthetic, f"test_N{N}.pkl"))
        real_te = load(os.path.join(a.real, f"test_N{N}.pkl"))
        test, nte = balanced_mix(syn_te, real_te, rng, 0)
        save(os.path.join(a.out, f"test_N{N}.pkl"), test)

        manifest.append((N, ntr, nva, nte, len(train), len(val), len(test)))
        print(
            f"N={N}: train {ntr}+{ntr}={len(train)}, "
            f"val {nva}+{nva}={len(val)}, "
            f"diagnostic test {nte}+{nte}={len(test)}"
        )

    with open(os.path.join(a.out, "MIXED_DATASET.txt"), "w") as f:
        f.write("Domain-balanced synthetic + real training dataset\n")
        f.write("Paper testing must use untouched data/synthetic/test_N*.pkl "
                "and data/real/test_N*.pkl separately.\n\n")
        for x in manifest:
            f.write(
                f"N={x[0]} train/domain={x[1]} val/domain={x[2]} "
                f"diag_test/domain={x[3]}\n"
            )

    print("wrote", os.path.join(a.out, "MIXED_DATASET.txt"))


if __name__ == "__main__":
    main()
