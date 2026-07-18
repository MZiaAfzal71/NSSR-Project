"""Generate the synthetic dataset (pickled samples).

Usage:  python scripts/make_synthetic_dataset.py --n_train 800 --n_val 100 \
            --n_test 100 --slices 5 7 9 15 --out data/synthetic
"""
import sys, os, argparse, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nssr.synthetic import make_sample

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--slices", type=int, nargs="+", default=[7])
    ap.add_argument("--out", default="data/synthetic")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seed = 0
    for split, n in (("train", a.n_train), ("val", a.n_val), ("test", a.n_test)):
        for N in a.slices:
            path = os.path.join(a.out, f"{split}_N{N}.pkl")
            samples = []
            for _ in range(n):
                samples.append(make_sample(seed, N=N)); seed += 1
            with open(path, "wb") as f:
                pickle.dump(samples, f)
            print(f"wrote {path}  ({n} samples, N={N})")

if __name__ == "__main__":
    main()
