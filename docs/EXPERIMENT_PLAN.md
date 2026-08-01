# EXPERIMENT PLAN — every command, in order

Run top to bottom. Each phase gates the next. Times assume one T4
(Kaggle/Colab); adjust `--surf_sub`/`--m` down if memory is tight.

Everything writes into `results/` and `runs/`. Keep both — they are your
paper's evidence.

---

## Phase A — verification (no GPU, ~2 min total)

```bash
python scripts/verify_fixes.py              # confirm this copy is current
python tests/numpy_sanity_check.py          # analytic objects reconstruct
python tests/parity_check.py                # NSSR classical == your CiSE code
python tests/synthetic_family_check.py      # 3 geometric families sane
python tests/euler_filter_check.py          # genus filter keeps/rejects right
python scripts/smoke_test.py                # torch == numpy, gradients flow
```

All must pass. `verify_fixes.py` checks the source ON DISK for every
accumulated fix and prints file checksums, so it cannot be fooled by a
stale in-memory import -- run it first whenever a result looks like a
regression.

All others must pass too. If `parity_check` fails, stop — everything downstream
depends on the classical baseline being exactly your published method.

---

## Phase B — synthetic data (~10 min)

```bash
python scripts/make_synthetic_dataset.py \
    --n_train 800 --n_val 100 --n_test 100 \
    --slices 5 7 9 15 --out data/synthetic
```

Record the printed family mix per split — it goes in the paper's dataset
paragraph.

---

## Phase C — synthetic training, one run per slice count (~30-45 min each)

```bash
for N in 5 7 9 15; do
  python scripts/train_model.py --data data/synthetic --N $N --m 256 \
      --epochs 100 --surf_sub 8000 --gt_sub 8000 \
      --val_every 5 --val_subset 25 --patience 30 \
      --out runs/synth_N$N
done
```

Keep `runs/synth_N*/log.csv` — epoch 0 is the classical baseline, so each
log contains its own before/after comparison.

---

## Phase D — synthetic evaluation + the sparsity sweep (~10 min)

```bash
for N in 5 7 9 15; do
  python scripts/evaluate.py --data data/synthetic --N $N \
      --ckpt runs/synth_N$N/best.pt --n_u 32 \
      --out results/eval_synth_N$N.csv
done
```

This produces the **sparsity sweep** (accuracy vs. slice count) — one of
the two headline figures. Normal-consistency direction is now labelled
correctly in the printout.

---

## Phase E — the "just retune the constants" ablation (~15 min per N)

```bash
for N in 7 9; do
  python scripts/ablation_global_constants.py --data data/synthetic --N $N \
      --ckpt runs/synth_N$N/best.pt --out results/ablation_global_N$N.csv
done
```

Three-way table: classical / best global constants (fit on train, scored on
test) / learned. This is the answer to the first question every reviewer
asks. Watch for the grid-boundary warning.

---

## Phase F — designer shapes, all modes (~5 min)

```bash
for ds in banana apple vase; do
  python scripts/reconstruct_designer.py --ds $ds --mode classical
  python scripts/reconstruct_designer.py --ds $ds --mode net \
      --ckpt runs/synth_N9/best.pt
  python scripts/reconstruct_designer.py --ds $ds --mode tto \
      --tto_init net --ckpt runs/synth_N9/best.pt
  python scripts/reconstruct_designer.py --ds $ds --mode tto \
      --tto_init classical
done
python scripts/reconstruct_designer.py --ds vase --mode classical --show_crown
```

12 renders + the crown-visible variant. `classical` reproduces your CiSE
Figure 1; `net` is zero-shot synthetic->designer generalization; the two
`tto` modes are per-object fine-tune and fully training-data-free.

---

## Phase G — real meshes: acquisition (~30-60 min, mostly download)

```bash
pip install 'thingi10k[clip]' trimesh shapely

# Sanity-check the mesh path on generated meshes first (no download):
python scripts/make_test_meshes.py --out data/meshes_test
python scripts/check_mesh_pipeline.py --meshes data/meshes_test --N 9

# Then fetch real ones:
python scripts/fetch_thingi10k.py --out data/meshes --limit 200
python scripts/check_mesh_pipeline.py --meshes data/meshes --N 9 --limit 25
```

**Filter set (determined empirically, not guessed).** Running
`scripts/diagnose_thingi10k.py --variant tetwild` on the real dataset gave:

| filter | entries (of 9976) | verdict |
|---|---:|---|
| `closed=True` | 9007 | use |
| `num_components=1` | 7754 | use |
| `euler=2` | 2911 | **use — equals genus 0** |
| `self_intersecting=False` | 9976 | no-op (TetWild removed them all) |
| `solid=True` | **0** | column unpopulated — never use |
| `vertex_manifold=True` | **0** | column unpopulated — never use |
| `edge_manifold=True` | **0** | column unpopulated — never use |
| `genus=0` | **0** | column unpopulated — use `euler=2` instead |

The first run wrote zero meshes because it required `solid=True`, which is
`False` for every entry in this variant (those boolean columns describe
properties that are simply not filled in for the TetWild remeshes). The
fix uses `euler=2`, which for a closed orientable manifold is exactly the
genus-0 condition we want, and drops the unpopulated filters. Expected
pool: roughly 1900 candidates.

CLIP queries return only ~20 hits each, so the script now runs 16 queries
and then tops up from the geometric pool to reach `--limit`.

If a future run still writes 0, re-run the diagnostic and send the output:

```bash
python scripts/diagnose_thingi10k.py --variant tetwild
```

**Download-free fallback.** You are not blocked on Thingi10K. A corpus of
randomized genus-0 solids of revolution (verified watertight, varied
profiles, bent axes, circumferential lobes) can be generated locally and
run through the *identical* mesh pipeline -- slicing, dataset build,
training:

```bash
python scripts/make_test_meshes.py --out data/meshes_gen --count 200
python scripts/check_mesh_pipeline.py --meshes data/meshes_gen --N 9 --limit 20
python scripts/make_mesh_dataset.py --meshes data/meshes_gen --N 5 7 9 15 \
    --out data/real
```

Be precise about this in the paper: these are *generated* meshes exercised
through the real-mesh path, not scanned real-world objects. They validate
the mesh pipeline end to end and give a second, independent evaluation
corpus, but they are not a substitute for a real-data claim. Use them to
keep progressing while the Thingi10K issue is resolved.

**Critical choice:** `fetch_thingi10k.py` defaults to the **tetwild**
variant. Raw Thingi10K is 50% non-solid, 26% multi-component, 22%
non-manifold, 11% open — nearly all unusable for slice-based
reconstruction. TetWild-remeshed meshes are closed and high quality. The
script also applies the package's own filters (`closed`, `num_components=1`,
`self_intersecting=False`, `solid=True`), a genus-0 Euler check, and an
elongation filter (near-perfect spheres are trivially easy and inflate your
numbers).

It also uses **CLIP semantic search** to target suitable object classes
("a vase", "a bottle", "a jar", "a cup", "a bowl", "an egg", "a pear", "a
simple round pot") instead of scanning 10k models and discarding most.
Add `--no_query` for geometric filters only, or pass your own `--queries`.

Read the rejection report from `check_mesh_pipeline.py` — the kept/rejected
counts and reasons belong in the paper.

---

## Phase H — real-mesh dataset + three experiments (~1-2 h)

```bash
python scripts/make_mesh_dataset.py --meshes data/meshes --N 5 7 9 15 \
    --out data/real --val_frac 0.15 --test_frac 0.15

# H1. train on real, test on real  (headline real-data result)
python scripts/train_model.py --data data/real --N 9 --m 256 \
    --epochs 100 --surf_sub 8000 --gt_sub 8000 \
    --val_every 5 --val_subset 15 --patience 30 --out runs/real_N9
python scripts/evaluate.py --data data/real --N 9 \
    --ckpt runs/real_N9/best.pt --n_u 32 --out results/eval_real_N9.csv

# H2. ZERO-SHOT: train on synthetic, test on real
#     (strong claim if it holds: no real training data needed)
python scripts/evaluate.py --data data/real --N 9 \
    --ckpt runs/synth_N9/best.pt --n_u 32 \
    --out results/eval_real_zeroshot_N9.csv

# H3. synthetic pre-train -> fine-tune on real (usually best numbers)
python scripts/train_model.py --data data/real --N 9 --m 256 \
    --epochs 40 --lr 3e-4 --surf_sub 8000 --gt_sub 8000 \
    --val_every 5 --val_subset 15 --patience 15 \
    --init_ckpt runs/synth_N9/best.pt --out runs/real_ft_N9
python scripts/evaluate.py --data data/real --N 9 \
    --ckpt runs/real_ft_N9/best.pt --n_u 32 \
    --out results/eval_real_finetune_N9.csv
```

Also worth running on a few real objects, since it needs no training data
at all and is the fair comparison against per-shape methods like OReX:

```bash
python scripts/ablation_global_constants.py --data data/real --N 9 \
    --ckpt runs/real_N9/best.pt --out results/ablation_real_N9.csv
```

---

## Phase H2 — visualize real results (do this as soon as H1 finishes)

```bash
python scripts/visualize_real.py --data data/real --N 9 \
    --ckpt runs/real_N9/best.pt --n 8 --out results/real_figs
```

Per object, four panels: (a) the ground-truth mesh, (b) the input slice
contours -- worth showing, because it makes vivid how little information
the method actually gets, (c) the classical reconstruction and (d) the
learned one, both coloured by distance to ground truth on a SHARED colour
scale so they are directly comparable, with mean error and % improvement in
each title. This is the real-data figure for the paper.

You can also run it straight from meshes without building a dataset:
```bash
python scripts/visualize_real.py --meshes data/meshes --N 9 --n 6 \
    --out results/real_figs_quick
```

### Inward-dipping poles on real data

The apple's inward-folding caps are intended (the paper's non-monotone test
case). For DESIGNER shapes this comes from the hardcoded `Null_Hts`. For
real meshes there is no such prior, so `preprocess_object` now DETECTS a
dimpled pole from the data: the height sequence reversing at an end
(`Z[1] < Z[0]`, or `Z[-1] < Z[-2]`) means the slices are already folding
back, so the cap must close inward. Detected ends get a cap reference
INSIDE the contour range and use the non-circular cap formulas (Eqs. 33-34
/ 37-38) -- exactly the apple's convention.

Verified on the designer shapes: apple flags both poles, banana neither.
(An earlier version of this detector tested mean RADIUS instead and was
almost exactly anti-correlated with the truth -- the apple's radii shrink
monotonically toward both poles, so radius never distinguished it.)

Caveat worth stating in the paper: whether a real object's end is a dimple
or a normal cap is only inferable when the slicing actually captures the
reversal. If the outermost slice sits above the stem well, the object looks
like a normal taper and gets an outward cap. Slice placement therefore
determines what is recoverable -- an honest limitation of cross-sectional
input generally, not of NSSR specifically.

## Phase I — external baseline (optional but strengthens the paper)

```bash
python baselines/implicit_baseline.py --sample data/synthetic/test_N9.pkl --idx 0
```

For the headline comparison also run the official **OReX**
(github.com/haimsaw/OReX) on the same slice inputs. That is the method
reviewers will name, and NSSR's guarantee argument lands hardest next to it.

---

## What to send me for the write-up

- `results/eval_synth_N*.csv` (sparsity sweep)
- `results/ablation_global_N*.csv` (the retuning ablation)
- `runs/*/log.csv` (training curves + epoch-0 baselines)
- `results/eval_real_*.csv` (three real-data experiments)
- the 12 designer renders
- the rejection report from `check_mesh_pipeline.py`

With those I can build the tables and figures and draft the results and
method sections.
