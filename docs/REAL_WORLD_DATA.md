# Real-world data: getting from meshes to NSSR results

## Step 0 — validate the pipeline with zero downloads (do this first)

```bash
pip install trimesh shapely --break-system-packages
python scripts/make_test_meshes.py --out data/meshes_test
python scripts/check_mesh_pipeline.py --meshes data/meshes_test --N 9
```

`make_test_meshes.py` writes four watertight OBJ solids of revolution
(sphere, vaselike, applelike with dimpled poles, bananalike with a bent
axis) using pure NumPy -- no downloads, no trimesh needed to CREATE them.
Watertightness is verified by construction (every edge in exactly 2 faces,
Euler characteristic V-E+F=2).

`check_mesh_pipeline.py` then runs load -> align -> slice -> preprocess ->
classical reconstruct on each, and reports where anything fails. Expect all
four to pass with classical surface error around 0.02-0.04.

Three real bugs were found and fixed in `nssr/slicing.py` by review (it is
the one module that cannot be executed in an environment without trimesh):

1. **Axis alignment was sometimes a REFLECTION.** The permutation matrix
   used to stand the longest axis up along +z has determinant -1 for 3 of
   the 6 possible orderings -- mirroring the mesh and inverting its
   normals, which would silently corrupt the normal-consistency metric on
   roughly half your real meshes. Now forced to a proper rotation.
2. **Section-to-2D used an implicit frame.** trimesh's `to_planar()` picks
   its own in-plane axes, which can differ between slices; contours from
   different heights would then live in mutually rotated frames, destroying
   cross-slice correspondence and with it the entire reconstruction. Now
   passes an explicit `to_2D` transform so planar coords are exactly world
   (x, y).
3. **Slices with holes were accepted.** Only multi-polygon slices were
   rejected; a single polygon with interior rings passed through. Now
   rejected.

The full mesh -> slice -> reconstruct path was additionally validated with
an independent pure-NumPy plane-section implementation on all four test
meshes (classical error 0.019-0.036), so the geometry is confirmed sound
even though trimesh itself could not be run here.

## Step 1 — get real meshes

Ranked by ease:

**Thingi10k** (https://ten-thousand-models.appspot.com) -- large and free;
filter to watertight, then to roughly star-shaped/genus-0 categories
(fruit, bottles, simple vessels). Expect a substantial rejection rate: the
single-closed-loop-per-slice requirement is strict, which is exactly why
you should run `check_mesh_pipeline.py` on the first 10-20 before
committing to a big download.

**ShapeNetCore** -- the `bottle`, `jar`, `vase`, `can`, `mug` categories
match your paper's object family closely. Note many ShapeNet meshes are
NOT watertight; you may need `trimesh.repair` / voxel remeshing first.

**Your own scans** -- phone photogrammetry or a structured-light scanner on
real fruit and pottery. This is the most compelling option for the paper:
it closes the loop with the exact object categories the classical method
was designed around (apple, banana, vase), and gives you genuine
ground-truth meshes rather than synthetic proxies. Even 10-20 scanned
objects would make a strong real-data section.

Put everything under `data/meshes/` (subdirectories are fine, the scanner
recurses).

## Step 2 — build the dataset

```bash
python scripts/check_mesh_pipeline.py --meshes data/meshes --N 9 --limit 20
python scripts/make_mesh_dataset.py --meshes data/meshes --N 5 7 9 15 \
    --out data/real --val_frac 0.15 --test_frac 0.15
```

Output format is identical to the synthetic generator, so training and
evaluation need no changes. The rejection report (counts by reason) is
worth quoting directly in the paper's dataset-construction paragraph.

Cap conventions: real data gives no way to infer whether the circular or
non-circular cap formula is right, so it is a flag --
`--no_base_circular` / `--no_crown_circular` switch to the apple's
convention. Worth trying both on a small subset and keeping whichever the
classical baseline reconstructs better.

## Step 3 — train and evaluate

```bash
python scripts/train_model.py --data data/real --N 9 --m 256 \
    --epochs 100 --surf_sub 8000 --gt_sub 8000 \
    --val_every 5 --val_subset 25 --patience 30 --out runs/real_N9
python scripts/evaluate.py --data data/real --N 9 \
    --ckpt runs/real_N9/best.pt --n_u 32 --out results/eval_real_N9.csv
```

Three experiments worth running, in order of value to the paper:

1. **Train on real, test on real** -- the headline real-data result.
2. **Train on synthetic, test on real** (zero-shot) -- tests whether the
   three synthetic families actually span real geometry. If this holds up
   it is a strong claim: the method needs no real training data.
3. **Train on synthetic then fine-tune on real** -- usually the best
   numbers, and shows synthetic pre-training has value.

Also run `--mode tto --tto_init classical` on a few real objects: that mode
uses no training data at all and is the fair comparison against per-shape
methods like OReX.

## Known constraint to state explicitly in the paper

NSSR (like the classical pipeline it extends) assumes each cross-section is
a single closed loop -- genus-0, star-shaped-ish objects. Meshes with
handles, branching structures, or disconnected components at any height are
out of scope and are rejected at dataset-construction time. Report the
kept/rejected counts; it is a legitimate scope statement, not a weakness,
and it is the same assumption Goodman et al. and Siddiqi & Afzal make.
