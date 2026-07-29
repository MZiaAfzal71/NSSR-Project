# First result and tuning guide

## Your first run (5 epochs, m=128, single T4)

| epoch | val Chamfer-L2 | val Hausdorff | C1 min |
|------:|---------------:|--------------:|-------:|
| 0 (classical) | 0.01013 | 0.393 | 1.000 |
| 5 (learned)   | 0.00425 | 0.386 | 0.860 |

**Interpretation.** Learning the pipeline's free parameters cut held-out
Chamfer error by ~58% in five epochs while preserving the C1 guarantee
(min tangent magnitude stayed ~0.86, far from the cusp at 0). This is the
paper's central claim, already visible: guarantees + data-driven priors,
better than either alone. The classical row is literally your published
method (parity-verified), so the gap is a clean, attributable measurement.

**What to watch next.**
- *Hausdorff barely moved while Chamfer halved.* The network improves the
  typical fit but not the worst-case point — usually a few hard regions
  (cap junctions, highest-curvature slice). Turning on / up the normal loss
  and inspecting per-region error heatmaps is the natural follow-up; also
  report Hausdorff-95 (already computed) which is less outlier-sensitive.
- *train_loss (0.0077) > val_chamfer (0.00425)* is expected: train_loss
  includes normal + reg + smoothness terms. Compare epoch-0 val_chamfer to
  later val_chamfer for the honest classical-vs-learned number.

## Memory (why m=256 blew past 15 GB, and the fix)

Peak memory was the Chamfer distance matrix: surface points P (=
n_patches * n_u * m) times GT points Q, materialized at once. At m=256,
n_u=24, N=9 that is ~59k x 20k floats per object.

`nssr/losses.py` now uses a **double-chunked** nearest-neighbour search
(both sides tiled), so peak memory is ~ chunk_q * chunk_t (default
4096 x 16384) regardless of m or Q -- provably identical results to brute
force (verified). Training also **subsamples the predicted surface**
(`--surf_sub`, default 20000) for an unbiased, cheaper gradient. Evaluation
caps point counts at 200k. Net effect: m=256 fits comfortably in one T4.

Knobs if you still hit limits: lower `--surf_sub`/`--gt_sub` (e.g. 10000),
lower `--n_u` during training (raise it only for final rendering/eval),
lower `--m`. None of these change the method, only the sampling density of
the loss.

## Using both T4s on Kaggle

`scripts/train_multigpu.py` shards each epoch's objects across both GPUs,
all-reduces (averages) gradients, and steps identical model copies —
object-level data parallelism, which fits NSSR because objects are
independent. Roughly ~2x throughput. Single-GPU is automatic if only one
device is visible.

```bash
!python scripts/train_multigpu.py --data data/synthetic --N 7 \
    --epochs 200 --m 256 --out runs/exp1
```

If you prefer the simple single-GPU script, it now also accepts
`--surf_sub`, so m=256 works there too:

```bash
!python scripts/train_model.py --data data/synthetic --N 7 \
    --epochs 200 --m 256 --out runs/exp1
```

## Suggested next experiments (in order)

1. **Full run at N=7, m=256, ~200 epochs.** Establish the real
   classical-vs-learned gap with converged training. Save `best.pt`.
2. **Sparsity sweep.** Retrain / evaluate at N in {5,7,9,15}. Hypothesis:
   learned tangents help most at small N. This is a headline figure.
3. **`scripts/evaluate.py`** on the test split -> the main comparison table
   (classical vs learned; Chamfer, Hausdorff, normal consistency, C1 min).
4. **Designer generalization.** `scripts/reconstruct_designer.py --ds vase
   --mode net --ckpt runs/exp1/best.pt` — trained-on-synthetic applied to
   your paper's shapes. Also `--mode tto` for the training-data-free result.
5. **Ablations** (METHOD.md sec. 7): free-residual vs cone-constrained;
   each learnable group off; normal-loss on/off; reg strength sweep.
6. **OReX** on the same inputs for the headline external comparison.

## Speeding up training (from ~2-3 min/epoch)

Your GPU showed 80%+ utilization, so it is compute-bound on the Chamfer
nearest-neighbour search, whose cost is proportional to surf_sub * gt_sub.
That subsample size is the dominant lever.

Changes made (all opt-in via flags; defaults now favor speed):
- **surf_sub / gt_sub default 20000 -> 8000.** Unbiased estimator; ~6x less
  Chamfer compute. This is the single biggest win.
- **--val_every (default 5).** Validation used to run every epoch on all 100
  val objects at full resolution. Now every 5th epoch (and the last).
- **--val_subset K.** Validate on the first K val objects during training
  (use the full test set once at the end via scripts/evaluate.py).
- **--eval_n_u.** Lower n_u for validation only; raise it for final render.
- **--patience P.** Early stop after P epochs with no val improvement.

Estimated wall-clock (rough; verify on your box) for 200 epochs:

| config | ~sec/epoch | 200 epochs |
|---|---:|---:|
| old default (20000, val every epoch) | ~137 | ~7.6 h |
| surf/gt 8000, val_every 5 | ~25 | ~1.4 h |
| + val_subset 20 | ~24 | ~1.4 h |
| surf/gt 6000, n_u 20 | ~13 | ~0.7 h |
| 8000 + 2x T4 (train_multigpu) | ~12 | ~0.7 h |

### Recommended commands

Fast single-GPU run (should be ~1.5 h for 200 epochs):
```bash
!python scripts/train_model.py --data data/synthetic --N 7 --m 256 \
    --epochs 200 --surf_sub 8000 --gt_sub 8000 \
    --val_every 5 --val_subset 25 --patience 40 --out runs/exp1
```

Both T4s (roughly halves it again):
```bash
!python scripts/train_multigpu.py --data data/synthetic --N 7 --m 256 \
    --epochs 200 --surf_sub 8000 --gt_sub 8000 \
    --val_every 5 --val_subset 25 --out runs/exp1
```

With --patience 40 the run will usually stop well before 200 epochs once
val Chamfer plateaus, so real time is often lower. Do the final,
full-resolution, full-test-set evaluation ONCE at the end:
```bash
!python scripts/evaluate.py --data data/synthetic --N 7 \
    --ckpt runs/exp1/best.pt --n_u 32 --out results/eval_N7.csv
```

### Fidelity note for the paper

surf_sub/gt_sub affect only the LOSS sampling density during training, not
the method or the final surface. Report final metrics from scripts/evaluate.py
at full resolution (it caps at 200k points for memory but is otherwise full).
The training subsample is a standard, unbiased Monte-Carlo estimate of
Chamfer -- reviewers expect this in geometry-learning work.

### If you later become memory- or launch-bound instead

If GPU utilization drops (spiky), the bottleneck shifts to per-object Python
overhead; then the fix is batching several small objects' Chamfer together
or moving to the 2x T4 script. For now (compute-bound), subsample size wins.

## Designer-shape bugs found and fixed (post-sparsity-sweep)

Three distinct issues surfaced when running `reconstruct_designer.py` on the
CiSE shapes. Only two were code bugs; the third is a genuine finding.

**1. `vase --mode tto` crashed with NaN loss.** The vase's raw data has
slices 0 and 2 at the identical height (z=0.75 raw), straddling the flat
base ring (index 1) -- a deliberate non-monotone test case from the paper.
Leaving out index 1 during leave-one-out makes indices 0 and 2 adjacent in
the subset with an EXACT zero height gap, causing a division by zero in
the boundary-direction formula (Eq. 22-24's beta1/beta2). Fixed by clamping
that denominator away from zero (`nssr/geometry.py` and `geometry_np.py`);
verified this does not change any well-separated case (parity check still
passes to ~1e-11) and that every leave-one-out fold for the vase now
produces a finite surface.

**2. `apple --mode tto` collapsed both caps into a point.** The leave-one-out
loss used a standard TWO-SIDED Chamfer distance between the entire
reconstructed surface and the single held-out ring. The "surface -> ring"
side of that distance penalizes every point on the whole surface --
including the opposite cap -- for not being near that one small ring;
summed over every fold, gradient descent minimizes this by shrinking the
whole object toward the interior, collapsing both caps into a cone. Fixed
in `scripts/reconstruct_designer.py` by switching to a ONE-SIDED distance
(ring -> surface only: "does the reconstruction pass near this ring",
with no penalty on the rest of the surface for being elsewhere). Since the
contour data R is fixed (not optimized), the surface can't run away to
compensate, so this one-sided loss doesn't have an equivalent degenerate
solution.

**3. `banana --mode net` showed a spurious neck near the crown.** This is
NOT a code bug -- the network architecture (circular conv + geometric
features) is size- and scale-agnostic and was checked directly. This is a
genuine out-of-distribution generalization limitation: the checkpoint was
trained only on synthetic generalized-cylinder objects, and the banana's
crown ring geometry differs enough from anything seen in training that the
crown-scaling prediction misbehaves. Worth reporting as-is in a limitations
section, or as a target for a brief per-shape fine-tune (a handful of `tto`
steps initialized from the trained network, rather than from classical,
would test whether a small correction closes the gap).

Re-run all three designer modes after pulling the fix; the vase should no
longer NaN and the apple should keep its dimpled caps instead of collapsing.

## Plan change: verify banana/apple/vase before gathering final results

Priority shifted to: make sure the trained model regenerates the three
CiSE designer shapes as accurately as possible, THEN gather results, THEN
move to real-world data. Three things changed to support this.

### 1. Vase crown correction (accuracy fix, not cosmetic)

An earlier version of this project set `closed_top=False` for the vase,
skipping the crown patch entirely on the theory that "the top is open."
That was wrong: `tests/parity_check.py` proves the reference implementation
computes a full, circular crown patch for the vase identically to the
base -- the paper's Figure 1c simply doesn't DRAW it (the reconstruction
script's wireframe loop stops one patch early). Treating the crown as
computationally absent was a real accuracy loss.

Fixed: `preprocess_designer` now always sets `closed_top=True` (matching
the reference exactly, re-verified by parity_check to ~1e-11). Hiding the
crown in a picture is now purely a rendering choice:
```bash
python scripts/reconstruct_designer.py --ds vase --mode classical
# crown hidden by default (matches Fig 1c); underlying surface has it
python scripts/reconstruct_designer.py --ds vase --mode classical --show_crown
# same surface, crown patch drawn too
```

### 2. Synthetic training data now covers three geometric families

The previous synthetic generator only produced smooth convex blobs with
monotonic height -- a training distribution that never included the
non-monotone, non-circular-cap, or open-top geometry the three designer
shapes actually present. That's very likely why `--mode net` (a network
trained on the old data) produced the banana crown artifact: genuine
out-of-distribution input.

`nssr/synthetic.py` now draws each training object from one of three
families:
- **standard** (50%): the original convex-blob generator, with an
  increased bend probability/magnitude for elongated, curved objects --
  the BANANA-like case.
- **dimpled** (25%): height is a deliberately NON-MONOTONIC function near
  one or both poles (a calibrated local "fold-back"), so adjacent interior
  slices can have inverted height order -- the same stress case the paper
  explicitly designed the APPLE around ("non-monotone cross sections...
  inward-dipping profiles of an apple"). Calibrated empirically (zero
  endpoint-monotonicity violations across 3000+ trials; ~35-40% of dimpled
  samples get a genuine interior reversal) so the TRUE first/last slice
  stays well-behaved while interior slices fold, exactly mirroring the
  apple's own structure.
- **open_top** (25%): no crown contour at all (`closed_top=False`), wide
  un-tapered rim -- the case a genuinely open real-world scan (a bowl, a
  broken-rim vase) would present, as distinct from the paper's vase (which
  DOES have crown data, see point 1 above).

Independently of family, `base_circular` / `crown_circular` are randomized
(~80% True) so the network also sees the apple's non-circular-cap
convention on other shapes, not just non-monotone height.

Verify with `python tests/synthetic_family_check.py` (family mix, zero
endpoint violations, zero NaN/Inf, correct patch counts for closed_top).
Regenerate the dataset before retraining:
```bash
python scripts/make_synthetic_dataset.py --n_train 800 --n_val 100 \
    --n_test 100 --slices 5 7 9 15 --out data/synthetic
```
The printed family mix per split is worth keeping for the paper's dataset
description.

### 3. Verification gate before gathering final results

Recommended order, each step gating the next:

1. `python tests/synthetic_family_check.py`, `numpy_sanity_check.py`,
   `parity_check.py` -- all PASS (no torch needed, run anywhere).
2. Regenerate the synthetic dataset (above).
3. Retrain: `python scripts/train_model.py --data data/synthetic --N 7 \
   --m 256 --epochs 100 --surf_sub 8000 --gt_sub 8000 --val_every 5 \
   --val_subset 25 --patience 30 --out runs/exp_v2`
4. **Visual gate** -- for each of the three shapes, all three modes:
   ```bash
   for ds in banana apple vase; do
     for mode in classical net tto; do
       python scripts/reconstruct_designer.py --ds $ds --mode $mode \
           --ckpt runs/exp_v2/best.pt
     done
   done
   ```
   `classical` should look identical to before (parity-verified, unaffected
   by any of this). `tto` should no longer show the apple cone-collapse or
   the vase NaN. `net` is the real test of point 2 above -- share the nine
   images here before moving on; if `net` still misbehaves on a shape, that
   tells us the family mix or a specific geometric parameter needs another
   pass, which is exactly what this gate is for.
5. Only once all nine look right: proceed to the full sparsity sweep,
   ablations, and evaluation tables for the paper.

### Real-world data (after the gate passes)

`scripts/make_mesh_dataset.py` (new) scans a directory of meshes, slices
each one the same way as the designer shapes, keeps only watertight
meshes where every slice is a single closed loop, reports rejection
reasons (quotable in the paper's dataset section), and writes
train/val/test pickles in the exact same format the synthetic pipeline
uses -- `train_model.py` and `evaluate.py` need no changes:
```bash
python scripts/make_mesh_dataset.py --meshes data/meshes --N 5 7 9 15 \
    --out data/real --val_frac 0.15 --test_frac 0.15
python scripts/train_model.py --data data/real --N 7 --out runs/real_v1
```

Candidate sources, roughly in order of ease:
- **Thingi10k** (https://ten-thousand-models.appspot.com) -- large, free,
  many watertight prints; filter to roughly star-shaped/genus-0 categories
  (fruit, bottles, simple vases/vessels) since that's the object class the
  classical pipeline (and NSSR) assumes.
- **ShapeNetCore** -- `bottle`, `jar`, `vase`, `can`, `mug` categories are
  a natural match to your paper's own object family.
- **Your own scans** -- a phone photogrammetry app (e.g. any structured-
  from-motion tool) or a structured-light scanner on real fruit or pottery
  gives you a genuine ground-truth mesh AND a direct visual/quantitative
  comparison against your own apple/banana/vase results -- probably the
  most compelling real-world validation for the paper, since it closes the
  loop with the exact object categories the method was designed around.

Start with a small batch (10-20 meshes) through `make_mesh_dataset.py` to
confirm the rejection rate is reasonable before committing to a larger
scrape -- genus-0/single-loop-per-slice is a real constraint and it's
better to learn the yield early.

## TTO fix #2: per-point noise causing self-intersecting "loops"

After fixing the collapse bug (one-sided loss), a DIFFERENT artifact
appeared: self-intersecting loops on the apple's caps and mild zigzag
waviness on the banana's interior, while the reported loss stayed low in
both cases (the loss doesn't penalize this at all, which is exactly why
it wasn't caught by the numbers).

Root cause: `tto_leave_one_out` gave every individual point around every
ring its own free parameter (a full `(N, m)` tensor per field: s_a, s_b,
s_tau), with only a weak L2-to-zero pull and no coupling between
neighbouring points. The network-based ("net") mode never has this
problem because the circular-conv architecture structurally forces nearby
points to behave similarly; tto's raw per-point tensors have no such
inductive bias, and with 400 optimization steps individual columns can
drift independently. Each leave-one-out fold only supervises against ONE
ring at a time -- nowhere near enough signal to justify point-by-point
independent freedom within a ring, so the extra degrees of freedom were
pure overfitting surface.

Fixed structurally, not with a stronger regularizer: s_a/s_b/s_tau are now
ONE SCALAR PER ROW (broadcast uniformly around the ring, `(N, 1)` instead
of `(N, m)`), and s_fB/s_fC are one scalar per object. This removes the
ability to represent circumferential noise at all, rather than merely
discouraging it. Also added gradient-norm clipping (matching the main
training loop's practice), which the original tto optimizer lacked.
Re-run `--mode tto` for all three shapes after pulling this fix.

## TTO fix #3 (the real one): optimize a FUNCTION, not per-row numbers

Symptom: with `classical` and `net` both looking correct on all three
designer shapes, `--mode tto` still badly distorted the apple and vase
(exploded caps, displaced rings) while the banana looked fine -- and the
reported loss was low throughout.

Root cause (a design flaw in the tto formulation, not a coding slip):
the optimized parameters were indexed BY ROW (`s_a[i]`, one knob per
contour). Every leave-one-out fold removes a row, so the surviving rows
sit across LARGER vertical gaps than they do in the real object. The
optimizer therefore tuned each row's tangent scaling to be correct for
those widened gaps -- then the final render applied those same numbers to
the FULL object, where the gaps are back to normal, so the tangents
systematically overshoot.

Why the banana was fine and the other two were not: the banana has 7 rows,
smooth and monotonic in Z, so removing one row barely changes local
geometry. The apple and vase both have NON-MONOTONIC Z with tightly-spaced
rows, where dropping a row changes the local configuration drastically --
so the subset-vs-full mismatch is severe exactly there. (An earlier
hypothesis -- per-point circumferential noise -- was wrong; restricting
the parameters to one scalar per row did not fix it, which is what pointed
at the row-indexing itself.)

Fix: tto now optimizes the WEIGHTS OF A ParamNet -- a function from local
geometry to tangent corrections -- instead of free per-row numbers. A
geometry-conditioned function is evaluated on whatever configuration it is
handed, so the subset folds and the final full-object evaluation stay
coherent. This is precisely why `--mode net` transfers cleanly, and tto
now inherits that property.

Two operating modes:
```bash
# per-object fine-tune of the trained checkpoint (default, usually best)
python scripts/reconstruct_designer.py --ds apple --mode tto \
    --ckpt runs/exp_v2/best.pt --tto_init net

# fully training-data-free: starts from the classical pipeline, uses no
# ground truth and no training set at all -- the fair comparison against
# per-shape methods like OReX
python scripts/reconstruct_designer.py --ds apple --mode tto \
    --tto_init classical
```
Both are worth reporting in the paper: the first shows how much a few
per-object steps add on top of a trained model; the second is the
strongest answer to "where does training data come from in practice?".

## TTO fix #4: the loss could be satisfied by the WRONG patch

After fix #3 (optimize a geometry-conditioned function, not per-row
numbers), the apple's tto result became essentially correct, but the vase
was still badly distorted -- huge flared cones at both ends -- in BOTH
`--tto_init net` and `--tto_init classical`. Two further causes, both
verified numerically rather than guessed:

**(a) Cap-cheating.** The one-sided loss searched the ENTIRE reconstructed
surface for each held-out ring point's nearest neighbour. Checked directly:
for the vase's i=7 fold, the nearest patch to the held-out ring is the
CROWN CAP (patch 8), not the gap patch that should be interpolating it. So
the optimizer could lower the loss by ballooning a cap outward until it
swept through the target ring -- low loss, destroyed shape. Same class of
degeneracy as the original collapse bug, in the opposite direction.

Fixed: the loss now targets ONLY the gap patch -- the single patch that
actually spans the hole left by the removed row. Patch order is
`[base cap, interior 1..Nsub-1, crown cap]` and interior patch k spans
subset rows k-1,k; removing row i makes original rows i-1,i+1 into subset
rows i-1,i, so the required patch is at list index i (verified empirically
for all three shapes). Caps are now excluded from the loss entirely and
cannot be exploited.

**(b) An unsatisfiable fold.** Removing the vase's row 1 leaves rows 0 and
2 at IDENTICAL height (both -1.104, straddling its flat base ring -- the
duplicate-height quirk that also caused the earlier NaN). The resulting gap
patch has zero vertical extent but is asked to pass through a ring at
z=-1.186, BELOW both -- geometrically impossible, so the fold contributed
nothing but large gradients. Fixed: folds whose neighbouring rows sit at
(near-)identical heights are detected and skipped, with a printed notice.
For the vase this skips exactly fold 1 and keeps folds 2-7; banana and
apple skip nothing.

Note for the paper: the leave-one-out fold quality varies a lot on the
vase (gap-patch distance at classical init ranges 0.004 to 0.285 across
folds) because its slices are very unevenly spaced. That is worth
mentioning as a property of test-time optimization on irregular
cross-sectional data -- it is exactly the regime where the self-supervised
signal is weakest.

## TTO fix #5: the caps are NEVER constrained by leave-one-out

The decisive finding, obtained by enumerating which full-object patches the
objective actually touches:

```
vase (N=9) usable folds=[2,3,4,5,6,7]
   base_cap           *** NEVER CONSTRAINED ***
   interior(0,1)      *** NEVER CONSTRAINED ***   (its only fold is the
                                                   skipped degenerate one)
   interior(1,2) ... interior(7,8)   supervised
   crown_cap          *** NEVER CONSTRAINED ***
```

No leave-one-out fold ever targets a cap -- folds only ever supervise
interior gap patches. In fix #3 the regularizer was also set to zero, so
NOTHING anchored the cap outputs: the shared network weights were pushed
hard by the interior folds, and the cap predictions drifted wherever those
weights happened to take them. That is the exploding cone -- it was never
being optimized toward anything, it was unconstrained drift. The vase is
the worst case because it loses `interior(0,1)` as well (its only fold is
the degenerate skipped one), giving it TWO unconstrained regions.

Two fixes:
1. **Freeze the cap scalings** (`s_fB`, `s_fC`) at classical during tto.
   They receive literally zero data signal from this objective, so there
   is nothing to learn and everything to lose by letting them move.
2. **Proximity-to-classical anchor** (`--tto_prox`, default 1e-2) on the
   predicted `s_a`/`s_b`/`s_tau`. Supervised regions still move freely --
   the data term dominates there -- while unsupervised regions stay at the
   classical solution instead of drifting.

Also added: tto now prints per-row `|s|` magnitudes at the end. `s` is
bounded to +-2 (tangent weights within e^{+-2}); values pinned near 2.0,
especially on rows the folds never supervise, mean the anchor is too weak
and `--tto_prox` should be raised. Please paste this diagnostic if the
distortion persists -- it localizes the problem to specific rows.

### Conceptual note for the paper

This is worth stating explicitly in the method section: leave-one-out
self-supervision constrains ONLY the interior interpolation, because caps
have no held-out ring to be validated against. Test-time optimization can
therefore improve interior fidelity but cannot, even in principle, learn
anything about cap behaviour without ground truth. That is a real and
honest limitation of the training-data-free mode, and a clean argument for
why the trained-network mode (which learns cap behaviour from the
synthetic corpus) is the stronger operating point.
