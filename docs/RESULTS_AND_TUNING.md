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
