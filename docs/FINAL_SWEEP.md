# FINAL SWEEP — the commands to run, and why these settings

Decisions locked in from the 2x2 cap ablation (see the analysis below).
Everything unnecessary has been removed from the codebase, so the flags
listed here are the complete set that matters.

## What the cap ablation settled

| arm | cap_err | vs classical | body_err | vs classical |
|---|---:|---:|---:|---:|
| classical | 0.1163 | — | 0.0169 | — |
| A  w=1.00, heights OFF | 0.0660 | **-43.3%** | 0.0149 | -12.0% |
| **B  w=1.00, heights ON** | **0.0641** | **-44.8%** | 0.0150 | -11.4% |
| C  w=0.25, heights OFF | 0.0671 | -42.3% | 0.0147 | -13.0% |
| D  w=0.25, heights ON | 0.0706 | -39.3% | 0.0148 | -12.6% |

Three conclusions:

1. **The flat caps were never an accuracy problem.** Every learned arm cuts
   cap error ~40-45% below classical. The flattening is the model REDUCING
   cap error; the rounded classical cap is the less accurate one. What
   looked like a defect was the model correcting a classical inaccuracy.
2. **Neither remedy helps; `cap_weight` hurts.** Isolating factors:
   heights ON gives +2.8% at w=1.0 but -5.2% at w=0.25; cap_weight=0.25
   costs -1.7% (heights off) and -10.0% (heights on). Spread across all
   four arms is 10% of the best value against a 44.8% gap from classical --
   all the signal is "learned vs classical", not in these knobs.
   Down-weighting the caps hurt because the caps are where the model was
   making its largest gains.
3. **Arm B is the configuration to use** -- best cap error, tied on body,
   and no extra machinery. It is the default.

Quotable for the paper: the cap/body error ratio falls from **6.9x
(classical) to 4.4x (learned)** -- learning helps most exactly where the
classical construction is weakest.

## Removed from the codebase

* `scripts/train_multigpu.py` -- the per-epoch stepping bug made its results
  invalid, and single-GPU converges in well under an hour.
* `--free_residual` / the `rho` parameter -- an ablation path that was never
  run; it also broke the shape-preserving cone guarantee, so it was a
  liability to keep around.
* `--cap_weight` is retained ONLY so `scripts/ablation_caps.py` can
  reproduce the table above. Do not use it for the main runs: it is a
  documented negative result, default 1.0.

## The full sweep

```bash
# 0. confirm this copy is current and the maths is intact  (~3 min, no GPU)
bash run_all.sh verify

# 1. synthetic data + 4 trainings + evaluation + ablations + designer figs
bash run_all.sh synthetic

# 2. real meshes: fetch, dataset, train, evaluate, visualize
bash run_all.sh real
```

Or step by step, if you prefer to watch each stage — see
`docs/EXPERIMENT_PLAN.md`.

## Flags that matter (the complete list)

**training** — `--N --m --epochs --surf_sub --gt_sub --val_every
--val_subset --patience --init_ckpt --no_learn_heights --cap_weight`
(the last two only for the ablation).

**designer renders** — `--ds --mode {classical,net,tto} --tto_init
{net,classical} --ckpt --freeze_caps --show_crown --tag`.

`--freeze_caps` produces classical-looking caps but MEASURABLY WORSE ones
(~43% higher cap error). Use it only if you want the classical visual
convention in a figure, and say so explicitly; do not present it as the
better model.

## What to send back

- `results/eval_synth_N*.csv` (sparsity sweep)
- `results/ablation_global_N*.csv` (constant-retuning ablation)
- `results/ablation_caps.csv` (full-length rerun of the 2x2)
- `runs/*/log.csv` (training curves; epoch 0 is the classical baseline)
- `results/eval_real_*.csv` (real-data: trained-on-real, zero-shot, finetune)
- `results/real_figs/` including the `pole_*` close-ups
- the designer renders (filenames now encode mode / tto init / freeze_caps)
- the rejection report from `check_mesh_pipeline.py`

With those I can build every table and figure and draft the results and
method sections.


## IMPORTANT: c_bound must be 1.0 (was 2.0) — retrain before final figures

The designer renders from the ablation checkpoints showed the vase's neck
pinching to a point and flaring into a "cone", and a spindle artifact at the
apple's poles. `--freeze_caps` did not help, which ruled out every cap
parameter. Diagnosis, measured directly:

The learned multiplier is bounded by e^{+-c_bound}. At the old default
c_bound=2.0 the network may amplify a tangent by **7.4x**, and an amplified
tangent on a NARROW feature overshoots inward THROUGH the central axis:

| shape | narrowest contour | min interior radius @ s_tau=0 | @ 1.0 | @ 2.0 |
|---|---:|---:|---:|---:|
| vase (neck) | 0.191 | 0.191 | 0.174 | **0.002** |
| apple (poles) | 0.027 | 0.027 | 0.027 | **0.003** |

The pinch-to-a-point IS the surface crossing the axis; the "cone" is the
wireframe on the far side. Chamfer barely notices, because the collapsing
region is small -- which is why the ablation numbers looked healthy while
the pictures were wrong. The falling `C1min` in the training logs
(0.878 -> 0.135) was the same symptom seen from another angle: tangent
magnitudes being driven to the bound.

**Fixes applied**
1. `--c_bound` is exposed on `train_model.py`, `evaluate.py` and
   `reconstruct_designer.py`, **default now 1.0** (2.7x amplification),
   which is safe on all three designer shapes. It MUST match between
   training and inference.
2. `nssr.metrics.axis_clearance` reports the smallest distance from the
   interior surface to the object's LOCAL axis (interpolated per patch, so
   it is valid for bent objects like the banana), relative to the narrowest
   input contour. `reconstruct_designer.py` prints it and warns when the
   ratio drops below 0.10. Validated: all three shapes pass at classical
   and at c_bound=1.0; only the vase's real failure flags at 2.0.

**What this means for the results you already gathered:** the synthetic
numbers (sparsity sweep, global-constant ablation, cap ablation) are
internally consistent and remain valid as reported at c_bound=2.0. Only the
DESIGNER FIGURES are affected. For the final paper, retrain at c_bound=1.0
so the figures and the tables come from the same model, and consider
raising `--reg` if `C1min` still falls sharply.

```bash
python scripts/train_model.py --data data/synthetic --N 9 --m 256 \
    --epochs 100 --c_bound 1.0 --surf_sub 8000 --gt_sub 8000 \
    --val_every 5 --val_subset 25 --patience 30 --out runs/synth_N9_c1
python scripts/reconstruct_designer.py --ds vase --mode net \
    --ckpt runs/synth_N9_c1/best.pt --c_bound 1.0
```

Worth a paragraph in the paper: bounding the multiplier is not merely a
regularizer, it is a GEOMETRIC SAFETY constraint. The admissible
amplification is set by the narrowest feature in the object, and an
aggregate surface metric will not detect a violation.
