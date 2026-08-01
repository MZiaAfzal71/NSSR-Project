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
