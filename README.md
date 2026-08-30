# NSSR: Neural Shape-Preserving Surface Reconstruction

This repository accompanies the NSSR study, **Neural Shape-Preserving Surface Reconstruction from Sparse Planar Cross-Sections**.  NSSR is a hybrid reconstruction method: a neural network predicts bounded corrections to a deterministic cubic-Hermite surface decoder, while a failure-aware projection can return an output satisfying the repository's sampled Jacobian-orientation and cap turn-back checks.

The repository contains the paper checkpoints, CSV results, publication figures, supplementary PDF, and the code used to create them.  It is intended both for inspecting the reported experiment and for rerunning the synthetic, real-mesh, mixed-domain, and figure-generation workflows.

> **Important scope of the safety result.** `SAFE` in this repository means that the sampled checks implemented in `nssr.safety` pass: the signed-Jacobian orientation check and the base/crown cap turn-back check.  It is not a proof against every possible between-sample or global self-intersection.

## Repository contents

```text
nssr/                       NSSR geometry, network, losses, metrics, and safety code
reference/                  Earlier/reference reconstruction implementations
scripts/                    Dataset, training, evaluation, and figure commands
runs/                       Released trained checkpoints and training logs
results/                    Paper CSVs, figures, LaTeX snippet, and supplementary PDF
```

The main released checkpoints are:

| Experiment | Checkpoint directory | Intended use |
| --- | --- | --- |
| Synthetic full sweep | `runs/paper_full_100ep/N{9,11,15}/best.pt` | Designer shapes and synthetic table |
| Real-only sweep | `runs/paper_real_100ep/N{9,11,15}/best.pt` | Held-out real-mesh table and figures |
| Domain-balanced mixed sweep | `runs/paper_mixed_100ep/N{9,11,15}/best.pt` | Mixed-to-real and mixed-to-synthetic comparisons |

`best.pt` is the safety-aware selected checkpoint.  `best_accuracy.pt`, `best_safe.pt`, `last.pt`, and `log.csv` are also retained for audit and training-history inspection.

## Installation

Use Python 3.10 or newer.  Create an isolated environment and install the minimum-version dependency set from `requirements.txt`.

```bash
git clone https://github.com/MZiaAfzal71/NSSR-Project.git
cd NSSR-Project

python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` includes the optional `thingi10k[clip]` dependency used by the corpus-download script.  If you only need the synthetic workflow, it may be omitted from a local installation.  For a CUDA-enabled PyTorch build, follow the wheel-selection command at [pytorch.org](https://pytorch.org/get-started/locally/) and then rerun `python -m pip install -r requirements.txt` if needed.

The scripts automatically use CUDA when `torch.cuda.is_available()` is true; CPU execution is supported but training and high-resolution rendering will be slower.

## Data availability and conventions

The released repository does **not** include `data/synthetic`, `data/real`, `data/mixed`, or a raw mesh collection.  These can be large, may be regenerated, and may have external-license conditions.  Create them using the commands below.

Every dataset directory must contain matching pickles for each slice count:

```text
train_N9.pkl     val_N9.pkl     test_N9.pkl
train_N11.pkl    val_N11.pkl    test_N11.pkl
train_N15.pkl    val_N15.pkl    test_N15.pkl
```

Here `N` is the number of ordered planar cross-sections.  The paper's primary visual examples use the released `N=15` checkpoints.

## Fast path: inspect the released paper results

No data generation or retraining is needed to inspect the committed tables, figures, checkpoints, or logs.

```bash
# Synthetic multi-N summary used for the main synthetic results table.
column -s, -t < results/paper_full_100ep/summary.csv

# Real-only / synthetic-transfer / mixed-domain comparison.
column -s, -t < results/paper_real/domain_comparison.csv

# N=15 cap-specific analysis.
column -s, -t < results/paper_real/base_crown_N15.csv

# Open the supplementary document (choose the command for your system).
open results/NSSR_supplementary.pdf       # macOS
xdg-open results/NSSR_supplementary.pdf   # Linux
start results/NSSR_supplementary.pdf      # Windows Command Prompt
```

The aggregation command can be rerun safely from the committed per-object CSVs:

```bash
python scripts/collect_results.py \
  --results results/paper_full_100ep \
  --out results/paper_full_100ep/summary.csv
```

## Reproduce the synthetic multi-N experiment

The recommended command is the orchestrator `run_full_sweep.py`.  It uses the frozen safety-aware settings, trains one model per `N`, runs the common validator and projection evaluation, then writes a single `summary.csv`.

### 1. Generate a synthetic dataset

```bash
python scripts/make_synthetic_dataset.py \
  --n_train 800 --n_val 100 --n_test 100 \
  --slices 9 11 15 \
  --out data/synthetic
```

### 2. Train, validate, project, and collect results

Use fresh output directories to preserve the released artifacts:

```bash
python scripts/run_full_sweep.py \
  --data data/synthetic \
  --Ns 9 11 15 \
  --epochs 100 \
  --runs runs/repro_synthetic_100ep \
  --results results/repro_synthetic_100ep
```

This writes the following artifacts for each `N`:

```text
runs/repro_synthetic_100ep/N15/best.pt
results/repro_synthetic_100ep/N15_validate.csv
results/repro_synthetic_100ep/N15_projection.csv
results/repro_synthetic_100ep/summary.csv
```

Use `--dry_run` to print the subprocess commands without executing them, `--force` to rerun stages whose output already exists, and repeat `--ckpt N=PATH` to reuse a known checkpoint instead of training it again.

### Run individual stages

These commands are useful for a checkpoint already trained with matching settings:

```bash
# Raw classical-versus-learned accuracy and sampled safety.
python scripts/validate.py \
  --data data/synthetic --split test --N 15 \
  --ckpt runs/repro_synthetic_100ep/N15/best.pt \
  --m 128 --n_u 16 --c_bound 1.0 --max_cap_fold 1e-3 \
  --out results/repro_synthetic_100ep/N15_validate.csv

# Raw and post-projection safety/accuracy.  "staged" is the paper policy.
python scripts/evaluate_projection.py \
  --data data/synthetic --split test --N 15 \
  --ckpt runs/repro_synthetic_100ep/N15/best.pt \
  --m 128 --n_u 16 --c_bound 1.0 --max_cap_fold 1e-3 \
  --projection_mode staged \
  --out results/repro_synthetic_100ep/N15_projection.csv

# Aggregate all N-specific validation/projection CSVs into one table.
python scripts/collect_results.py \
  --results results/repro_synthetic_100ep \
  --out results/repro_synthetic_100ep/summary.csv
```

## Reproduce one reconstruction

`reconstruct.py` exports both a JSON safety report and an `.npz` package containing the reconstructed surface and inputs.  It is a convenient single-object debugging and reproducibility entry point.

```bash
python scripts/reconstruct.py \
  --data data/synthetic --split test --N 15 --index 0 \
  --mode net --ckpt runs/repro_synthetic_100ep/N15/best.pt \
  --m 128 --n_u 32 --project_safe \
  --out results/reconstruction_N15_0
```

The projection is applied only when the raw network output fails the sampled checks.  The JSON report records the raw and final states, projection stage, and retained correction scale `alpha`.

For the original Banana, Apple, and Vase designer workflow, use:

```bash
python scripts/reconstruct_designer.py \
  --ds banana --mode net \
  --ckpt runs/paper_full_100ep/N15/best.pt \
  --out results/designer --tag N15
```

Set `--ds apple` or `--ds vase` for the other shapes.  Use `--mode classical` to render the deterministic baseline and `--mode tto` only for the script's test-time-optimization experiment.

## Real-mesh and mixed-domain workflow

### 1. Obtain or provide meshes

You may use your own watertight `OBJ`, `PLY`, `STL`, or `OFF` meshes, or first create known-good test meshes:

```bash
python scripts/make_test_meshes.py --out data/meshes_test
python scripts/check_mesh_pipeline.py \
  --meshes data/meshes_test --N 15 --m 128 --axis_select search
```

To fetch a new Thingi10K corpus, install the optional package, respect its terms of use, and run:

```bash
python scripts/fetch_thingi10k.py --out data/meshes --limit 150
python scripts/diagnose_thingi10k.py --variant tetwild --query "a vase"
```

### 2. Create real-data pickles

```bash
python scripts/make_mesh_dataset.py \
  --meshes data/meshes --N 9 11 15 \
  --out data/real --axis_select search \
  --val_frac 0.15 --test_frac 0.15 --seed 0
```

`data/real/mesh_manifest.csv` records every accepted/rejected mesh and the assigned train/validation/test split.  The pipeline accepts one usable closed contour loop per retained section; meshes that do not meet this condition are recorded as rejected.

### 3. Run the paper-oriented real/mixed pipeline

Run the synthetic full sweep first, because zero-shot transfer uses its checkpoints.  Then execute:

```bash
python scripts/run_real_paper_pipeline.py \
  --meshes data/meshes --real data/real --synthetic data/synthetic \
  --mixed data/mixed --Ns 9 11 15 --epochs 100 \
  --real_runs runs/repro_real_100ep \
  --mixed_runs runs/repro_mixed_100ep \
  --synthetic_runs runs/repro_synthetic_100ep \
  --results results/repro_real \
  --figures
```

To let this command build the real dataset itself, add `--build_real`; add `--check_meshes` to run the preflight check first.  It performs the following sequence:

1. real-only training and evaluation;
2. synthetic-to-real zero-shot evaluation;
3. domain-balanced synthetic-plus-real training;
4. mixed-to-real and mixed-to-synthetic evaluation on untouched test sets;
5. optional four-panel real-object figures and `domain_comparison.csv`;
6. supplementary-PDF assembly, unless `--skip_supplementary` is supplied.

The mixed dataset's `test_N*.pkl` is diagnostic only.  The manuscript's mixed-domain claims must be evaluated separately on the untouched `data/real/test_N*.pkl` and `data/synthetic/test_N*.pkl` files, which `run_real_paper_pipeline.py` does automatically.

## Recreate the manuscript figures

The main qualitative figures are generated by `make_paper_qualitative_figures.py`.  It creates PDF and PNG files plus a LaTeX insertion snippet.  With the released assets, the following regenerates the `N=15` designer, real, and combined-gallery figures:

```bash
python scripts/make_paper_qualitative_figures.py \
  --N 15 \
  --designer-ckpt runs/paper_full_100ep/N15/best.pt \
  --real-ckpt runs/paper_real_100ep/N15/best.pt \
  --real-source existing \
  --formats pdf png \
  --out results/paper_qualitative_figures
```

`--real-source existing` uses the committed four-panel real-object images when raw real-data pickles are unavailable.  Use `--real-source reconstruct --real-data data/real` to reconstruct those panels from the dataset and checkpoint.  Add `--check` to report all resolved files and dependencies without rendering.

| Output | Manuscript role |
| --- | --- |
| `results/paper_qualitative_figures/figure_designer_comparison_N15.pdf` | Main designer-shape comparison: Banana, Apple, Vase |
| `results/paper_qualitative_figures/figure_real_comparison_N15.pdf` | Main held-out real-mesh comparison |
| `results/paper_qualitative_figures/figure_qualitative_combined_N15.pdf` | Optional combined gallery / graphical abstract |
| `results/paper_qualitative_figures/qualitative_figures_N15.tex` | Ready-to-adapt LaTeX figure environments and captions |

The combined gallery is intentionally supplementary/optional: the separate designer and real figures are the clearer main-paper presentation.

## Generate the supplementary PDF

```bash
python scripts/make_supplementary_pdf.py \
  --synthetic_summary results/paper_full_100ep/summary.csv \
  --domain_summary results/paper_real/domain_comparison.csv \
  --real_figs results/paper_real/figures \
  --out results/NSSR_supplementary.pdf
```

The script combines summary tables with real, synthetic, and designer supplementary figures.  Use `--max_figures` to limit the number of image panels included.

## Results-to-manuscript map

| Repository artifact | What it contains | Manuscript use |
| --- | --- | --- |
| `results/paper_full_100ep/summary.csv` | Synthetic held-out multi-N accuracy, raw safety, post-projection safety, activation, and retention statistics | Main synthetic quantitative table and (N=15) selection rationale |
| `results/paper_full_100ep/N*_validate.csv` | Per-object classical and raw learned metrics/safety | Audit trail for the synthetic table |
| `results/paper_full_100ep/N*_projection.csv` | Per-object raw/post-projection outcomes, stage, and `alpha` | Projection analysis and safety claims |
| `results/paper_real/real_only/summary.csv` | Real-only held-out results | Real-data quantitative table |
| `results/paper_real/synthetic_to_real/` | Synthetic-trained checkpoints evaluated on real test meshes | Zero-shot transfer comparison |
| `results/paper_real/mixed_to_real/` | Mixed checkpoints evaluated on real test meshes | Mixed-domain real generalization |
| `results/paper_real/mixed_to_synthetic/` | Mixed checkpoints evaluated on synthetic test objects | Mixed-domain synthetic retention |
| `results/paper_real/domain_comparison.csv` | Compact domain/training/test comparison across N values | Main domain-comparison table |
| `results/paper_real/base_crown_N15.csv` | Base/crown/full-surface errors and cap behaviour | Cap-focused supplementary analysis |
| `results/NSSR_supplementary.pdf` | Tables and expanded visual evidence | Supplementary material |

### Reading the principal metrics

- `classical_chamfer_l2`, `learned_chamfer_l2`, and `post_chamfer_l2` compare baseline, raw NSSR, and projected NSSR reconstruction error; lower is better.
- `raw_j_valid_rate` and `raw_cap_safe_rate` are the rates passing the two sampled safety components before projection.
- `raw_safe_rate` requires both components to pass.
- `post_safe_rate` is the corresponding rate after failure-aware projection.
- `projection_activation_rate` is the fraction of test objects requiring projection.
- `alpha_*` reports how much of the corrected parameter group is retained for projected objects; one means no reduction.
- `stage_cap_all_*`, `stage_tangent_*`, and `stage_all_*` identify the repair route selected by the failure-aware projection.

## Useful utilities

```bash
# Cap-patch analysis for one real-trained N=15 checkpoint.
python scripts/analyze_base_crown.py \
  --data data/real --split test --N 15 \
  --ckpt runs/paper_real_100ep/N15/best.pt \
  --project_safe --out results/base_crown_N15.csv

# Render a small set of real examples, including pole zooms.
python scripts/visualize_real.py \
  --data data/real --N 15 --m 128 --n_u 16 \
  --ckpt runs/paper_real_100ep/N15/best.pt \
  --n 8 --project_safe --pole_zoom --surface_render \
  --out results/real_figs_N15

# Global-constant classical-baseline ablation.
python scripts/ablation_global_constants.py \
  --data data/synthetic --N 15 \
  --ckpt runs/paper_full_100ep/N15/best.pt \
  --out results/ablation_global.csv
```

## Reproducibility notes

- Run commands from the repository root so imports such as `nssr.*` resolve correctly.
- Keep `m`, `n_u`, `c_bound`, and `max_cap_fold` consistent between training, validation, projection, and figure generation.  The paper pipeline uses `m=128`, training `n_u=12`, evaluation `n_u=16`, `c_bound=1.0`, and `max_cap_fold=1e-3`.
- Do not use test objects for checkpoint selection.  `best.pt` is selected during training/validation; the CSV files report held-out test evaluations.
- Use the orchestrator scripts rather than manually combining commands when reproducing paper tables, because they preserve the same validation and projection settings across `N=9,11,15`.
- The checkpoints and outputs are provided for transparency.  Exact reruns can vary slightly with PyTorch, CUDA, and hardware versions; record your environment and random seeds for a new comparison.

## License

See [LICENSE](LICENSE).  Meshes obtained from external sources, including Thingi10K, remain subject to their own licenses and terms.
