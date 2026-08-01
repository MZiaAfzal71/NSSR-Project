#!/usr/bin/env bash
# NSSR: run every experiment in order. See docs/EXPERIMENT_PLAN.md for
# details, expected runtimes, and what each phase produces.
#
#   bash run_all.sh verify      # phase A only (fast, no GPU)
#   bash run_all.sh synthetic   # phases B-F (synthetic + designer shapes)
#   bash run_all.sh real        # phases G-H (needs trimesh + thingi10k)
#   bash run_all.sh all
set -e
mkdir -p results runs

phase_verify() {
  echo "=== A: verification ==="
  python scripts/verify_fixes.py
  python tests/numpy_sanity_check.py
  python tests/parity_check.py
  python tests/synthetic_family_check.py
  python tests/euler_filter_check.py
  python tests/cap_height_check.py
  python scripts/smoke_test.py
}

phase_synthetic() {
  echo "=== B: synthetic dataset ==="
  python scripts/make_synthetic_dataset.py --n_train 800 --n_val 100 \
      --n_test 100 --slices 5 7 9 15 --out data/synthetic

  echo "=== C: training (4 runs) ==="
  for N in 5 7 9 15; do
    python scripts/train_model.py --data data/synthetic --N $N --m 256 \
        --epochs 100 --surf_sub 8000 --gt_sub 8000 \
        --val_every 5 --val_subset 25 --patience 30 --out runs/synth_N$N
  done

  echo "=== C2: cap ablation (2x2, cap/body error separated) ==="
  python scripts/ablation_caps.py --data data/synthetic --N 9 \
      --epochs 60 --out results/ablation_caps.csv

  echo "=== D: evaluation / sparsity sweep ==="
  for N in 5 7 9 15; do
    python scripts/evaluate.py --data data/synthetic --N $N \
        --ckpt runs/synth_N$N/best.pt --n_u 32 \
        --out results/eval_synth_N$N.csv
  done

  echo "=== E: global-constant ablation ==="
  for N in 7 9; do
    python scripts/ablation_global_constants.py --data data/synthetic --N $N \
        --ckpt runs/synth_N$N/best.pt \
        --out results/ablation_global_N$N.csv
  done

  echo "=== F: designer shapes, all modes ==="
  for ds in banana apple vase; do
    python scripts/reconstruct_designer.py --ds $ds --mode classical
    python scripts/reconstruct_designer.py --ds $ds --mode net \
        --ckpt runs/synth_N9/best.pt
    python scripts/reconstruct_designer.py --ds $ds --mode tto \
        --tto_init net --ckpt runs/synth_N9/best.pt
    python scripts/reconstruct_designer.py --ds $ds --mode tto \
        --tto_init classical
    python scripts/reconstruct_designer.py --ds $ds --mode net \
        --ckpt runs/synth_N9/best.pt --freeze_caps
  done
  python scripts/reconstruct_designer.py --ds vase --mode classical --show_crown
}

phase_real() {
  echo "=== G: mesh pipeline sanity (no download) ==="
  python scripts/make_test_meshes.py --out data/meshes_test
  python scripts/check_mesh_pipeline.py --meshes data/meshes_test --N 9

  echo "=== G: fetch Thingi10K ==="
  python scripts/fetch_thingi10k.py --out data/meshes --limit 200
  python scripts/check_mesh_pipeline.py --meshes data/meshes --N 9 --limit 25

  echo "=== H: real dataset ==="
  python scripts/make_mesh_dataset.py --meshes data/meshes --N 5 7 9 15 \
      --out data/real --val_frac 0.15 --test_frac 0.15

  echo "=== H1: train on real ==="
  python scripts/train_model.py --data data/real --N 9 --m 256 \
      --epochs 100 --surf_sub 8000 --gt_sub 8000 \
      --val_every 5 --val_subset 15 --patience 30 --out runs/real_N9
  python scripts/evaluate.py --data data/real --N 9 \
      --ckpt runs/real_N9/best.pt --n_u 32 --out results/eval_real_N9.csv

  echo "=== H3: synthetic pre-train -> fine-tune on real ==="
  python scripts/train_model.py --data data/real --N 9 --m 256 \
      --epochs 40 --lr 3e-4 --surf_sub 8000 --gt_sub 8000 \
      --val_every 5 --val_subset 15 --patience 15 \
      --init_ckpt runs/synth_N9/best.pt --out runs/real_ft_N9
  python scripts/evaluate.py --data data/real --N 9 \
      --ckpt runs/real_ft_N9/best.pt --n_u 32 \
      --out results/eval_real_finetune_N9.csv

  echo "=== H2b: visualize real results (incl. pole close-ups) ==="
  python scripts/visualize_real.py --data data/real --N 9 \
      --ckpt runs/real_N9/best.pt --n 8 --pole_zoom --out results/real_figs

  echo "=== H2: zero-shot synthetic -> real ==="
  python scripts/evaluate.py --data data/real --N 9 \
      --ckpt runs/synth_N9/best.pt --n_u 32 \
      --out results/eval_real_zeroshot_N9.csv

  echo "=== H: global-constant ablation on real ==="
  python scripts/ablation_global_constants.py --data data/real --N 9 \
      --ckpt runs/real_N9/best.pt --out results/ablation_real_N9.csv
}

case "${1:-all}" in
  verify)    phase_verify ;;
  synthetic) phase_synthetic ;;
  real)      phase_real ;;
  all)       phase_verify; phase_synthetic; phase_real ;;
  *) echo "usage: bash run_all.sh [verify|synthetic|real|all]"; exit 1 ;;
esac
echo "done. results/ and runs/ hold the outputs."
