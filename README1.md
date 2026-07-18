# NSSR — Neural Shape-Preserving Surface Reconstruction

Research codebase for the paper idea:

> **Learning the Free Parameters of a Shape-Preserving Cross-Sectional
> Reconstruction Pipeline: A Differentiable Spline Approach**

Core claim: neural implicit methods (OReX and relatives) discard geometric
guarantees; classical shape-preserving splines discard data-driven priors.
We make the classical Goodman / Siddiqi–Afzal pipeline **differentiable in
PyTorch** and **learn its free parameters** (tangent weights, tangent scales,
boundary scalings) with small neural networks, trained end-to-end through the
spline evaluation against dense ground-truth surfaces.

Key structural property exploited everywhere:

> **With all network outputs = 0, the model is *exactly* the classical
> pipeline.** Every improvement over that initialization is a measurable,
> attributable gain of learning.

---

## Repository layout

```
nssr-project/
├── README.md                  ← this file (roadmap)
├── requirements.txt
├── docs/
│   └── METHOD.md              ← full mathematics: learnable reformulation,
│                                losses, differentiability analysis, metrics,
│                                experiment/ablation plan
├── nssr/
│   ├── preprocess.py          ← NumPy: contour resampling to fixed m,
│   │                            cyclic alignment (Eqs. 5–9), base/crown
│   │                            point & height estimation (Eqs. 10–17)
│   ├── geometry.py            ← PyTorch: differentiable tangents (18–24),
│   │                            scalings (25–26), Hermite surface (27–39),
│   │                            with learnable log-weight hooks
│   ├── networks.py            ← circular 1-D CNNs predicting the learnable
│   │                            parameters from invariant contour features
│   ├── losses.py              ← Chamfer, normal consistency, regularizers
│   ├── metrics.py             ← Chamfer / Hausdorff / normal metrics +
│   │                            C1 tangent-magnitude diagnostic
│   ├── synthetic.py           ← analytic generalized-cylinder dataset
│   │                            (train immediately, no downloads needed)
│   ├── slicing.py             ← trimesh-based slicing of real meshes
│   │                            (Thingi10k / ShapeNet / scans)
│   └── train.py               ← training loop
├── scripts/
│   ├── smoke_test.py          ← RUN THIS FIRST on your machine
│   ├── make_synthetic_dataset.py
│   ├── train_model.py
│   └── evaluate.py
├── baselines/
│   └── implicit_baseline.py   ← minimal OReX-style occupancy MLP baseline
└── tests/
    └── numpy_sanity_check.py  ← NumPy mirror of the core math (verified)
```

---

## Step-by-step research plan

### Phase 0 — Environment and verification (Day 1)

```bash
python -m venv venv && source venv/bin/activate     # or conda
pip install -r requirements.txt
python tests/numpy_sanity_check.py    # verifies the math (no torch needed)
python scripts/smoke_test.py          # verifies torch pipeline + gradients
```

The smoke test checks: (a) with zero parameters the torch pipeline matches
the NumPy classical implementation; (b) gradients flow from a Chamfer loss
back to the network parameters; (c) a sphere is reconstructed from 7 circles
with small error.

**Also in Phase 0:** reconcile `nssr/geometry.py` with your reference
implementation from the CiSE paper. Spots marked `TODO(verify)` are places
where the manuscript equations were ambiguous (notably the crown tangent
Eq. 21 denominators and the boundary-direction η definition). Your original
code is the ground truth; adjust mine to match, then re-run the smoke test.

### Phase 1 — Data (Week 1–2)

1. `python scripts/make_synthetic_dataset.py` — generates ~1000 generalized
   cylinders / solids of revolution with harmonic cross-sections, analytic
   dense ground truth, and sparse slice inputs at N ∈ {5, 7, 9, 15}. This is
   enough to train and debug the whole system.
2. Real meshes: download watertight meshes (Thingi10k subset, ShapeNet
   bottle/vase/jar categories, or your own scans) into `data/meshes/`, then
   `nssr/slicing.py` extracts contours + dense GT samples. Keep
   apple/banana/vase from the CiSE paper as **held-out qualitative cases**.

### Phase 2 — Training (Week 2–4)

```bash
python scripts/train_model.py --data data/synthetic --epochs 200
```

Monitor: (a) loss vs. the θ=0 classical baseline (logged automatically at
epoch 0); (b) the C1 tangent-magnitude diagnostic; (c) parameter magnitudes
(if they explode, increase `--reg`).

### Phase 3 — Evaluation and baselines (Week 4–6)

- `python scripts/evaluate.py` → Chamfer, Hausdorff, normal consistency,
  per-object tables, sparsity sweep (accuracy vs. slice count N).
- Run `baselines/implicit_baseline.py` (simple occupancy MLP) on the same
  slices.
- Clone and run the **official OReX** code on the same inputs for the
  headline comparison table.
- Ablations (see METHOD.md §7): weight-space vs. free-residual tangents,
  each learnable group on/off, generalization synthetic→real.

### Phase 4 — Writing (Week 6+)

Suggested venues: Computer Aided Geometric Design, Computer-Aided Design,
Computers & Graphics, The Visual Computer. Story: guarantees + learning.

---

## What to send back to me after each phase

- Phase 0: output of `smoke_test.py` and any mismatches vs. your reference code.
- Phase 1: dataset statistics printed by the scripts.
- Phase 2: training curves (`runs/*/log.csv`) and a rendered surface or two.
- Phase 3: the tables from `evaluate.py`.

I will help you interpret results, fix issues, design ablations, and draft
the manuscript sections as results arrive.
