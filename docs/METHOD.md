# METHOD — Learning the Free Parameters of a Shape-Preserving Reconstruction Pipeline

This document gives the complete mathematics connecting the classical
pipeline (Goodman 1988; Goodman–Ong–Unsworth 1991; Siddiqi–Afzal 2002;
equation numbers below refer to your CiSE manuscript) to the learned model.

Notation: contours `R ∈ R^{N×m×2}` at heights `Z ∈ R^N` (already resampled
to a common point count m and cyclically aligned); base/crown points
`R_B, R_C ∈ R^2` and heights `B_h, T_h`; Hermite parameter `u ∈ [0,1]`
sampled at n values; surface `S(i,j,u) = (F_R(i,j,u), F_z(i,j,u)) ∈ R^3`.

---

## 1. Which stages stay classical, which become learnable

| Stage | Eqs. | Treatment |
|---|---|---|
| Closed-curve resampling (rational cubic spline) | 1–4 | classical, NumPy preprocessing (learnable λ, μ, γ, δ is Phase B, §8) |
| Normalization + cyclic alignment | 5–9 | classical, NumPy preprocessing (argmin over integer shifts is not differentiable and does not need to be: it depends only on data, never on θ) |
| Base/crown points and heights | 10–17 | classical, NumPy preprocessing; enters the torch graph as constants |
| **Tangent vectors (interior/base/crown)** | 18–21 | **learnable** (§2) |
| **Boundary directions + scalings** | 22–26 | **learnable** (§3) |
| Hermite blending | 27–39 | differentiable evaluation in torch, exact as written |

Rationale: the tangent field is where all of the pipeline's heuristic
freedom lives (the chord-norm weighting, the factor 2 at base/crown, the
κ = 1 + (j mod 15) rule, the √2 scaling), and the tangent field is exactly
what determines interpolation quality between the sparse slices. The
Hermite formulas themselves are forced by C1 interpolation.

---

## 2. Learnable tangent field (weight-space parameterization)

### 2.1 Classical interior tangents (Eq. 18)

With ΔR_{i,j} = R_{i,j} − R_{i−1,j}, ΔZ_i = Z_i − Z_{i−1}, and
a_{ij} = ‖ΔR_{i+1,j}‖, b_{ij} = ‖ΔR_{i,j}‖:

    g_R(i,j) = (a_{ij} ΔR_{i,j} + b_{ij} ΔR_{i+1,j}) / D_{ij}
    g_Z(i,j) = (a_{ij} ΔZ_i     + b_{ij} ΔZ_{i+1})   / D_{ij}
    D_{ij}   =  a_{ij} |ΔZ_i| + b_{ij} |ΔZ_{i+1}|

### 2.2 Learned reweighting

A network Φ_θ (§4) outputs three fields s^a, s^b, s^τ ∈ R^{N×m}. Define

    ã_{ij} = a_{ij} · exp(s^a_{ij}),   b̃_{ij} = b_{ij} · exp(s^b_{ij})

and replace (a, b) → (ã, b̃) in §2.1, then scale the resulting tangent:

    g̃(i,j) = exp(s^τ_{ij}) · g(i,j; ã, b̃)                            (L1)

Properties (these are the paper's selling points — state them as a lemma):

1. **Exact classical recovery.** s ≡ 0 ⇒ g̃ = g. The classical pipeline is
   a point in parameter space, and it is the initialization.
2. **Cone preservation.** Because exp(·) > 0, the direction of g̃_R stays in
   the convex cone spanned by ΔR_{i,j} and ΔR_{i+1,j} — the same structural
   constraint that gives the classical scheme its shape-preserving
   behaviour. The learned model cannot produce a tangent that "points
   backwards" relative to the data, no matter what the network outputs.
3. **C1 continuity is preserved by construction.** Adjacent Hermite patches
   (i−1,i) and (i,i+1) share the *same* g̃(i,·) at their common contour, so
   the surface remains C1 across every junction for **all** θ. This is a
   guarantee no neural-field method offers.
4. Bound the outputs, s = c·tanh(raw) with c ≈ 2, so weights vary within
   e^{±2} ≈ [0.14, 7.4]× classical — stable training, no degeneracy.

### 2.3 Base and crown tangents (Eqs. 20–21)

Base (i = 0) uses the virtual point R_{−1,j} = R_B, Z_{−1} = B_h,
ΔR_{0,j} = R_{0,j} − R_B, ΔZ_0 = Z_0 − B_h, and the classical factor 2 on
the a-side. We absorb that factor into the same parameterization by
initializing the bias of the s^a output head at ln 2 for boundary rows:

    g̃_R(0,j) = (ã ΔR_{0,j} + b̃ ΔR_{1,j}) / (ã |ΔZ_0| + b̃ |ΔZ_1|),
    ã = ‖ΔR_{1,j}‖ · exp(s^a_{0j}),  s^a_{0j} initialized ≈ ln 2.

Crown (i = N−1) is symmetric with virtual point R_{N,j} = R_C, Z_N = T_h
and the factor 2 on the b-side.

> **RESOLVED (reference code, verified by tests/parity_check.py):** the
> manuscript's Eq. 21 denominators are typos; the reference implementation
> computes the symmetric analog of the base formula —
> g_R(N−1,j) = (a_R (B−A) + 2 b_R (R_C−B)) / (a_R |ΔZ_{N−1}| + 2 b_R |ΔZ_N|)
> with a_R = ‖R_C−B‖, b_R = ‖B−A‖ — which is what NSSR implements.
> **Fix Eq. 21 in the new manuscript.**

### 2.4 Ablation: free-residual tangents (breaking the cone)

To test whether the cone constraint costs accuracy, an ablation variant
adds an unconstrained residual:

    g̃_R = exp(s^τ) g_R + ρ_{ij},   ρ ∈ R^{N×m×2} predicted directly.  (L1')

This loses guarantee (2) (and shape preservation) but keeps C1. If (L1)
matches (L1') in accuracy, the paper's "guarantees are free" claim is
strengthened; if not, you quantify the price of the guarantee. Either
outcome is a result.

---

## 3. Learnable boundary scalings

Classical scalings (Eqs. 25–26) set the magnitude of the boundary tangent
along the base/crown cap. We multiply them by a learned positive factor:

    f̃_B(j) = f_B(j) · exp(s^B_j),    f̃_C(j) = f_C(j) · exp(s^C_j)     (L2)

with s^B, s^C ∈ R^m from the boundary head of Φ_θ, initialized 0. The
boundary *directions* g_RB, g_RC (Eqs. 22–24) stay classical: they encode a
discrete orientation selection (c < ε / d > 0 branching) that is naturally
data-determined; learning them is a possible extension, not needed for v1.

The cap height interpolation F_z(0,j,u) = (1−u²)B_h + u²Z_0 (Eq. 32) and
its crown analog (Eq. 36) are kept exactly.

---

## 4. The parameter networks Φ_θ

**Inputs — geometric invariants only** (rotation/translation invariance in
the (x,y) plane; scale handled by normalizing each object to unit bounding
box in preprocessing). Per contour point (i,j), the feature vector:

    x_{ij} = [ log(‖ΔR_{i,j}‖+ε), log(‖ΔR_{i+1,j}‖+ε),      chord norms to prev/next slice
               ΔZ_i, ΔZ_{i+1},                              slice spacings
               κ_{ij},                                      in-plane discrete curvature (Eq. 1)
               ‖R_{ij} − c_i‖ / r̄_i,                        normalized radial distance (c_i = contour centroid, r̄_i = mean radius)
               i/(N−1), sin(2π i/(N−1)), cos(2π i/(N−1)) ]  height position encoding

**Architecture.** A 1-D CNN along the circumferential index j with
**circular padding** (the contour is closed — this bakes in cyclic
equivariance: rotating the starting point of the parameterization rotates
the predicted fields identically). Three conv layers (kernel 5, 64
channels, SiLU), applied to every row i with shared weights, followed by a
linear head → (s^a, s^b, s^τ) and, for rows 0 and N−1, the boundary head →
(s^B, s^C). ~50k parameters total. Small on purpose: the model must work
when trained on ~10³ shapes, and a small hypernetwork over an exact
geometric decoder is the point of the paper.

Why not a PointNet/transformer: the data is an ordered cyclic grid
(N × m); a circular CNN is the minimal architecture with the right
symmetry. (A transformer along i is a reasonable extension for large N.)

---

## 5. Losses

Sample the predicted surface on the (patch, j, u) grid → point set
Ŝ ∈ R^{P×3}. Ground truth: dense point sample X ∈ R^{Q×3} of the true
surface (analytic for synthetic data; area-weighted mesh sampling for real
meshes), with unit normals n_X.

**(a) Two-sided Chamfer distance**

    L_CD = (1/P) Σ_p min_q ‖Ŝ_p − X_q‖² + (1/Q) Σ_q min_p ‖X_q − Ŝ_p‖²

(min over points is subdifferentiable; standard in geometry learning).

**(b) Normal consistency.** Predicted normals from the parameter grid by
finite differences: n̂ = (∂S/∂j × ∂S/∂u), normalized. For each GT point,
compare with the normal of its nearest predicted point:

    L_N = (1/Q) Σ_q (1 − |n_X(q) · n̂(NN(q))|)

**(c) Proximity regularizer** — stay near the classical solution unless
data says otherwise (also the knob for the "how far from classical do we
drift" analysis):

    L_reg = mean(s²)  over all predicted fields

**(d) Circumferential smoothness** of the predicted fields (optional,
prevents high-frequency wiggle in the parameter maps):

    L_sm = mean( (s_{i,j+1} − s_{i,j})² )   (cyclic)

Total: `L = L_CD + λ_N L_N + λ_r L_reg + λ_s L_sm`, defaults
λ_N = 0.1, λ_r = 1e−3, λ_s = 1e−3.

---

## 6. Differentiability audit

- Norms ‖·‖ and divisions: ε-guarded (ε = 1e−12 f64 / 1e−8 f32).
- |ΔZ|: non-smooth only at ΔZ = 0, which cannot occur between distinct
  slices (and B_h < Z_0, T_h > Z_{N−1} by construction).
- Cyclic-shift argmin (Eq. 7): integer, non-differentiable — deliberately
  kept in preprocessing; it depends only on the input contours, never on θ,
  so no gradient needs to pass through it.
- Curvature-sign case selection in the resampling spline (Eqs. 3–4) and the
  orientation selection in Eq. 24: discrete branches determined by the
  *data*, not by θ; implemented with `torch.where` — gradients flow through
  the selected branch, which is correct.
- Root selection (Eqs. 15–16): min/max of quadratic roots — preprocessing
  constants in v1.
- Coincident-point guard (Eq. 19): `torch.where` on a data mask.

Conclusion: L is (almost-everywhere) differentiable in θ everywhere it
needs to be, with the classical pipeline at θ = 0.

---

## 7. Experiments and ablations (paper skeleton)

1. **Main table.** Chamfer / Hausdorff / normal consistency on held-out
   synthetic + real objects, N ∈ {5, 7, 9, 15} slices, methods:
   classical (θ=0) · NSSR (ours) · NSSR free-residual · occupancy-MLP
   baseline · official OReX. Report mean ± std over test set.
2. **Sparsity sweep.** Accuracy vs. N. Hypothesis: learned tangents help
   most at small N (the interpolation is most underdetermined there).
3. **Ablations.** (i) each learnable group off (tangent weights / τ scale /
   boundary scalings); (ii) cone-constrained vs. free residual; (iii) no
   normal loss; (iv) regularization strength sweep → "distance from
   classical" vs. accuracy curve.
4. **Guarantee verification.** Your existing C1 diagnostic (minimum
   tangent-vector magnitude per junction) reported for the *learned* model
   — by construction it should remain clear of zero; this is the figure
   neural-field baselines cannot produce.
5. **Generalization.** Train on synthetic only → test on real meshes and on
   apple/banana/vase. Then fine-tune per-object (test-time optimization of
   s directly, no network — a second operating mode that needs no training
   set at all and is a fair comparison to OReX, which also fits per shape).
6. **Qualitative.** Wireframes/renders: classical vs. learned vs. GT at low
   N; error heatmaps on the surface.

Point 5 deserves emphasis: **per-object test-time optimization** (optimize
the s-fields for one shape by gradient descent on L_CD against its own
dense scan, or in the truly sparse setting, against a leave-one-slice-out
loss) is a training-data-free mode unique to this design and a strong
reviewer answer to "where does training data come from in practice?" —
leave-one-slice-out: hold out each contour in turn, reconstruct from the
rest, penalize distance of the held-out contour to the surface.

## 8. Phase B extension (second paper or added section)

Make the resampling stage (Eqs. 1–4) differentiable and learn per-segment
λ, μ ∈ (0,1] via sigmoid and γ, δ with γ+δ < 1 via a scaled softmax. This
couples learning into contour geometry itself. Kept out of v1 to keep the
contribution clean.
