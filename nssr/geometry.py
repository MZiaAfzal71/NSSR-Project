"""Differentiable core of the shape-preserving reconstruction pipeline.

Implements Eqs. 18-39 of the CiSE manuscript in PyTorch with learnable
log-weight hooks (see docs/METHOD.md).  All functions operate on ONE object
(batching over objects is done in the training loop; N varies per object).

Conventions
-----------
R   : (N, m, 2) float tensor - aligned, resampled contours (x, y)
Z   : (N,)      float tensor - strictly ordered heights (monotone case)
RB, RC : (2,)   base / crown 2-D points        (constants from preprocess)
Bh, Th : ()     base / crown heights           (constants from preprocess)
params : dict of learnable fields, all zero => EXACT classical pipeline:
    's_a'   : (N, m)  log-multiplier on the a-weight (next-chord norm)
    's_b'   : (N, m)  log-multiplier on the b-weight (prev-chord norm)
    's_tau' : (N, m)  log tangent scale
    's_fB'  : (m,)    log-multiplier on base scaling f_B
    's_fC'  : (m,)    log-multiplier on crown scaling f_C
    'rho'   : (N, m, 2) optional free residual for the ablation (L1')

Note on the boundary "factor 2" (Eqs. 20-21): it is applied EXPLICITLY here
(not through s_a init), so that params==0 reproduces the classical formulas
exactly, row 0 and row N-1 included.
"""
from __future__ import annotations
import torch

EPS = 1e-12


def _norm(v, dim=-1, keepdim=True):
    return torch.sqrt((v * v).sum(dim=dim, keepdim=keepdim) + EPS)


def _cross2(u, v):
    """Scalar 2-D cross product u x v, last dim = 2."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


# ----------------------------------------------------------------------
# Tangent field  (Eqs. 18-21, learnable reweighting per METHOD.md sec.2)
# ----------------------------------------------------------------------
def tangent_field(R, Z, RB, RC, Bh, Th, params, closed_top=True):
    """Return gR (N, m, 2) and gZ (N, m) for all contours 0..N-1.

    closed_top=False (e.g. the vase): the crown row uses the plain interior
    one-sided formula with no virtual crown point; the crown cap is skipped
    later in hermite_surface as well.
    """
    N, m, _ = R.shape
    dev, dt = R.device, R.dtype

    # Chords between successive contours, with virtual base/crown rows.
    # dR[i] = R_i - R_{i-1} for i = 0..N  (row 0 uses RB, row N uses RC)
    dR = torch.empty(N + 1, m, 2, device=dev, dtype=dt)
    dR[1:N] = R[1:] - R[:-1]
    dR[0] = R[0] - RB                       # Delta R_0 (virtual base)
    dR[N] = (RC - R[-1]) if closed_top else dR[N - 1]
    dZ = torch.empty(N + 1, device=dev, dtype=dt)
    dZ[1:N] = Z[1:] - Z[:-1]
    dZ[0] = Z[0] - Bh
    dZ[N] = (Th - Z[-1]) if closed_top else dZ[N - 1]

    nrm = _norm(dR).squeeze(-1)             # (N+1, m)  ||Delta R_i||

    # classical weights: a = ||dR_{i+1}||, b = ||dR_i||   (per row i)
    a = nrm[1:]                             # (N, m)
    b = nrm[:-1]                            # (N, m)

    # boundary factor 2 (Eqs. 20-21): on the a-side at the base row,
    # on the b-side at the crown row (symmetric analog; TODO(verify)
    # against reference implementation, see METHOD.md sec. 2.3).
    fac_a = torch.ones(N, 1, device=dev, dtype=dt)
    fac_b = torch.ones(N, 1, device=dev, dtype=dt)
    fac_a[0] = 2.0
    if closed_top:
        fac_b[N - 1] = 2.0

    a = a * fac_a * torch.exp(params["s_a"])
    b = b * fac_b * torch.exp(params["s_b"])

    dZi = dZ[:-1].abs().unsqueeze(-1)       # (N, 1) -> broadcast over m
    dZip = dZ[1:].abs().unsqueeze(-1)
    denom = a * dZi + b * dZip + EPS        # (N, m)

    gR = (a.unsqueeze(-1) * dR[:-1] + b.unsqueeze(-1) * dR[1:]) / denom.unsqueeze(-1)
    gZ = (a * dZ[:-1].unsqueeze(-1) + b * dZ[1:].unsqueeze(-1)) / denom

    # Coincident-point guard (Eq. 19): if both chords vanish, force axial.
    tiny = 1e-9
    coincident = (nrm[:-1] < tiny) & (nrm[1:] < tiny)             # (N, m)
    axial = torch.sign(dZ[1:]).unsqueeze(-1).expand(N, m)
    gZ = torch.where(coincident, axial, gZ)
    gR = torch.where(coincident.unsqueeze(-1), torch.zeros_like(gR), gR)

    # learned tangent scale (L1)
    tau = torch.exp(params["s_tau"])
    gR = gR * tau.unsqueeze(-1)
    gZ = gZ * tau

    # optional free residual (ablation L1')
    if "rho" in params and params["rho"] is not None:
        gR = gR + params["rho"]
    return gR, gZ


# ----------------------------------------------------------------------
# Boundary directions (Eqs. 22-24) and scalings (Eqs. 25-26)
# ----------------------------------------------------------------------
def boundary_directions(R, Z, RB, RC, Bh, Th, at_base=True, eps=1e-8):
    """gRB / gRC (m, 2) — reconciled with reference/surfaces_pytorch.py:
    base : E = (R0 - RB)/||.||, beta1 = (Bh - a1*(Z0-Bh)) / (Z1-Z0)
    crown: E = (RC - R_{N-1})/||.||,
           beta1 = (Th - a1*(Z_{N-1}-Z_{N-2})) / (Th - Z_{N-1})
    with a1 = 1 + (j mod 15), D_k = a_k*BA + b_k*CB, F = (D1-D2)/||.||,
    selection: E if |ExF| < eps else (F if E.F > 0 else -F).
    Verified elementwise by tests/parity_check.py."""
    N, m, _ = R.shape
    dev, dt = R.device, R.dtype
    j = torch.arange(m, device=dev, dtype=dt)
    a1 = 1.0 + torch.remainder(j, 15.0)
    if at_base:
        B, C = R[0], R[1]
        b1 = (Bh - a1 * (Z[0] - Bh)) / (Z[1] - Z[0])
        b2 = (Bh + a1 * (Z[0] - Bh)) / (Z[1] - Z[0])
        BA = B - RB
        CB = C - B
        E = BA / _norm(BA)
    else:
        A, B = R[N - 2], R[N - 1]
        b1 = (Th - a1 * (Z[N - 1] - Z[N - 2])) / (Th - Z[N - 1])
        b2 = (Th + a1 * (Z[N - 1] - Z[N - 2])) / (Th - Z[N - 1])
        BA = B - A
        CB = RC - B
        E = CB / _norm(CB)
    D1 = a1.unsqueeze(-1) * BA + b1.unsqueeze(-1) * CB
    D2 = -a1.unsqueeze(-1) * BA + b2.unsqueeze(-1) * CB
    F = (D1 - D2) / _norm(D1 - D2)
    c = _cross2(E, F).abs()
    d = (E * F).sum(-1)
    out = torch.where((c < eps).unsqueeze(-1), E,
          torch.where((d > 0).unsqueeze(-1), F, -F))
    return out


def boundary_scalings(R, RB, RC, Z, Bh, Th, gR, params, closed_top=True):
    """f_B(j), f_C(j) with learned multipliers (L2).
    Reconciled with the reference code: DIVIDES by the sqrt term
    (the manuscript Eqs. 25-26 print a multiplication; the reference
    implementation divides — the code is authoritative).  Verified by
    tests/parity_check.py."""
    m = R.shape[1]
    nB = _norm(R[0] - RB).squeeze(-1)
    gB = _norm(gR[0] / nB.unsqueeze(-1)).squeeze(-1)
    fB = (2.0 ** 0.5) * nB / torch.sqrt(1.0 + (Z[0] - Bh).abs() * gB)
    fB = fB * torch.exp(params["s_fB"])
    if closed_top:
        nC = _norm(RC - R[-1]).squeeze(-1)
        gC = _norm(gR[-1] / nC.unsqueeze(-1)).squeeze(-1)
        fC = (2.0 ** 0.5) * nC / torch.sqrt(1.0 + (Th - Z[-1]).abs() * gC)
        fC = fC * torch.exp(params["s_fC"])
    else:
        fC = torch.zeros(m, device=R.device, dtype=R.dtype)
    return fB, fC


# ----------------------------------------------------------------------
# Hermite surface  (Eqs. 27-39)
# ----------------------------------------------------------------------
def hermite_basis(u):
    L0 = 1 - 3 * u ** 2 + 2 * u ** 3
    L1 = 3 * u ** 2 - 2 * u ** 3
    H0 = u - 2 * u ** 2 + u ** 3
    H1 = -(u ** 2) + u ** 3
    return L0, L1, H0, H1


def hermite_surface(R, Z, RB, RC, Bh, Th, params, n_u=32,
                    closed_top=True, base_circular=True,
                    crown_circular=True):
    """Evaluate the full surface.  Returns:
       pts : (n_patches, n_u, m, 3) surface points, patch order =
             [base cap (if any), interior 1..N-1, crown cap (if any)]
    """
    N, m, _ = R.shape
    dev, dt = R.device, R.dtype
    u = torch.linspace(0.0, 1.0, n_u, device=dev, dtype=dt)
    L0, L1, H0, H1 = hermite_basis(u)                       # (n_u,)
    L0 = L0.view(-1, 1, 1); L1 = L1.view(-1, 1, 1)
    H0 = H0.view(-1, 1, 1); H1 = H1.view(-1, 1, 1)

    gR, gZ = tangent_field(R, Z, RB, RC, Bh, Th, params, closed_top)
    fB, fC = boundary_scalings(R, RB, RC, Z, Bh, Th, gR, params, closed_top)

    patches = []

    # ---- base cap (patch between RB and contour 0), Eqs. 31-32 ----------
    dZ0 = (Z[0] - Bh).abs()
    if base_circular:
        gRB = boundary_directions(R, Z, RB, RC, Bh, Th, at_base=True)
        FR = (L0 * RB.view(1, 1, 2) + L1 * R[0].unsqueeze(0)
              + H0 * (fB.unsqueeze(-1) * gRB).unsqueeze(0)
              + 2.0 * dZ0 * H1 * gR[0].unsqueeze(0))          # (n_u, m, 2)
        Fz = ((1 - u ** 2) * Bh + (u ** 2) * Z[0]).view(-1, 1).expand(n_u, m)
    else:                                                     # Eqs. 33-34
        FR = (L0 * RB.view(1, 1, 2) + L1 * R[0].unsqueeze(0)
              + 2.0 * dZ0 * H1 * gR[0].unsqueeze(0))
        Fz = (Bh * L0 + Z[0] * L1 + (Z[0] - Bh) * H0
              ).view(-1, 1) + dZ0 * gZ[0].unsqueeze(0) * H1.view(-1, 1)
    patches.append(torch.cat([FR, Fz.unsqueeze(-1)], dim=-1))

    # ---- interior patches i = 1..N-1 (Eqs. 29-30), vectorized over i ----
    dZi = (Z[1:] - Z[:-1]).abs().view(1, -1, 1, 1)            # (1, N-1, 1, 1)
    L0b = L0.unsqueeze(1); L1b = L1.unsqueeze(1)              # (n_u,1,1,1)
    H0b = H0.unsqueeze(1); H1b = H1.unsqueeze(1)
    FR = (L0b * R[:-1].unsqueeze(0) + L1b * R[1:].unsqueeze(0)
          + dZi * (H0b * gR[:-1].unsqueeze(0) + H1b * gR[1:].unsqueeze(0)))
    Fz = (L0b.squeeze(-1) * Z[:-1].view(1, -1, 1)
          + L1b.squeeze(-1) * Z[1:].view(1, -1, 1)
          + dZi.squeeze(-1) * (H0b.squeeze(-1) * gZ[:-1].unsqueeze(0)
                               + H1b.squeeze(-1) * gZ[1:].unsqueeze(0)))
    interior = torch.cat([FR, Fz.unsqueeze(-1)], dim=-1)      # (n_u, N-1, m, 3)
    patches.extend(interior.permute(1, 0, 2, 3))              # N-1 patches

    # ---- crown cap (Eqs. 35-36 / 37-38) ---------------------------------
    if closed_top:
        dZN = (Th - Z[-1]).abs()
        if crown_circular:
            gRC = boundary_directions(R, Z, RB, RC, Bh, Th, at_base=False)
            FR = (L0 * R[-1].unsqueeze(0) + L1 * RC.view(1, 1, 2)
                  + H1 * (fC.unsqueeze(-1) * gRC).unsqueeze(0)
                  + 2.0 * dZN * H0 * gR[-1].unsqueeze(0))
            Fz = ((1 - u) ** 2 * Z[-1] + u * (2 - u) * Th).view(-1, 1).expand(n_u, m)
        else:
            FR = (L0 * R[-1].unsqueeze(0) + L1 * RC.view(1, 1, 2)
                  + 2.0 * dZN * H0 * gR[-1].unsqueeze(0))
            Fz = (Z[-1] * L0 + Th * L1).view(-1, 1) \
                 + dZN * gZ[-1].unsqueeze(0) * H0.view(-1, 1) \
                 + (Th - Z[-1]) * H1.view(-1, 1)
        patches.append(torch.cat([FR, Fz.unsqueeze(-1)], dim=-1))

    return torch.stack(patches, dim=0)                        # (P, n_u, m, 3)


def surface_points(S):
    """Flatten patch grid to a point cloud (P*n_u*m, 3)."""
    return S.reshape(-1, 3)


def surface_normals(S):
    """Finite-difference normals on the (u, j) grid, per patch.
    Returns (P, n_u, m, 3) unit normals (used by the normal loss/metric)."""
    du = S[:, 1:, :, :] - S[:, :-1, :, :]
    du = torch.cat([du, du[:, -1:, :, :]], dim=1)
    dj = torch.roll(S, shifts=-1, dims=2) - S       # circular in j
    n = torch.cross(du, dj, dim=-1)
    return n / _norm(n)


def zero_params(N, m, device="cpu", dtype=torch.float64, free_residual=False):
    """Parameter dict that reproduces the classical pipeline exactly."""
    z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
    p = {"s_a": z(N, m), "s_b": z(N, m), "s_tau": z(N, m),
         "s_fB": z(m), "s_fC": z(m)}
    if free_residual:
        p["rho"] = z(N, m, 2)
    return p
