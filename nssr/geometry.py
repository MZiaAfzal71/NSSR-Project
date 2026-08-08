"""NSSR-V2 geometry orchestration with legacy-compatible surface semantics.

This module preserves the current NSSR/CiSE reconstruction equations while
connecting them to the V2 core infrastructure.

Important compatibility note
----------------------------
The existing NSSR surface is cubic Hermite in the inter-slice parameter u,
while the circumferential direction is represented by aligned discrete
samples j.  This module deliberately preserves that behavior.  It does NOT
silently replace the current reconstruction with a bicubic tensor-product
surface in (u,v), because doing so would change the reference geometry.

The new ``nssr.core.hermite`` tensor-product kernel remains available for
future fully bicubic patches, but the compatibility path here uses the
canonical 1-D basis functions from ``nssr.core.basis``.

Public layers
-------------
1. Classical/learned geometry primitives
   - apply_cap_heights
   - tangent_field
   - boundary_directions
   - boundary_scalings

2. Reference-compatible reconstruction
   - hermite_surface
   - surface_points
   - surface_normals
   - zero_params

3. V2 orchestration
   - sampled_surface_derivatives
   - evaluate_geometry

The V2 orchestration can additionally compute Jacobian, curvature and
validation reports without changing the reconstructed point positions.
"""

from __future__ import annotations

from typing import Optional

import torch

from .core.basis import hermite_basis
from .core.curvature import curvature_from_evaluation
from .core.jacobian import surface_jacobian
from .core.types import (
    GeometryOutput,
    SurfaceDerivatives,
    SurfaceEvaluation,
    SurfacePatch,
    TangentField,
)
from .core.validation import validate_surface


EPS = 1.0e-12


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _norm(v: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
    return torch.sqrt((v * v).sum(dim=dim, keepdim=keepdim) + EPS)


def _cross2(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scalar 2-D cross product, last dimension = 2."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _basis_components(u: torch.Tensor):
    """Return L0,L1,H0,H1 from the canonical V2 basis implementation."""
    H = hermite_basis(u)
    return H[..., 0], H[..., 1], H[..., 2], H[..., 3]


# ---------------------------------------------------------------------------
# Learnable cap heights
# ---------------------------------------------------------------------------

def apply_cap_heights(
    Z: torch.Tensor,
    Bh: torch.Tensor,
    Th: torch.Tensor,
    params: Optional[dict[str, torch.Tensor]],
    max_log: float = 0.7,
):
    """Apply learned multiplicative changes to classical cap-height gaps.

    The sign of the original gap is preserved:

        Bh' = Z[0]  - exp(s_bh) * (Z[0] - Bh)
        Th' = Z[-1] + exp(s_th) * (Th - Z[-1])

    Zero parameters reproduce the classical heights exactly.
    """
    if params is None or "s_bh" not in params:
        return Bh, Th

    sb = max_log * torch.tanh(params["s_bh"])
    st = max_log * torch.tanh(params["s_th"])

    Bh_new = Z[0] - torch.exp(sb) * (Z[0] - Bh)
    Th_new = Z[-1] + torch.exp(st) * (Th - Z[-1])

    return Bh_new, Th_new


# ---------------------------------------------------------------------------
# Tangent field -- reference-compatible equations
# ---------------------------------------------------------------------------

def tangent_field(
    R: torch.Tensor,
    Z: torch.Tensor,
    RB: torch.Tensor,
    RC: torch.Tensor,
    Bh: torch.Tensor,
    Th: torch.Tensor,
    params: dict[str, torch.Tensor],
    closed_top: bool = True,
):
    """Return reference-compatible gR and gZ tangent fields.

    Shapes
    ------
    R   : (N,m,2)
    Z   : (N,)
    gR  : (N,m,2)
    gZ  : (N,m)

    This preserves the current NSSR equations and learned log-multipliers.
    The separate ``nssr.core.tangent`` module provides additional monotone/
    vector-limited tangent constructors for safe-reference generation and
    repair, but is not substituted here automatically.
    """
    N, m, _ = R.shape
    dev, dt = R.device, R.dtype

    Bh, Th = apply_cap_heights(Z, Bh, Th, params)

    dR = torch.empty(N + 1, m, 2, device=dev, dtype=dt)
    dR[1:N] = R[1:] - R[:-1]
    dR[0] = R[0] - RB
    dR[N] = (RC - R[-1]) if closed_top else dR[N - 1]

    dZ = torch.empty(N + 1, device=dev, dtype=dt)
    dZ[1:N] = Z[1:] - Z[:-1]
    dZ[0] = Z[0] - Bh
    dZ[N] = (Th - Z[-1]) if closed_top else dZ[N - 1]

    nrm = _norm(dR).squeeze(-1)

    a = nrm[1:]
    b = nrm[:-1]

    fac_a = torch.ones(N, 1, device=dev, dtype=dt)
    fac_b = torch.ones(N, 1, device=dev, dtype=dt)

    fac_a[0] = 2.0
    if closed_top:
        fac_b[N - 1] = 2.0

    a = a * fac_a * torch.exp(params["s_a"])
    b = b * fac_b * torch.exp(params["s_b"])

    dZi = dZ[:-1].abs().unsqueeze(-1)
    dZip = dZ[1:].abs().unsqueeze(-1)

    denom = a * dZi + b * dZip + EPS

    gR = (
        a.unsqueeze(-1) * dR[:-1]
        + b.unsqueeze(-1) * dR[1:]
    ) / denom.unsqueeze(-1)

    gZ = (
        a * dZ[:-1].unsqueeze(-1)
        + b * dZ[1:].unsqueeze(-1)
    ) / denom

    tiny = 1.0e-9
    coincident = (nrm[:-1] < tiny) & (nrm[1:] < tiny)

    axial = torch.sign(dZ[1:]).unsqueeze(-1).expand(N, m)
    gZ = torch.where(coincident, axial, gZ)
    gR = torch.where(
        coincident.unsqueeze(-1),
        torch.zeros_like(gR),
        gR,
    )

    tau = torch.exp(params["s_tau"])
    gR = gR * tau.unsqueeze(-1)
    gZ = gZ * tau

    return gR, gZ


# ---------------------------------------------------------------------------
# Boundary direction/scaling -- reference-compatible
# ---------------------------------------------------------------------------

def boundary_directions(
    R: torch.Tensor,
    Z: torch.Tensor,
    RB: torch.Tensor,
    RC: torch.Tensor,
    Bh: torch.Tensor,
    Th: torch.Tensor,
    at_base: bool = True,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Compute gRB/gRC boundary directions."""
    N, m, _ = R.shape
    dev, dt = R.device, R.dtype

    j = torch.arange(m, device=dev, dtype=dt)
    a1 = 1.0 + torch.remainder(j, 15.0)

    if at_base:
        B, C = R[0], R[1]

        denom = Z[1] - Z[0]
        denom = torch.where(
            denom.abs() < EPS,
            torch.full_like(denom, EPS),
            denom,
        )

        b1 = (Bh - a1 * (Z[0] - Bh)) / denom
        b2 = (Bh + a1 * (Z[0] - Bh)) / denom

        BA = B - RB
        CB = C - B
        E = BA / _norm(BA)

    else:
        A, B = R[N - 2], R[N - 1]

        denom = Th - Z[N - 1]
        denom = torch.where(
            denom.abs() < EPS,
            torch.full_like(denom, EPS),
            denom,
        )

        b1 = (Th - a1 * (Z[N - 1] - Z[N - 2])) / denom
        b2 = (Th + a1 * (Z[N - 1] - Z[N - 2])) / denom

        BA = B - A
        CB = RC - B
        E = CB / _norm(CB)

    D1 = a1.unsqueeze(-1) * BA + b1.unsqueeze(-1) * CB
    D2 = -a1.unsqueeze(-1) * BA + b2.unsqueeze(-1) * CB

    F = (D1 - D2) / _norm(D1 - D2)

    c = _cross2(E, F).abs()
    d = (E * F).sum(-1)

    return torch.where(
        (c < eps).unsqueeze(-1),
        E,
        torch.where((d > 0).unsqueeze(-1), F, -F),
    )


def boundary_scalings(
    R: torch.Tensor,
    RB: torch.Tensor,
    RC: torch.Tensor,
    Z: torch.Tensor,
    Bh: torch.Tensor,
    Th: torch.Tensor,
    gR: torch.Tensor,
    params: dict[str, torch.Tensor],
    closed_top: bool = True,
):
    """Compute base/crown radial tangent scales fB and fC."""
    m = R.shape[1]

    nB = _norm(R[0] - RB).squeeze(-1)
    gB = _norm(gR[0] / nB.unsqueeze(-1)).squeeze(-1)

    fB = (
        (2.0 ** 0.5)
        * nB
        / torch.sqrt(1.0 + (Z[0] - Bh).abs() * gB)
    )
    fB = fB * torch.exp(params["s_fB"])

    if closed_top:
        nC = _norm(RC - R[-1]).squeeze(-1)
        gC = _norm(gR[-1] / nC.unsqueeze(-1)).squeeze(-1)

        fC = (
            (2.0 ** 0.5)
            * nC
            / torch.sqrt(1.0 + (Th - Z[-1]).abs() * gC)
        )
        fC = fC * torch.exp(params["s_fC"])
    else:
        fC = torch.zeros(m, device=R.device, dtype=R.dtype)

    return fB, fC


# ---------------------------------------------------------------------------
# Reference-compatible Hermite surface
# ---------------------------------------------------------------------------

def hermite_surface(
    R: torch.Tensor,
    Z: torch.Tensor,
    RB: torch.Tensor,
    RC: torch.Tensor,
    Bh: torch.Tensor,
    Th: torch.Tensor,
    params: dict[str, torch.Tensor],
    n_u: int = 32,
    closed_top: bool = True,
    base_circular: bool = True,
    crown_circular: bool = True,
) -> torch.Tensor:
    """Evaluate the current NSSR surface.

    Returns
    -------
    torch.Tensor
        Shape ``(P,n_u,m,3)``. Patch order is:
        base cap, N-1 interior patches, optional crown cap.
    """
    if n_u < 2:
        raise ValueError("n_u must be >= 2")

    N, m, _ = R.shape
    dev, dt = R.device, R.dtype

    Bh, Th = apply_cap_heights(Z, Bh, Th, params)

    u = torch.linspace(0.0, 1.0, n_u, device=dev, dtype=dt)
    L0, L1, H0, H1 = _basis_components(u)

    L0 = L0.view(-1, 1, 1)
    L1 = L1.view(-1, 1, 1)
    H0 = H0.view(-1, 1, 1)
    H1 = H1.view(-1, 1, 1)

    # Avoid applying learned height multipliers twice.
    p_no_h = {
        key: value
        for key, value in params.items()
        if key not in ("s_bh", "s_th")
    }

    gR, gZ = tangent_field(
        R, Z, RB, RC, Bh, Th, p_no_h, closed_top
    )

    fB, fC = boundary_scalings(
        R, RB, RC, Z, Bh, Th, gR, p_no_h, closed_top
    )

    patches: list[torch.Tensor] = []

    # Base cap.
    dZ0 = (Z[0] - Bh).abs()

    if base_circular:
        gRB = boundary_directions(
            R, Z, RB, RC, Bh, Th, at_base=True
        )

        FR = (
            L0 * RB.view(1, 1, 2)
            + L1 * R[0].unsqueeze(0)
            + H0 * (fB.unsqueeze(-1) * gRB).unsqueeze(0)
            + 2.0 * dZ0 * H1 * gR[0].unsqueeze(0)
        )

        Fz = (
            (1.0 - u ** 2) * Bh
            + (u ** 2) * Z[0]
        ).view(-1, 1).expand(n_u, m)

    else:
        FR = (
            L0 * RB.view(1, 1, 2)
            + L1 * R[0].unsqueeze(0)
            + 2.0 * dZ0 * H1 * gR[0].unsqueeze(0)
        )

        Fz = (
            Bh * L0
            + Z[0] * L1
            + (Z[0] - Bh) * H0
        ).view(-1, 1) + (
            dZ0
            * gZ[0].unsqueeze(0)
            * H1.view(-1, 1)
        )

    patches.append(
        torch.cat([FR, Fz.unsqueeze(-1)], dim=-1)
    )

    # Interior patches.
    dZi = (Z[1:] - Z[:-1]).abs().view(1, -1, 1, 1)

    L0b = L0.unsqueeze(1)
    L1b = L1.unsqueeze(1)
    H0b = H0.unsqueeze(1)
    H1b = H1.unsqueeze(1)

    FR = (
        L0b * R[:-1].unsqueeze(0)
        + L1b * R[1:].unsqueeze(0)
        + dZi * (
            H0b * gR[:-1].unsqueeze(0)
            + H1b * gR[1:].unsqueeze(0)
        )
    )

    Fz = (
        L0b.squeeze(-1) * Z[:-1].view(1, -1, 1)
        + L1b.squeeze(-1) * Z[1:].view(1, -1, 1)
        + dZi.squeeze(-1) * (
            H0b.squeeze(-1) * gZ[:-1].unsqueeze(0)
            + H1b.squeeze(-1) * gZ[1:].unsqueeze(0)
        )
    )

    interior = torch.cat(
        [FR, Fz.unsqueeze(-1)],
        dim=-1,
    )

    patches.extend(interior.permute(1, 0, 2, 3))

    # Crown cap.
    if closed_top:
        dZN = (Th - Z[-1]).abs()

        if crown_circular:
            gRC = boundary_directions(
                R, Z, RB, RC, Bh, Th, at_base=False
            )

            FR = (
                L0 * R[-1].unsqueeze(0)
                + L1 * RC.view(1, 1, 2)
                + H1 * (fC.unsqueeze(-1) * gRC).unsqueeze(0)
                + 2.0 * dZN * H0 * gR[-1].unsqueeze(0)
            )

            Fz = (
                (1.0 - u) ** 2 * Z[-1]
                + u * (2.0 - u) * Th
            ).view(-1, 1).expand(n_u, m)

        else:
            FR = (
                L0 * R[-1].unsqueeze(0)
                + L1 * RC.view(1, 1, 2)
                + 2.0 * dZN * H0 * gR[-1].unsqueeze(0)
            )

            Fz = (
                Z[-1] * L0
                + Th * L1
            ).view(-1, 1) + (
                dZN * gZ[-1].unsqueeze(0) * H0.view(-1, 1)
            ) + (
                (Th - Z[-1]) * H1.view(-1, 1)
            )

        patches.append(
            torch.cat([FR, Fz.unsqueeze(-1)], dim=-1)
        )

    return torch.stack(patches, dim=0)


# ---------------------------------------------------------------------------
# Surface utilities
# ---------------------------------------------------------------------------

def surface_points(S: torch.Tensor) -> torch.Tensor:
    """Flatten patch grid to ``(P*n_u*m,3)``."""
    return S.reshape(-1, 3)


def surface_normals(S: torch.Tensor) -> torch.Tensor:
    """Finite-difference normals on the sampled (u,j) patch grids."""
    derivatives = sampled_surface_derivatives(S)
    return derivatives["normals"]


def zero_params(
    N: int,
    m: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Parameter dict reproducing the classical pipeline exactly."""
    z = lambda *shape: torch.zeros(
        *shape,
        device=device,
        dtype=dtype,
    )

    return {
        "s_a": z(N, m),
        "s_b": z(N, m),
        "s_tau": z(N, m),
        "s_fB": z(m),
        "s_fC": z(m),
        "s_bh": torch.zeros((), device=device, dtype=dtype),
        "s_th": torch.zeros((), device=device, dtype=dtype),
    }


# ---------------------------------------------------------------------------
# Sampled differential geometry
# ---------------------------------------------------------------------------

def _first_difference_nonperiodic(
    x: torch.Tensor,
    dim: int,
    spacing: float,
) -> torch.Tensor:
    """Second-order centered interior / one-sided boundary derivative."""
    if x.shape[dim] < 2:
        raise ValueError("need at least two samples for finite differences")

    out = torch.empty_like(x)

    sl_all = [slice(None)] * x.ndim

    if x.shape[dim] == 2:
        s0 = sl_all.copy()
        s1 = sl_all.copy()
        s0[dim] = 0
        s1[dim] = 1
        d = (x[tuple(s1)] - x[tuple(s0)]) / spacing
        out[tuple(s0)] = d
        out[tuple(s1)] = d
        return out

    # Interior centered differences.
    si = sl_all.copy()
    sp = sl_all.copy()
    sm = sl_all.copy()

    si[dim] = slice(1, -1)
    sp[dim] = slice(2, None)
    sm[dim] = slice(None, -2)

    out[tuple(si)] = (
        x[tuple(sp)] - x[tuple(sm)]
    ) / (2.0 * spacing)

    # Forward boundary.
    s0 = sl_all.copy()
    s1 = sl_all.copy()
    s2 = sl_all.copy()

    s0[dim] = 0
    s1[dim] = 1
    s2[dim] = 2

    out[tuple(s0)] = (
        -3.0 * x[tuple(s0)]
        + 4.0 * x[tuple(s1)]
        - x[tuple(s2)]
    ) / (2.0 * spacing)

    # Backward boundary.
    sn = sl_all.copy()
    sn1 = sl_all.copy()
    sn2 = sl_all.copy()

    sn[dim] = -1
    sn1[dim] = -2
    sn2[dim] = -3

    out[tuple(sn)] = (
        3.0 * x[tuple(sn)]
        - 4.0 * x[tuple(sn1)]
        + x[tuple(sn2)]
    ) / (2.0 * spacing)

    return out


def _first_difference_periodic(
    x: torch.Tensor,
    dim: int,
    spacing: float,
) -> torch.Tensor:
    """Centered periodic derivative."""
    return (
        torch.roll(x, shifts=-1, dims=dim)
        - torch.roll(x, shifts=1, dims=dim)
    ) / (2.0 * spacing)


def sampled_surface_derivatives(
    S: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Estimate derivatives on the reference-compatible sampled surface.

    The existing NSSR representation has an analytic Hermite u direction but
    only a discrete circumferential index j. To avoid changing geometry, this
    routine computes consistent finite-difference derivatives on the sampled
    patch grid.

    Parameters are normalized to:
        u in [0,1]
        v in [0,1) around the circumference.
    """
    if S.ndim != 4 or S.shape[-1] != 3:
        raise ValueError("S must have shape (P,n_u,m,3)")

    _, n_u, m, _ = S.shape

    if n_u < 2:
        raise ValueError("surface must contain at least two u samples")
    if m < 3:
        raise ValueError("surface must contain at least three circumference samples")

    du = 1.0 / (n_u - 1)
    dv = 1.0 / m

    Su = _first_difference_nonperiodic(S, dim=1, spacing=du)
    Sv = _first_difference_periodic(S, dim=2, spacing=dv)

    Suu = _first_difference_nonperiodic(Su, dim=1, spacing=du)
    Svv = _first_difference_periodic(Sv, dim=2, spacing=dv)

    Suv_a = _first_difference_periodic(Su, dim=2, spacing=dv)
    Suv_b = _first_difference_nonperiodic(Sv, dim=1, spacing=du)
    Suv = 0.5 * (Suv_a + Suv_b)

    normals = torch.linalg.cross(Su, Sv, dim=-1)
    normals = normals / _norm(normals)

    return {
        "Su": Su,
        "Sv": Sv,
        "Suu": Suu,
        "Suv": Suv,
        "Svv": Svv,
        "normals": normals,
    }


# ---------------------------------------------------------------------------
# V2 orchestration
# ---------------------------------------------------------------------------

def evaluate_geometry(
    R: torch.Tensor,
    Z: torch.Tensor,
    RB: torch.Tensor,
    RC: torch.Tensor,
    Bh: torch.Tensor,
    Th: torch.Tensor,
    params: dict[str, torch.Tensor],
    *,
    n_u: int = 32,
    closed_top: bool = True,
    base_circular: bool = True,
    crown_circular: bool = True,
    compute_jacobian: bool = True,
    compute_curvature: bool = True,
    run_validation: bool = True,
    reference_normal: Optional[torch.Tensor] = None,
    max_abs_curvature: float = 100.0,
    max_tangent_magnitude: Optional[float] = None,
    max_tangent_energy: Optional[float] = None,
) -> GeometryOutput:
    """Run the reference-compatible reconstruction plus V2 diagnostics.

    This function is the preferred high-level entry point for new V2 code.
    Existing training scripts may continue using ``hermite_surface`` during
    migration.

    ``reference_normal`` is strongly recommended when fold orientation must be
    judged against a fixed classical/target surface. If omitted, jacobian.py
    estimates a reference orientation independently from each patch.
    """
    S = hermite_surface(
        R,
        Z,
        RB,
        RC,
        Bh,
        Th,
        params,
        n_u=n_u,
        closed_top=closed_top,
        base_circular=base_circular,
        crown_circular=crown_circular,
    )

    # Recompute the same tangent field used by the surface for diagnostics.
    Bh_eff, Th_eff = apply_cap_heights(Z, Bh, Th, params)
    p_no_h = {
        key: value
        for key, value in params.items()
        if key not in ("s_bh", "s_th")
    }

    gR, gZ = tangent_field(
        R,
        Z,
        RB,
        RC,
        Bh_eff,
        Th_eff,
        p_no_h,
        closed_top=closed_top,
    )

    # Pack gR and gZ into a physical 3-D tangent field.
    tangent_xyz = torch.cat(
        [gR, gZ.unsqueeze(-1)],
        dim=-1,
    )

    tangent_mag = torch.linalg.vector_norm(tangent_xyz, dim=-1)
    tangent_eng = (tangent_xyz * tangent_xyz).sum(dim=-1)

    tangents = TangentField(
        radial=gR,
        axial=gZ,
        magnitude=tangent_mag,
        energy=tangent_eng,
    )

    d = sampled_surface_derivatives(S)

    derivatives = SurfaceDerivatives(
        Su=d["Su"],
        Sv=d["Sv"],
        Suu=d["Suu"],
        Suv=d["Suv"],
        Svv=d["Svv"],
    )

    surface = SurfacePatch(
        xyz=S,
        normals=d["normals"],
        uv=None,
        valid_mask=None,
    )

    evaluation = SurfaceEvaluation(
        xyz=S,
        Su=d["Su"],
        Sv=d["Sv"],
        Suu=d["Suu"],
        Suv=d["Suv"],
        Svv=d["Svv"],
        normals=d["normals"],
    )

    jac = None
    curvature = None
    validation = None

    if compute_jacobian or run_validation:
        jac = surface_jacobian(
            evaluation.Su,
            evaluation.Sv,
            reference_normal=reference_normal,
            reference_sample_dims=(1, 2),
        )

    if compute_curvature or run_validation:
        curvature = curvature_from_evaluation(evaluation)

    if run_validation:
        validation = validate_surface(
            evaluation,
            jac,
            curvature=curvature,
            tangents=tangent_xyz,
            max_abs_curvature=max_abs_curvature,
            max_tangent_magnitude=max_tangent_magnitude,
            max_tangent_energy=max_tangent_energy,
        )

        # Attach the per-sample primary validity mask to the surface.
        surface.valid_mask = (
            jac.valid_mask
            & (~jac.flipped_mask)
            & (~jac.degenerate_mask)
        )

    return GeometryOutput(
        surface=surface,
        tangents=tangents,
        derivatives=derivatives,
        jacobian=jac,
        curvature=curvature,
        validation=validation,
        statistics=None,
    )


__all__ = [
    "apply_cap_heights",
    "tangent_field",
    "boundary_directions",
    "boundary_scalings",
    "hermite_surface",
    "surface_points",
    "surface_normals",
    "zero_params",
    "sampled_surface_derivatives",
    "evaluate_geometry",
]
