"""NumPy twin of nssr/geometry.py (classical path + learnable fields).

Semantics reconciled with the reference implementation
(reference/surfaces_python_loops.py, reference/surfaces_pytorch.py):

  * crown tangent:  aR*(B-A) + 2*bR*(RC-B)  over  aR*|bZ| + 2*bR*|aZ|
  * fb, fc DIVIDE by sqrt(1 + |dZ|*||gR||/||B-P||)   (manuscript Eq. 25-26
    print a multiplication; the reference code divides — code wins)
  * gRB: E = (R0-RB)/||.||,  beta1 = (Bh - a1*(Z0-Bh)) / (Z1-Z0)
  * gRC: E = (RC-R_{N-1})/||.||,
         beta1 = (T - a1*(Z_{N-1}-Z_{N-2})) / (T - Z_{N-1})

tests/parity_check.py verifies this file against the reference pipeline on
banana / apple / vase; scripts/smoke_test.py verifies torch geometry.py
against this file.  params=None => classical.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12


def _nrm(v, keepdims=True):
    return np.sqrt((v * v).sum(-1, keepdims=keepdims) + EPS)


def zero_params_np(N, m, free_residual=False):
    p = {"s_a": np.zeros((N, m)), "s_b": np.zeros((N, m)),
         "s_tau": np.zeros((N, m)), "s_fB": np.zeros(m), "s_fC": np.zeros(m)}
    if free_residual:
        p["rho"] = np.zeros((N, m, 2))
    return p


def tangent_field_np(R, Z, RB, RC, Bh, Th, params=None, closed_top=True):
    N, m, _ = R.shape
    if params is None:
        params = zero_params_np(N, m)
    dR = np.empty((N + 1, m, 2)); dZ = np.empty(N + 1)
    dR[1:N] = R[1:] - R[:-1]; dR[0] = R[0] - RB
    dR[N] = (RC - R[-1]) if closed_top else dR[N - 1]
    dZ[1:N] = Z[1:] - Z[:-1]; dZ[0] = Z[0] - Bh
    dZ[N] = (Th - Z[-1]) if closed_top else dZ[N - 1]

    nrm = _nrm(dR, keepdims=False)                      # (N+1, m)
    a = nrm[1:].copy(); b = nrm[:-1].copy()             # (N, m)
    a[0] *= 2.0                                         # base factor
    if closed_top:
        b[N - 1] *= 2.0                                 # crown factor
    a *= np.exp(params["s_a"]); b *= np.exp(params["s_b"])

    denom = a * np.abs(dZ[:-1])[:, None] + b * np.abs(dZ[1:])[:, None] + EPS
    gR = (a[..., None] * dR[:-1] + b[..., None] * dR[1:]) / denom[..., None]
    gZ = (a * dZ[:-1][:, None] + b * dZ[1:][:, None]) / denom

    tiny = 1e-9
    coincident = (nrm[:-1] < tiny) & (nrm[1:] < tiny)
    gZ = np.where(coincident, np.sign(dZ[1:])[:, None], gZ)
    gR = np.where(coincident[..., None], 0.0, gR)

    tau = np.exp(params["s_tau"])
    gR = gR * tau[..., None]; gZ = gZ * tau
    if params.get("rho") is not None:
        gR = gR + params["rho"]
    return gR, gZ


def boundary_directions_np(R, Z, RB, RC, Bh, Th, at_base=True, eps=1e-8):
    N, m, _ = R.shape
    j = np.arange(m)
    a1 = 1.0 + (j % 15)
    if at_base:
        B, C = R[0], R[1]
        b1 = (Bh - a1 * (Z[0] - Bh)) / (Z[1] - Z[0])
        b2 = (Bh + a1 * (Z[0] - Bh)) / (Z[1] - Z[0])
        BA = B - RB
        CB = C - B
        E = BA / _nrm(BA)
    else:
        A, B = R[N - 2], R[N - 1]
        b1 = (Th - a1 * (Z[N - 1] - Z[N - 2])) / (Th - Z[N - 1])
        b2 = (Th + a1 * (Z[N - 1] - Z[N - 2])) / (Th - Z[N - 1])
        BA = B - A
        CB = RC - B
        E = CB / _nrm(CB)
    D1 = a1[:, None] * BA + b1[:, None] * CB
    D2 = -a1[:, None] * BA + b2[:, None] * CB
    F = (D1 - D2) / _nrm(D1 - D2)
    cross = E[:, 0] * F[:, 1] - E[:, 1] * F[:, 0]
    dot = (E * F).sum(-1)
    out = np.where(np.abs(cross)[:, None] < eps, E,
          np.where(dot[:, None] > 0, F, -F))
    return out


def boundary_scalings_np(R, RB, RC, Z, Bh, Th, gR, params=None,
                         closed_top=True):
    N, m, _ = R.shape
    if params is None:
        params = zero_params_np(N, m)
    nB = _nrm(R[0] - RB, keepdims=False)
    gB = _nrm(gR[0] / nB[:, None], keepdims=False)
    fB = np.sqrt(2.0) * nB / np.sqrt(1.0 + np.abs(Z[0] - Bh) * gB)
    fB = fB * np.exp(params["s_fB"])
    if closed_top:
        nC = _nrm(RC - R[-1], keepdims=False)
        gC = _nrm(gR[-1] / nC[:, None], keepdims=False)
        fC = np.sqrt(2.0) * nC / np.sqrt(1.0 + np.abs(Th - Z[-1]) * gC)
        fC = fC * np.exp(params["s_fC"])
    else:
        fC = np.zeros(m)
    return fB, fC


def hermite_surface_np(R, Z, RB, RC, Bh, Th, params=None, n_u=32,
                       closed_top=True, base_circular=True,
                       crown_circular=True):
    N, m, _ = R.shape
    u = np.linspace(0, 1, n_u)
    L0 = 1 - 3*u**2 + 2*u**3; L1 = 3*u**2 - 2*u**3
    H0 = u - 2*u**2 + u**3;   H1 = -u**2 + u**3
    c = lambda v: v[:, None, None]                     # (n_u,1,1)
    r = lambda v: v[:, None]                           # (n_u,1)
    gR, gZ = tangent_field_np(R, Z, RB, RC, Bh, Th, params, closed_top)
    fB, fC = boundary_scalings_np(R, RB, RC, Z, Bh, Th, gR, params,
                                  closed_top)
    P = []
    dZ0 = abs(Z[0] - Bh)
    if base_circular:
        gRB = boundary_directions_np(R, Z, RB, RC, Bh, Th, at_base=True)
        FR = (c(L0) * RB + c(L1) * R[0] + c(H0) * (fB[:, None] * gRB)
              + 2 * dZ0 * c(H1) * gR[0])
        Fz = ((1 - u**2) * Bh + u**2 * Z[0])[:, None] * np.ones((1, m))
    else:
        FR = c(L0) * RB + c(L1) * R[0] + 2 * dZ0 * c(H1) * gR[0]
        Fz = (Bh * L0 + Z[0] * L1 + (Z[0] - Bh) * H0)[:, None] \
             + dZ0 * gZ[0][None, :] * r(H1)
    P.append(np.concatenate([FR, Fz[..., None]], -1))

    for i in range(1, N):
        dZi = abs(Z[i] - Z[i - 1])
        FR = (c(L0) * R[i-1] + c(L1) * R[i]
              + dZi * (c(H0) * gR[i-1] + c(H1) * gR[i]))
        Fz = (L0 * Z[i-1] + L1 * Z[i])[:, None] \
             + dZi * (r(H0) * gZ[i-1][None, :] + r(H1) * gZ[i][None, :])
        P.append(np.concatenate([FR, Fz[..., None]], -1))

    if closed_top:
        dZN = abs(Th - Z[-1])
        if crown_circular:
            gRC = boundary_directions_np(R, Z, RB, RC, Bh, Th, at_base=False)
            FR = (c(L0) * R[-1] + c(L1) * RC + c(H1) * (fC[:, None] * gRC)
                  + 2 * dZN * c(H0) * gR[-1])
            Fz = ((1 - u)**2 * Z[-1] + u * (2 - u) * Th)[:, None] \
                 * np.ones((1, m))
        else:
            FR = c(L0) * R[-1] + c(L1) * RC + 2 * dZN * c(H0) * gR[-1]
            Fz = (Z[-1] * L0 + Th * L1 + (Th - Z[-1]) * H1)[:, None] \
                 + dZN * gZ[-1][None, :] * r(H0)
        P.append(np.concatenate([FR, Fz[..., None]], -1))
    return np.stack(P)                                  # (P, n_u, m, 3)
