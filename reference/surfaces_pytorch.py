import torch
import functools


def _cross2d_torch(a, b):
    """2D cross product scalar for arrays of 2-vectors: a,b shape (...,2) -> (...,)"""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


# Calculate total no. of points at each height
def t_no_pts(I, n1, device=None):
    """
    Calculate total number of points (tot_pts) and segment points (seg_pts)
    at each height, compatible with NumPy and PyTorch tensors.

    Parameters
    ----------
    I : list of arrays or tensors
        Each element I[i] contains the point indices at height i.

    Returns
    -------
    tot_pts : torch.Tensor
        LCM of data points at all heights.
    seg_pts : torch.Tensor
        Number of segment points for each height.
    """

    if device is None:
        device = I[0].device

    dtype = torch.int

    # No of points at each height
    P = torch.tensor([x.shape[0] for x in I], dtype=dtype, device=device)

    # LCM of all segment lengths
    lcm_data_pts = functools.reduce(torch.lcm, P)

    lcm_data_pts *= n1
    # Compute segment points for each height
    seg_pts = lcm_data_pts // P

    return lcm_data_pts.item(), seg_pts   # return scalar + tensor


# To obtain R's by Matching Parameters to aviod/minimize twistness in the surface
def match_parameters_torch_seq(r, N, tot_pts, M, device=None):
    if device is None:
        device = r.device

    D = r.shape[-1]
    step = tot_pts // M
    R = r.clone()
    dtype = torch.int

    idx_base = torch.arange(M, dtype=dtype, device=device) * step  # (M,)
    shifts = torch.arange(M, dtype=dtype, device=device)           # (M,)
    idx_shifted = (idx_base[None, :] + shifts[:, None] * step) % tot_pts  # (M,M)

    for i in range(N - 1):
        A = R[i, idx_base, :].unsqueeze(0)       # (1, M, D)
        B = r[i + 1, idx_shifted, :]             # (M, M, D)

        diff = A[:, None, :, :] - B.unsqueeze(0) # (1, M, M, D)
        dists = (diff ** 2).sum(dim=-1).sum(dim=-1).squeeze(0)
        best_shift = torch.argmin(dists)

        perm = (torch.arange(tot_pts, dtype=dtype, device=device) + best_shift * step) % tot_pts
        R[i + 1, :] = r[i + 1, perm]

    R = torch.cat([R, R[:, 0:1, :]], dim=1)
    return R


# Calculates Base and Crown Points
def base_crown_pt(R, N, tot_pts, M, step, device=None):
    """
    Vectorized PyTorch + NumPy compatible version of base_crown_pt().
    Computes base and crown points without Python loops or lists.
    """

    if device is None:
        device = R.device

    dtype = R.dtype

    # Prepare index arrays (for the M//2 pairs)
    half_M = M // 2
    j_idx = torch.arange(half_M, dtype=torch.int, device=device) * step
    k_idx = j_idx + (tot_pts // 2)

    # ----------------- BASE POINTS -----------------
    A = R[0, j_idx]
    B = R[0, k_idx]
    C = R[1, j_idx]
    D = R[1, k_idx]

    alpha = torch.linalg.norm(A - C, dim=1)
    beta  = torch.linalg.norm(A - B, dim=1)
    gamma = torch.linalg.norm(B - D, dim=1)

    # weighted sum for base points
    C_B = (gamma.unsqueeze(1) * A +
            alpha.unsqueeze(1) * B +
            beta.unsqueeze(1) * (A + B)) / (alpha + 2 * beta + gamma).unsqueeze(1)

    # ----------------- CROWN POINTS -----------------
    A = R[N-1, j_idx]
    B = R[N-1, k_idx]
    C = R[N-2, j_idx]
    D = R[N-2, k_idx]

    alpha = torch.linalg.norm(A - C, dim=1)
    beta  = torch.linalg.norm(A - B, dim=1)
    gamma = torch.linalg.norm(B - D, dim=1)

    C_C = (gamma.unsqueeze(1) * A +
            alpha.unsqueeze(1) * B +
            beta.unsqueeze(1) * (A + B)) / (alpha + 2 * beta + gamma).unsqueeze(1)

    # Mean over all pairs
    B_Point = 2.0 / M * torch.sum(C_B, dim=0)
    C_Point = 2.0 / M * torch.sum(C_C, dim=0)

    return B_Point, C_Point


# Calculates Base and Crown Heights
def base_crown_ht(R, N, tot_pts, M, step, Z, Null_Hts, device=None):
    """
    Compute average base (B) and crown (T) heights using circle-fitting.
    Works with NumPy arrays or PyTorch tensors (torch preferred for GPU).

    Parameters
    ----------
    R : (N, tot_pts, 2)
    N : int
    tot_pts : int
    M : int
    step : int
    Z : (N,)
    Null_Hts : (2,)
    """
    if device is None:
      device = R.device


    # Sample index pairs (M equally spaced + opposite point)
    i_idx = torch.arange(M, dtype=torch.int, device=device)
    j_idx = i_idx * step
    k_idx = (i_idx * step + tot_pts // 2) % tot_pts

    # --- Base points ---
    A_b = R[0, j_idx]
    B_b = R[0, k_idx]
    C_b = R[1, j_idx]
    Z0, Z1 = Z[0], Z[1]

    # --- Crown points ---
    A_c = R[N - 1, j_idx]
    B_c = R[N - 1, k_idx]
    C_c = R[N - 2, j_idx]
    ZNm1, ZNm2 = Z[N - 1], Z[N - 2]

    def compute_heights(A, B, C, Z1, Z2, null_ht, reverse=False):
        """Shared block for base/crown height computation."""
        x1, y1 = A[:, 0], A[:, 1]
        x2, y2 = B[:, 0], B[:, 1]
        x3, y3 = C[:, 0], C[:, 1]
        z1 = torch.full_like(x1, Z1)
        z2 = torch.full_like(x2, Z1)
        z3 = torch.full_like(x3, Z2)

        # --- Determinant-based quantities (vectorized) ---
        def det3(a, b, c):
            return (
                a[:, 0] * (b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1])
                - a[:, 1] * (b[:, 0] * c[:, 2] - b[:, 2] * c[:, 0])
                + a[:, 2] * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
            )

        x4 = -det3(
            torch.stack([y1, z1, torch.ones_like(y1)], dim=1),
            torch.stack([y2, z2, torch.ones_like(y2)], dim=1),
            torch.stack([y3, z3, torch.ones_like(y3)], dim=1)
        )
        y4 = det3(
            torch.stack([x1, z1, torch.ones_like(x1)], dim=1),
            torch.stack([x2, z2, torch.ones_like(x2)], dim=1),
            torch.stack([x3, z3, torch.ones_like(x3)], dim=1)
        )
        z4 = -det3(
            torch.stack([x1, y1, torch.ones_like(x1)], dim=1),
            torch.stack([x2, y2, torch.ones_like(x2)], dim=1),
            torch.stack([x3, y3, torch.ones_like(x3)], dim=1)
        )

        b1 = -(x1**2 + y1**2 + z1**2)
        b2 = -(x2**2 + y2**2 + z2**2)
        b3 = -(x3**2 + y3**2 + z3**2)
        b4 = det3(
            torch.stack([x1, y1, z1], dim=1),
            torch.stack([x2, y2, z2], dim=1),
            torch.stack([x3, y3, z3], dim=1)
        )

        A_mat = torch.stack([
            torch.stack([2*x1, 2*y1, 2*z1, torch.ones_like(x1)], dim=1),
            torch.stack([2*x2, 2*y2, 2*z2, torch.ones_like(x2)], dim=1),
            torch.stack([2*x3, 2*y3, 2*z3, torch.ones_like(x3)], dim=1),
            torch.stack([x4,   y4,   z4,   torch.zeros_like(x4)], dim=1)
        ], dim=1)
        B_vec = torch.stack([b1, b2, b3, b4], dim=1)

        # --- Solve linear system (batched) ---
        X = torch.linalg.solve(A_mat, B_vec.unsqueeze(-1)).squeeze(-1)

        u, v, w, d = X.T
        mdx1, mdy1, mdz1 = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2
        cx2, cy2, cz2 = -u, -v, -w
        a = (cx2 - mdx1) / (cz2 - mdz1)
        b = (cy2 - mdy1) / (cz2 - mdz1)

        p1 = a**2 + b**2 + 1
        p2 = 2*a*(mdx1 - a*mdz1) + 2*b*(mdy1 - b*mdz1) - 2*u*a + 2*v*b + 2*w
        p3 = ((mdx1 - a*mdz1)**2 + (mdy1 - b*mdz1)**2
              + 2*u*(mdx1 - a*mdz1) + 2*v*(mdy1 - b*mdz1) + d)

        disc = p2**2 - 4*p1*p3
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0))
        root1 = (-p2 + sqrt_disc) / (2*p1)
        root2 = (-p2 - sqrt_disc) / (2*p1)

        if reverse:
            if null_ht > Z1:
                z_out = torch.minimum(null_ht, torch.maximum(root1, root2))
            else:
                z_out = torch.maximum(null_ht, torch.minimum(root1, root2))
        else:
            if null_ht < Z1:
                z_out = torch.maximum(null_ht, torch.minimum(root1, root2))
            else:
                z_out = torch.minimum(null_ht, torch.maximum(root1, root2))
        return z_out

    # --- Compute both heights ---
    zkb = compute_heights(A_b, B_b, C_b, Z0, Z1, Null_Hts[0], reverse=False)
    zkc = compute_heights(A_c, B_c, C_c, ZNm1, ZNm2, Null_Hts[1], reverse=True)

    B = torch.mean(zkb)
    T = torch.mean(zkc)
    return B, T


# Calculates Tangent to the surface
def surf_tangent(R, N, tot_pts, Z, Null_Hts, RB, RC, Bh, T, base_null='y', crown_null='y',  eps=1e-12, device=None):
    """
    Vectorized surf_tangent that works with numpy or torch arrays.
    - R: shape (N, M, 2) where M == tot_pts + 1 (consistent with your loops)
    - Z: shape (N,)
    - RB, RC: shape (2,)
    Returns: (gR, gz, gRB, gRC, fb, fc) of same library type as R
    """
    if device is None:
      device = R.device

    dtype = R.dtype

    # shapes
    M = R.shape[1]            # should be tot_pts + 1
    assert M >= 1
    # Preallocate
    gR = torch.zeros((N, M, 2), dtype=dtype, device=device)
    gz = torch.zeros((N, M), dtype=dtype, device=device)
    gRB = torch.zeros((M, 2), dtype=dtype, device=device)
    gRC = torch.zeros((M, 2), dtype=dtype, device=device)
    fb = torch.zeros((M,), dtype=dtype, device=device)
    fc = torch.zeros((M,), dtype=dtype, device=device)

    if base_null in ('n', 'N'):
        Bh = Null_Hts[0]
    if crown_null in ('n', 'N'):
        T = Null_Hts[1]

    # --------------------------
    # Interior points: i=1..N-2
    # --------------------------
    if N > 2:
        i_idx = torch.arange(1, N - 1, dtype=torch.int, device=device)                # shape (K,)
        K = i_idx.shape[0]

        # A, B, C shapes (K, M, 2)
        A = R[i_idx - 1]    # (K, M, 2)
        B = R[i_idx]        # (K, M, 2)
        C = R[i_idx + 1]    # (K, M, 2)

        # alphaR, betaR: (K, M)
        alphaR = torch.linalg.norm(C - B, axis=2)
        betaR  = torch.linalg.norm(B - A, axis=2)

        # alphaZ, betaZ per i (K,)
        alphaZ = Z[i_idx + 1] - Z[i_idx]     # (K,)
        betaZ  = Z[i_idx] - Z[i_idx - 1]     # (K,)

        # Expand for broadcasting to (K, M)
        alphaZ_b = torch.abs(alphaZ)[:, None]
        betaZ_b  = torch.abs(betaZ)[:, None]

        # denom (K, M) with safe fallback
        denom = alphaR * betaZ_b + betaR * alphaZ_b
        denom_safe = torch.where(denom == 0, eps, denom)

        # numerator (K, M, 2)
        numerator = alphaR[..., None] * (B - A) + betaR[..., None] * (C - B)
        gR_block = numerator / denom_safe[..., None]
        gR[i_idx, :, :] = gR_block

        # gz numerator: alphaR*betaZ + betaR*alphaZ (K, M)
        num_gz = alphaR * betaZ[:, None] + betaR * alphaZ[:, None]
        gz_block = num_gz / denom_safe
        gz[i_idx, :] = gz_block

    # --------------------------
    # Base section i = 0
    # --------------------------
    if N >= 2:
        A = RB               # (2,)
        B0 = R[0]            # (M,2)
        C0 = R[1]            # (M,2)
        alphaR0 = torch.linalg.norm(C0 - B0, axis=1)   # (M,)
        betaR0  = torch.linalg.norm(B0 - A, axis=1)    # (M,)
        alphaZ0 = Z[1] - Z[0]
        betaZ0  = Z[0] - Bh
        denom0 = 2 * alphaR0 * abs(betaZ0) + betaR0 * abs(alphaZ0)
        denom0_safe = torch.where(denom0 == 0, eps, denom0)
        numer0 = (2 * alphaR0)[:, None] * (B0 - A) + betaR0[:, None] * (C0 - B0)
        gR[0, :, :] = numer0 / denom0_safe[:, None]
        gz[0, :] = (2 * alphaR0 * betaZ0 + betaR0 * alphaZ0) / denom0_safe

    # --------------------------
    # Crown section i = N - 1
    # --------------------------
    if N >= 2:
        AN = R[N - 2]    # (M,2)
        BN = R[N - 1]    # (M,2)
        # alphaRN: norm of RC - BN per row
        alphaRN = torch.linalg.norm(RC[None, :] - BN, axis=1)
        betaRN  = torch.linalg.norm(BN - AN, axis=1)
        alphaZN = (T - Z[N - 1]).item()
        betaZN  = (Z[N - 1] - Z[N - 2]).item()
        denomN = alphaRN * abs(betaZN) + 2 * betaRN * abs(alphaZN)
        denomN_safe = torch.where(denomN == 0, eps, denomN)
        numerN = alphaRN[:, None] * (BN - AN) + 2 * betaRN[:, None] * (RC - BN)
        gR[N - 1, :, :] = numerN / denomN_safe[:, None]
        gz[N - 1, :] = (alphaRN * betaZN + 2 * betaRN * alphaZN) / denomN_safe

    # --------------------------
    # gRB vectorized across j
    # --------------------------
    j_idx = torch.arange(M, dtype=torch.int, device=device)
    # deterministic alpha1 (1.0 + j%15)
    alpha1_vec = 1.0 + (j_idx % 15) #.astype(float)
    # beta1 vector computations (vector)
    # For base:
    denom_Z01 = Z[1] - Z[0]
    beta1_vec = (Bh - alpha1_vec * (Z[0] - Bh)) / denom_Z01
    alpha2_vec = -alpha1_vec
    beta2_vec = (Bh - alpha2_vec * (Z[0] - Bh)) / denom_Z01

    B0 = R[0]  # (M,2)
    C0 = R[1]  # (M,2)
    # D1, D2 shapes (M,2)
    D1 = alpha1_vec[:, None] * (B0 - RB) + beta1_vec[:, None] * (C0 - B0)
    D2 = alpha2_vec[:, None] * (B0 - RB) + beta2_vec[:, None] * (C0 - B0)

    # E and F: normalized row-wise
    def _normalize_rows(X):
        norms = torch.linalg.norm(X, axis=1)
        norms_safe = torch.where(norms == 0, eps, norms)
        return X / norms_safe[:, None]

    E = _normalize_rows(B0 - RB)   # (M,2)
    F = _normalize_rows(D1 - D2)   # (M,2)

    # cross2d and dot
    cross_vals = E[:, 0] * F[:, 1] - E[:, 1] * F[:, 0]
    dot_vals = torch.einsum('ij,ij->i', E, F)

    choose_E = torch.abs(cross_vals) < (1e-8)
    choose_F = (~choose_E) & (dot_vals > 0)
    choose_negF = (~choose_E) & (~choose_F)

    gRB[choose_E, :] = E[choose_E, :]
    gRB[choose_F, :] = F[choose_F, :]
    gRB[choose_negF, :] = -F[choose_negF, :]

    # --------------------------
    # gRC vectorized across j
    # --------------------------
    # similar for crown
    alpha1_vec = 1.0 + (j_idx % 15) #.astype(float)
    denom_ZN = (T - Z[N - 1]).item()
    beta1_vec = (T - alpha1_vec * (Z[N - 1] - Z[N - 2])) / (T - Z[N - 1])
    alpha2_vec = -alpha1_vec
    beta2_vec = (T - alpha2_vec * (Z[N - 1] - Z[N - 2])) / (T - Z[N - 1])

    AN = R[N - 2]
    BN = R[N - 1]
    D1 = alpha1_vec[:, None] * (BN - AN) + beta1_vec[:, None] * (RC - BN)
    D2 = alpha2_vec[:, None] * (BN - AN) + beta2_vec[:, None] * (RC - BN)

    E = _normalize_rows(RC - BN)   # (M,2)
    F = _normalize_rows(D1 - D2)   # (M,2)

    cross_vals = E[:, 0] * F[:, 1] - E[:, 1] * F[:, 0]
    dot_vals = torch.einsum('ij,ij->i', E, F)

    choose_E = torch.abs(cross_vals) < (1e-8)
    choose_F = (~choose_E) & (dot_vals > 0)
    choose_negF = (~choose_E) & (~choose_F)

    gRC[choose_E, :] = E[choose_E, :]
    gRC[choose_F, :] = F[choose_F, :]
    gRC[choose_negF, :] = -F[choose_negF, :]

    # --------------------------
    # fb and fc vectorized
    # --------------------------
    B0 = R[0]   # (M,2)
    Cmag = torch.linalg.norm(B0 - RB, axis=1)
    Cmag_safe = torch.where(Cmag == 0, eps, Cmag)
    tmp = gR[0] / Cmag_safe[:, None]
    tmp_norm = torch.linalg.norm(tmp, axis=1)
    fb = torch.sqrt(torch.tensor(2.0)) * Cmag / torch.sqrt(1.0 + abs(Z[0] - Bh) * tmp_norm)

    BN = R[N - 1]
    CmagN = torch.linalg.norm(RC - BN, axis=1)
    CmagN_safe = torch.where(CmagN == 0, eps, CmagN)
    tmp = gR[N - 1] / CmagN_safe[:, None]
    tmp_norm = torch.linalg.norm(tmp, axis=1)
    fc = torch.sqrt(torch.tensor(2.0)) * CmagN / torch.sqrt(1.0 + abs(T - Z[N - 1]) * tmp_norm)

    return gR, gz, gRB, gRC, fb, fc


def surf_pts_inplace_vectorized(R, N, tot_pts, Z, RB, RC, Bh, T,
                                gRB, gRC, fb, fc, gR, gz, FR, Fz,
                                base_circular, crown_circular, n, device=None):
    """
    Fully-vectorized, in-place surf_pts that avoids allocating new large tensors
    by reusing preallocated temporary buffers.

    Preconditions (must be satisfied by caller):
      - FR shape == (N+1, M, n+1, 2) and already zeroed
      - Fz shape == (N+1, M, n+1) and already zeroed
      - All inputs on same device and dtype, requires_grad=False
      - device, dtype inferred from R if not provided

    This function writes into FR and Fz in-place and returns None.
    """

    if device is None:
        device = R.device
    dtype = R.dtype

    M = R.shape[1]  # tot_pts + 1

    # precompute 1D blending polynomials (n+1,)
    u = torch.linspace(0.0, 1.0, n + 1, dtype=dtype, device=device)  # (n+1,)
    # keep them on correct dtype/device
    L0 = (1.0 - 3.0 * u**2 + 2.0 * u**3).to(dtype=dtype, device=device)   # (n+1,)
    L1 = (3.0 * u**2 - 2.0 * u**3).to(dtype=dtype, device=device)
    H0 = (u - 2.0 * u**2 + u**3).to(dtype=dtype, device=device)
    H1 = (-u**2 + u**3).to(dtype=dtype, device=device)

    # ---- Prepare broadcasting views used in out= ops ----
    # shapes used by vectorized ops:
    # L0_row, etc. have shape (1, 1, n+1) to broadcast with (K,M,n+1)
    L0_row = L0.view(1, 1, -1)
    L1_row = L1.view(1, 1, -1)
    H0_row = H0.view(1, 1, -1)
    H1_row = H1.view(1, 1, -1)

    # ---- Interior block temp buffers ----
    # For N>1, K = N-1 interior segments to fill FR[1:N,...]
    if N > 1:
        K = N - 1

        # Preallocate a reusable large temp buffer for interior computations:
        # tmp_KMn will be reused for X and Y and Fz partial computations.
        # Shape: (K, M, n+1)
        tmp_KMn = torch.empty((K, M, n + 1), dtype=dtype, device=device)

        # Views / slices for convenience (no new allocation):
        A = R[0:K, :, :]       # (K, M, 2)
        B = R[1:K+1, :, :]     # (K, M, 2)
        gA = gR[0:K, :, :]     # (K, M, 2)
        gB = gR[1:K+1, :, :]   # (K, M, 2)
        gzA = gz[0:K, :]       # (K, M)
        gzB = gz[1:K+1, :]     # (K, M)

        # deltaZ_col shape (K,1,1) used in broadcasted multiplications
        deltaZ = torch.abs(Z[1:N] - Z[:N - 1]).to(dtype=dtype, device=device)  # (K,)
        deltaZ_col = deltaZ.view(K, 1, 1)

        # ---- Fill FR[1:N, :, :, 0] (X) in-place without temporaries ----
        FRx = FR[1:N, :, :, 0]   # view shape (K, M, n+1)
        # Step 1: FRx = A[...,0] * L0_row  (use out=FRx)
        torch.mul(A[..., 0].unsqueeze(-1), L0_row, out=FRx)   # writes into FRx directly

        # Step 2: FRx += B[...,0] * L1_row  (use addcmul pattern -> FRx += B * L1)
        # but addcmul_ expects two tensors multiplied; we can use addcmul_ with second operand L1_row
        # FRx += (B[...,0].unsqueeze(-1) * L1_row)
        FRx.addcmul_(B[..., 0].unsqueeze(-1), L1_row, value=1.0)

        # Step 3: FRx += deltaZ_col * (gA[...,0]*H0_row + gB[...,0]*H1_row)
        # compute tmp_KMn = gA[...,0].unsqueeze(-1) * H0_row  (out to tmp)
        torch.mul(gA[..., 0].unsqueeze(-1), H0_row, out=tmp_KMn)
        # tmp_KMn += gB[...,0].unsqueeze(-1) * H1_row
        tmp_KMn.addcmul_(gB[..., 0].unsqueeze(-1), H1_row, value=1.0)
        # Now scale by deltaZ_col and add to FRx in-place
        FRx.add_(tmp_KMn * deltaZ_col)   # tmp_KMn * deltaZ_col will broadcast deltaZ_col (small) — produces a *view-like* expression but this multiplication will allocate a temporary of shape (K,M,n+1) only if PyTorch cannot fuse; however it's a cheap elementwise op using tmp_KMn (already allocated) times small deltaZ_col (broadcast) and then added into FRx in one operation. If you want strictly no alloc even for this, use a loop over K slices to multiply by scalar deltaZ[k] per slice (see below).

        # ---- Fill FR[1:N, :, :, 1] (Y) ----
        FRy = FR[1:N, :, :, 1]
        torch.mul(A[..., 1].unsqueeze(-1), L0_row, out=FRy)
        FRy.addcmul_(B[..., 1].unsqueeze(-1), L1_row, value=1.0)
        # reuse tmp_KMn for Y temp
        torch.mul(gA[..., 1].unsqueeze(-1), H0_row, out=tmp_KMn)
        tmp_KMn.addcmul_(gB[..., 1].unsqueeze(-1), H1_row, value=1.0)
        FRy.add_(tmp_KMn * deltaZ_col)

        # ---- Fill Fz[1:N, :, :] (interior) ----
        Fz_interior = Fz[1:N, :, :]   # shape (K, M, n+1)
        # Fz = Z_prev * L0_row + Z_next * L1_row + deltaZ_col * (gzA.unsqueeze(-1)*H0_row + gzB.unsqueeze(-1)*H1_row)
        # compute tmp_KMn = gzA.unsqueeze(-1) * H0_row
        torch.mul(gzA.unsqueeze(-1), H0_row, out=tmp_KMn)
        tmp_KMn.addcmul_(gzB.unsqueeze(-1), H1_row, value=1.0)
        # now Fz_interior = Z_prev * L0_row (out) then add Z_next*L1_row and deltaZ_col*tmp_KMn
        Z_prev = Z[:N - 1].to(dtype=dtype, device=device).view(K, 1, 1)   # (K,1,1)
        Z_next = Z[1:N].to(dtype=dtype, device=device).view(K, 1, 1)
        # write Z_prev * L0_row into Fz_interior
        Fz_interior.addcmul_(Z_prev, L0_row, value=1.0)
        # add Z_next * L1_row
        Fz_interior.addcmul_(Z_next, L1_row, value=1.0)
        # add deltaZ_col * tmp_KMn
        Fz_interior.add_(tmp_KMn * deltaZ_col)

    else:
        # No interior segments; nothing to do here
        tmp_KMn = None
        deltaZ_col = None

    # ---- Base (index 0) vectorized, in-place ----
    # FR[0, :, :, 0] and FR[0, :, :, 1] are (M, n+1)
    FR0x = FR[0, :, :, 0]   # (M, n+1)
    FR0y = FR[0, :, :, 1]
    # Broadcast helpers
    L0_m = L0.view(1, -1)   # (1, n+1)
    L1_m = L1.view(1, -1)
    H0_m = H0.view(1, -1)
    H1_m = H1.view(1, -1)

    B0 = R[0, :, :]   # (M,2)

    if isinstance(base_circular, str) and base_circular in ('y', 'Y'):
        # FR0x = L0_m * RB[0] + L1_m * B0[:,0].unsqueeze(-1) + H0_m*(fb.unsqueeze(-1)*gRB[:,0].unsqueeze(-1)) + H1_m*(2*abs(Z0-Bh)*gR[0,:,0].unsqueeze(-1))
        # stepwise, in-place
        FR0x.fill_(0.0)
        # add L0*RB[0] (scalar broadcast)
        FR0x.add_(L0_m * RB[0])
        # add L1 * B0[:,0]
        FR0x.add_(B0[:, 0].unsqueeze(-1) * L1_m)
        # add H0 * (fb * gRB[:,0])
        FR0x.add_((fb.unsqueeze(-1) * gRB[:, 0].unsqueeze(-1)) * H0_m)
        # add H1 * (2 * |Z0-Bh| * gR[0,:,0])
        FR0x.add_((2.0 * torch.abs(Z[0] - Bh) * gR[0, :, 0].unsqueeze(-1)) * H1_m)

        FR0y.fill_(0.0)
        FR0y.add_(L0_m * RB[1])
        FR0y.add_(B0[:, 1].unsqueeze(-1) * L1_m)
        FR0y.add_((fb.unsqueeze(-1) * gRB[:, 1].unsqueeze(-1)) * H0_m)
        FR0y.add_((2.0 * torch.abs(Z[0] - Bh) * gR[0, :, 1].unsqueeze(-1)) * H1_m)

        # Fz[0] = (1-u^2)*Bh + u^2*Z0
        Fz0 = Fz[0, :, :]   # (M, n+1)
        Fz0.fill_(0.0)
        Fz0.add_((1.0 - u**2).view(1, -1) * Bh)
        Fz0.add_((u**2).view(1, -1) * Z[0])
    else:
        # Non-circular base
        FR0x.fill_(0.0)
        FR0x.add_(L0_m * RB[0])
        FR0x.add_(B0[:, 0].unsqueeze(-1) * L1_m)
        FR0x.add_((2.0 * torch.abs(Z[0] - Bh) * gR[0, :, 0].unsqueeze(-1)) * H1_m)

        FR0y.fill_(0.0)
        FR0y.add_(L0_m * RB[1])
        FR0y.add_(B0[:, 1].unsqueeze(-1) * L1_m)
        FR0y.add_((2.0 * torch.abs(Z[0] - Bh) * gR[0, :, 1].unsqueeze(-1)) * H1_m)

        Fz0 = Fz[0, :, :]
        Fz0.fill_(0.0)
        Fz0.add_(Bh * L0_m)
        Fz0.add_(Z[0] * L1_m)
        Fz0.add_(((Z[0] - Bh) * H0).view(1, -1))
        Fz0.add_(torch.abs(Z[0] - Bh) * (gz[0, :].unsqueeze(-1) * H1_m))

    # ---- Crown (index N) vectorized, in-place ----
    FRNx = FR[N, :, :, 0]
    FRNy = FR[N, :, :, 1]
    BN = R[N - 1, :, :]  # (M,2)

    if isinstance(crown_circular, str) and crown_circular in ('y', 'Y'):
        FRNx.fill_(0.0)
        FRNx.add_(BN[:, 0].unsqueeze(-1) * L0_m)
        FRNx.add_(L1_m * RC[0])
        FRNx.add_((fc.unsqueeze(-1) * gRC[:, 0].unsqueeze(-1)) * H1_m)
        FRNx.add_((2.0 * torch.abs(T - Z[N - 1]) * gR[N - 1, :, 0].unsqueeze(-1)) * H0_m)

        FRNy.fill_(0.0)
        FRNy.add_(BN[:, 1].unsqueeze(-1) * L0_m)
        FRNy.add_(L1_m * RC[1])
        FRNy.add_((fc.unsqueeze(-1) * gRC[:, 1].unsqueeze(-1)) * H1_m)
        FRNy.add_((2.0 * torch.abs(T - Z[N - 1]) * gR[N - 1, :, 1].unsqueeze(-1)) * H0_m)

        FzN = Fz[N, :, :]
        FzN.fill_(0.0)
        FzN.add_(((1.0 - u)**2).view(1, -1) * Z[N - 1])
        FzN.add_((u * (2.0 - u)).view(1, -1) * T)
    else:
        FRNx.fill_(0.0)
        FRNx.add_(BN[:, 0].unsqueeze(-1) * L0_m)
        FRNx.add_(L1_m * RC[0])
        FRNx.add_((2.0 * torch.abs(T - Z[N - 1]) * gR[N - 1, :, 0].unsqueeze(-1)) * H0_m)

        FRNy.fill_(0.0)
        FRNy.add_(BN[:, 1].unsqueeze(-1) * L0_m)
        FRNy.add_(L1_m * RC[1])
        FRNy.add_((2.0 * torch.abs(T - Z[N - 1]) * gR[N - 1, :, 1].unsqueeze(-1)) * H0_m)

        FzN = Fz[N, :, :]
        FzN.fill_(0.0)
        FzN.add_((Z[N - 1] * L0_m))
        FzN.add_(T * L1_m)
        FzN.add_(torch.abs(T - Z[N - 1]) * (gz[N - 1, :].unsqueeze(-1) * H0_m))
        FzN.add_(((T - Z[N - 1]) * H1).view(1, -1))

    # Done — function modifies FR and Fz in-place; no return necessary
    return None