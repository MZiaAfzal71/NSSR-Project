# surfaces_python_loops.py
# -----------------------
# Pure Python (loop-based) surface construction utilities.
# Implements point counting, parameter matching, base/crown geometry,
# surface tangents, and final surface point generation.
#
# ⚠️ This file uses standard Python + NumPy loops only.
# Numba-accelerated and PyTorch/GPU versions are provided in separate modules.

import numpy as np


def cross2d(a, b):
    """2D cross product scalar for arrays of 2-vectors: a,b shape (...,2) -> (...,)"""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


# Calculate total no. of points at each height
def t_no_pts(I, n1, dtype=np.int64):
    N = len(I)
    P = np.zeros(N, dtype=dtype)
    for i in range(N):
        P[i] = len(I[i])

    lcm_data_pts = np.lcm.reduce(P)
    tot_pts = lcm_data_pts * n1

    seg_pts = np.zeros(N, dtype=dtype)
    for i in range(N):
        seg_pts[i] = tot_pts / P[i]

    return tot_pts, seg_pts


# To obtain R's by Matching Parameters
def match_parameters(r, N, tot_pts, M=4):
    R = r.copy()
    step = tot_pts // M
    for i in range(N-1):
        for j in range(M):
            L = []
            #Find distance d(j) between two contours for each j
            for k in range(M):
                L.append(np.linalg.norm(R[i, k*step] - r[i+1, (k+j)*step%tot_pts])**2)
            d = sum(L)
            if j == 0:
                minval = d
                reqj = j
            elif d < minval:
                minval = d
                reqj = j
        for l in range(tot_pts):
            R[i + 1, l] = r[i + 1, (l + reqj * step) % tot_pts]
        #r[i+1] = R[i+1]

    #Add the first point at the end of each contour to close it
    # Create an empty list to store the modified sub-arrays
    modified_subarrays = []

    # Iterate through each sub-array
    for sub_arr in R:
        # Get the first point of the current sub-array
        first_point = sub_arr[0]
        # Append the first point to the end of the current sub-array
        # The 'axis=0' argument ensures the point is added as a new row
        modified_sub_arr = np.append(sub_arr, [first_point], axis=0)
        modified_subarrays.append(modified_sub_arr)

    # Convert the list of modified sub-arrays back to a NumPy array
    R = np.array(modified_subarrays)
    return R


# Calculates Base and Crown Points
def base_crown_pt(R, N, tot_pts, M, step, dtype=None):
    B_acc = np.zeros(2, dtype=dtype)
    C_acc = np.zeros(2, dtype=dtype)

    count = M // 2  # total iterations

    for i in range(count):
        j = i * step
        k = j + tot_pts // 2

        # -------- Base Point --------
        A = R[0, j]
        B = R[0, k]
        C = R[1, j]
        D = R[1, k]
        alpha = np.sqrt(np.sum((A - C) * (A - C)))
        beta  = np.sqrt(np.sum((A - B) * (A - B)))
        gamma = np.sqrt(np.sum((B - D) * (B - D)))
        denom = alpha + 2 * beta + gamma
        B_acc += (gamma * A + alpha * B + beta * (A + B)) / denom

        # -------- Crown Point --------
        A = R[N-1, j]
        B = R[N-1, k]
        C = R[N-2, j]
        D = R[N-2, k]
        alpha = np.sqrt(np.sum((A - C) * (A - C)))
        beta  = np.sqrt(np.sum((A - B) * (A - B)))
        gamma = np.sqrt(np.sum((B - D) * (B - D)))
        denom = alpha + 2 * beta + gamma
        C_acc += (gamma * A + alpha * B + beta * (A + B)) / denom

    scale = 2.0 / M
    B_Point = scale * B_acc
    C_Point = scale * C_acc
    return B_Point, C_Point


# Calculates Base and Crown Heights
def base_crown_ht(R, N, tot_pts, M, step, Z, Null_Hts, dtype=np.float32):
    """
    Compute average base (B) and crown (T) heights using circle-fitting
    approach. Uses NumPy arrays and vectorized indexing.

    Parameters
    ----------
    R : ndarray, shape (N, tot_pts, 2)
        2D ring coordinates stacked along N levels.
    N : int
        Number of levels along Z axis.
    tot_pts : int
        Number of points per level.
    M : int
        Number of samples to use.
    step : int
        Step size for sampling indices.
    Z : ndarray, shape (N,)
        Z-coordinates for each level.
    Null_Hts : ndarray, shape (2,)
        Reference heights for base and crown.

    Returns
    -------
    B : float
        Average base height.
    T : float
        Average crown height.
    """

    zkb = np.zeros(M, dtype=dtype)
    zkc = np.zeros(M, dtype=dtype)

    for i in range(M):
        j = i * step
        k = (i * step + tot_pts // 2) % tot_pts

        # ---------- Base height ----------
        x1, y1, z1 = R[0, j, 0], R[0, j, 1], Z[0]
        x2, y2, z2 = R[0, k, 0], R[0, k, 1], Z[0]
        x3, y3, z3 = R[1, j, 0], R[1, j, 1], Z[1]

        x4 = -np.linalg.det([[y1, z1, 1], [y2, z2, 1], [y3, z3, 1]])
        y4 =  np.linalg.det([[x1, z1, 1], [x2, z2, 1], [x3, z3, 1]])
        z4 = -np.linalg.det([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])

        b1 = -(x1**2 + y1**2 + z1**2)
        b2 = -(x2**2 + y2**2 + z2**2)
        b3 = -(x3**2 + y3**2 + z3**2)
        b4 =  np.linalg.det([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])

        X = np.linalg.solve(
            [[2*x1, 2*y1, 2*z1, 1],
             [2*x2, 2*y2, 2*z2, 1],
             [2*x3, 2*y3, 2*z3, 1],
             [x4,   y4,   z4,   0]],
            [b1, b2, b3, b4]
        )

        u, v, w, d = X
        mdx1, mdy1, mdz1 = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2
        cx2, cy2, cz2 = -u, -v, -w

        a = (cx2 - mdx1) / (cz2 - mdz1)
        b = (cy2 - mdy1) / (cz2 - mdz1)

        p1 = a**2 + b**2 + 1
        p2 = (2*a*(mdx1 - a*mdz1) +
              2*b*(mdy1 - b*mdz1) -
              2*u*a + 2*v*b + 2*w)
        p3 = ((mdx1 - a*mdz1)**2 +
              (mdy1 - b*mdz1)**2 +
              2*u*(mdx1 - a*mdz1) +
              2*v*(mdy1 - b*mdz1) + d)

        roots = np.roots([p1, p2, p3])
        if Null_Hts[0] < z1:
            zkb[i] = max(Null_Hts[0], np.min(roots))
        else:
            zkb[i] = min(Null_Hts[0], np.max(roots))

        # ---------- Crown height ----------
        x1, y1, z1 = R[N-1, j, 0], R[N-1, j, 1], Z[N-1]
        x2, y2, z2 = R[N-1, k, 0], R[N-1, k, 1], Z[N-1]
        x3, y3, z3 = R[N-2, j, 0], R[N-2, j, 1], Z[N-2]

        x4 = -np.linalg.det([[y1, z1, 1], [y2, z2, 1], [y3, z3, 1]])
        y4 =  np.linalg.det([[x1, z1, 1], [x2, z2, 1], [x3, z3, 1]])
        z4 = -np.linalg.det([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])

        b1 = -(x1**2 + y1**2 + z1**2)
        b2 = -(x2**2 + y2**2 + z2**2)
        b3 = -(x3**2 + y3**2 + z3**2)
        b4 =  np.linalg.det([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])

        X = np.linalg.solve(
            [[2*x1, 2*y1, 2*z1, 1],
             [2*x2, 2*y2, 2*z2, 1],
             [2*x3, 2*y3, 2*z3, 1],
             [x4,   y4,   z4,   0]],
            [b1, b2, b3, b4]
        )

        u, v, w, d = X
        mdx1, mdy1, mdz1 = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2
        cx2, cy2, cz2 = -u, -v, -w

        a = (cx2 - mdx1) / (cz2 - mdz1)
        b = (cy2 - mdy1) / (cz2 - mdz1)

        p1 = a**2 + b**2 + 1
        p2 = (2*a*(mdx1 - a*mdz1) +
              2*b*(mdy1 - b*mdz1) -
              2*u*a + 2*v*b + 2*w)
        p3 = ((mdx1 - a*mdz1)**2 +
              (mdy1 - b*mdz1)**2 +
              2*u*(mdx1 - a*mdz1) +
              2*v*(mdy1 - b*mdz1) + d)

        roots = np.roots([p1, p2, p3])
        if Null_Hts[1] > z1:
            zkc[i] = min(Null_Hts[1], np.max(roots))
        else:
            zkc[i] = max(Null_Hts[1], np.min(roots))

    B = np.mean(zkb)
    T = np.mean(zkc)
    return B, T


def surf_tangent(R, N, tot_pts, Z, Null_Hts, RB, RC, Bh, T, base_null='y', crown_null='y', dtype=np.float32):
    # Preallocate fixed-size arrays
    gR = np.zeros((N, tot_pts + 1, 2), dtype=dtype)
    gz = np.zeros((N, tot_pts + 1), dtype=dtype)
    gRB = np.zeros((tot_pts + 1, 2), dtype=dtype)
    gRC = np.zeros((tot_pts + 1, 2), dtype=dtype)
    fb = np.zeros(tot_pts + 1, dtype=dtype)
    fc = np.zeros(tot_pts + 1, dtype=dtype)

    if base_null in ('n', 'N'):
        Bh = Null_Hts[0]
    if crown_null in ('n', 'N'):
        T = Null_Hts[1]

    # --- Interior points ---
    for i in range(1, N - 1):
        for j in range(tot_pts + 1):
            A = R[i - 1, j]
            B = R[i, j]
            C = R[i + 1, j]

            if np.allclose(A, B) and np.allclose(A, C):
                gR[i, j] = np.array([0.0, 0.0])
                gz[i, j] = np.copysign(1.0, Z[i + 1] - Z[i])
            else:
                alphaR = np.linalg.norm(C - B)
                betaR = np.linalg.norm(B - A)
                alphaZ = Z[i + 1] - Z[i]
                betaZ = Z[i] - Z[i - 1]
                numerator = alphaR * (B - A) + betaR * (C - B)
                denom = alphaR * abs(betaZ) + betaR * abs(alphaZ)
                gR[i, j] = numerator / denom
                gz[i, j] = (alphaR * betaZ + betaR * alphaZ) / denom

    # --- Base section ---
    for j in range(tot_pts + 1):
        A = RB
        B = R[0, j]
        C = R[1, j]
        alphaR = np.linalg.norm(C - B)
        betaR = np.linalg.norm(B - A)
        alphaZ = Z[1] - Z[0]
        betaZ = Z[0] - Bh
        numerator = 2 * alphaR * (B - A) + betaR * (C - B)
        denom = 2 * alphaR * abs(betaZ) + betaR * abs(alphaZ)
        gR[0, j] = numerator / denom
        gz[0, j] = (2 * alphaR * betaZ + betaR * alphaZ) / denom

    # --- Crown section ---
    for j in range(tot_pts + 1):
        A = R[N - 2, j]
        B = R[N - 1, j]
        C = RC
        alphaR = np.linalg.norm(C - B)
        betaR = np.linalg.norm(B - A)
        alphaZ = T - Z[N - 1]
        betaZ = Z[N - 1] - Z[N - 2]
        numerator = alphaR * (B - A) + 2 * betaR * (C - B)
        denom = alphaR * abs(betaZ) + 2 * betaR * abs(alphaZ)
        gR[N - 1, j] = numerator / denom
        gz[N - 1, j] = (alphaR * betaZ + 2 * betaR * alphaZ) / denom

    # --- Base tangent vectors gRB ---
    for j in range(tot_pts + 1):
        A = RB
        B = R[0, j]
        C = R[1, j]

        # deterministic alpha (avoid random)
        alpha1 = 1.0 + (j % 15)
        beta1 = (Bh - alpha1 * (Z[0] - Bh)) / (Z[1] - Z[0])
        D1 = alpha1 * (B - A) + beta1 * (C - B)

        alpha2 = -alpha1
        beta2 = (Bh - alpha2 * (Z[0] - Bh)) / (Z[1] - Z[0])
        D2 = alpha2 * (B - A) + beta2 * (C - B)

        E = (B - A) / np.linalg.norm(B - A)
        F = (D1 - D2) / np.linalg.norm(D1 - D2)
        cross1 = cross2d(E, F)
        dot1 = np.dot(E, F)
        if abs(cross1) < 1e-8:
            gRB[j] = E
        elif dot1 > 0:
            gRB[j] = F
        else:
            gRB[j] = -F

    # --- Crown tangent vectors gRC ---
    for j in range(tot_pts + 1):
        A = R[N - 2, j]
        B = R[N - 1, j]
        C = RC

        alpha1 = 1.0 + (j % 15)
        beta1 = (T - alpha1 * (Z[N - 1] - Z[N - 2])) / (T - Z[N - 1])
        D1 = alpha1 * (B - A) + beta1 * (C - B)

        alpha2 = -alpha1
        beta2 = (T - alpha2 * (Z[N - 1] - Z[N - 2])) / (T - Z[N - 1])
        D2 = alpha2 * (B - A) + beta2 * (C - B)

        E = (C - B) / np.linalg.norm(C - B)
        F = (D1 - D2) / np.linalg.norm(D1 - D2)
        cross1 = cross2d(E, F)
        dot1 = np.dot(E, F)
        if abs(cross1) < 1e-8:
            gRC[j] = E
        elif dot1 > 0:
            gRC[j] = F
        else:
            gRC[j] = -F

    # --- fb and fc computations ---
    for j in range(tot_pts + 1):
        A = RB
        B = R[0, j]
        Cmag = np.linalg.norm(B - A)
        fb[j] = np.sqrt(2.0) * Cmag / np.sqrt(1.0 + abs(Z[0] - Bh) * np.linalg.norm(gR[0, j] / Cmag))

    for j in range(tot_pts + 1):
        A = RC
        B = R[N - 1, j]
        Cmag = np.linalg.norm(A - B)
        fc[j] = np.sqrt(2.0) * Cmag / np.sqrt(1.0 + abs(T - Z[N - 1]) * np.linalg.norm(gR[N - 1, j] / Cmag))

    return gR, gz, gRB, gRC, fb, fc


def surf_pts(R, N, tot_pts, Z, RB, RC, Bh, T, gRB, gRC, fb, fc, gR, gz, base_circular, crown_circular, n, dtype=np.float32):
    u = np.linspace(0, 1, n + 1, dtype=dtype)
    L0 = 1 - 3 * u**2 + 2 * u**3
    L1 = 3 * u**2 - 2 * u**3
    H0 = u - 2 * u**2 + u**3
    H1 = -u**2 + u**3

    FR = np.zeros((N + 1, tot_pts + 1, n + 1, 2), dtype=dtype)
    Fz = np.zeros((N + 1, tot_pts + 1, n + 1), dtype=dtype)

    # --- Interior ---
    for i in range(1, N):
        deltaZ = abs(Z[i] - Z[i - 1])
        for j in range(tot_pts + 1):
            FR[i, j, :, 0] = L0 * R[i - 1, j, 0] + L1 * R[i, j, 0] + deltaZ * (H0 * gR[i - 1, j, 0] + H1 * gR[i, j, 0])
            FR[i, j, :, 1] = L0 * R[i - 1, j, 1] + L1 * R[i, j, 1] + deltaZ * (H0 * gR[i - 1, j, 1] + H1 * gR[i, j, 1])
            Fz[i, j, :] = Z[i - 1] * L0 + Z[i] * L1 + deltaZ * (gz[i - 1, j] * H0 + gz[i, j] * H1)

    # --- Base ---
    for j in range(tot_pts + 1):
        if base_circular in ('y', 'Y'):  # For Numba string restriction
            FR[0, j, :, 0] = L0 * RB[0] + L1 * R[0, j, 0] + fb[j] * H0 * gRB[j, 0] + 2 * abs(Z[0] - Bh) * H1 * gR[0, j, 0]
            FR[0, j, :, 1] = L0 * RB[1] + L1 * R[0, j, 1] + fb[j] * H0 * gRB[j, 1] + 2 * abs(Z[0] - Bh) * H1 * gR[0, j, 1]
            Fz[0, j, :] = (1 - u**2) * Bh + u**2 * Z[0]
        else:
            FR[0, j, :, 0] = L0 * RB[0] + L1 * R[0, j, 0] + 2 * abs(Z[0] - Bh) * H1 * gR[0, j, 0]
            FR[0, j, :, 1] = L0 * RB[1] + L1 * R[0, j, 1] + 2 * abs(Z[0] - Bh) * H1 * gR[0, j, 1]
            Fz[0, j, :] = Bh * L0 + Z[0] * L1 + (Z[0] - Bh) * H0 + abs(Z[0] - Bh) * gz[0, j] * H1

    # --- Crown ---
    for j in range(tot_pts + 1):
        if crown_circular in ('y', 'Y'):
            FR[N, j, :, 0] = L0 * R[N - 1, j, 0] + L1 * RC[0] + fc[j] * H1 * gRC[j, 0] + 2 * abs(T - Z[N - 1]) * H0 * gR[N - 1, j, 0]
            FR[N, j, :, 1] = L0 * R[N - 1, j, 1] + L1 * RC[1] + fc[j] * H1 * gRC[j, 1] + 2 * abs(T - Z[N - 1]) * H0 * gR[N - 1, j, 1]
            Fz[N, j, :] = (1 - u)**2 * Z[N - 1] + u * (2 - u) * T
        else:
            FR[N, j, :, 0] = L0 * R[N - 1, j, 0] + L1 * RC[0] + 2 * abs(T - Z[N - 1]) * H0 * gR[N - 1, j, 0]
            FR[N, j, :, 1] = L0 * R[N - 1, j, 1] + L1 * RC[1] + 2 * abs(T - Z[N - 1]) * H0 * gR[N - 1, j, 1]
            Fz[N, j, :] = Z[N - 1] * L0 + T * L1 + abs(T - Z[N - 1]) * gz[N - 1, j] * H0 + (T - Z[N - 1]) * H1

    return FR, Fz