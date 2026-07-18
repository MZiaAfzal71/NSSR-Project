from curves.curves_pytorch import curve_goodman_torch
from matplotlib.backends.backend_pdf import PdfPages
from data.shapes_3D_data import data_3d_shape
from surfaces.surfaces_pytorch import *
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import torch, os
import time, sys
import gc

if "google.colab" in sys.modules:
    platform_env = 'Colab'
elif "kaggle_secrets" in sys.modules or os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
    platform_env = 'Kaggle'
else:
    platform_env = 'Unknown'
    print("Running locally or in another environment")

matplotlib.use("Agg")

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Target paths
banana_path = PROJECT_ROOT / "results" / "csv files" / f"TensorV_CPU_stats_banana_{platform_env}.csv"
apple_path = PROJECT_ROOT / "results" / "csv files" / f"TensorV_CPU_stats_apple_{platform_env}.csv"
vase_path = PROJECT_ROOT / "results" / "csv files" / f"TensorV_CPU_stats_vase_{platform_env}.csv"
fig_path = PROJECT_ROOT / "results" / "pdf fig files" / f"pytorch_cpu_{platform_env}.pdf"

# Safely create directories
banana_path.parent.mkdir(parents=True, exist_ok=True)
apple_path.parent.mkdir(parents=True, exist_ok=True)
vase_path.parent.mkdir(parents=True, exist_ok=True)
fig_path.parent.mkdir(parents=True, exist_ok=True)


def run_pytorch_benchmark(N1, N2, dtypes, objects, device, R=10, warmup=1):
    columns = ['run_idx', 'object', 'n1', 'n2', 'dtype', 'device',
               't_data_create_cpu', 't_h2d_transfer',
               't_curves', 't_match_and_base_crown', 't_tangent',
               't_alloc_output', 't_kernel_surf_pts', 't_d2h_transfer',
               't_total']
    results = {ds: pd.DataFrame(columns=columns) for ds in objects}

    start_time_loops = time.perf_counter()
    for n1, n2 in zip(N1, N2):
        for dt in dtypes:
            for ds in objects:
                for i in range(-warmup, R):
                    is_warmup = (i < 0)

                    # --- Stage: create raw input data on CPU (simulates
                    # loading contour points from disk / a CPU-side source) ---
                    t0 = time.perf_counter()
                    I, Z, Null_Hts = data_3d_shape(ds, backend='torch', dtype=dt, device='cpu')
                    t_data_create_cpu = time.perf_counter() - t0

                    # --- Stage: explicit H2D transfer ---
                    t0 = time.perf_counter()

                    t_h2d_transfer = time.perf_counter() - t0

                    tot_pts, seg_pts = t_no_pts(I, n1)
                    N = len(seg_pts)
                    M = 4
                    step = tot_pts // M

                    # --- Stage: curve construction ---
                    t0 = time.perf_counter()
                    r = [curve_goodman_torch(I[k], seg_pts[k]) for k in range(len(I))]
                    r = torch.stack(r).to(device)

                    t_curves = time.perf_counter() - t0

                    # --- Stage: parameter matching + base/crown points ---
                    t0 = time.perf_counter()
                    Rm = match_parameters_torch_seq(r, N, tot_pts, M)
                    B_Point, C_Point = base_crown_pt(Rm, N, tot_pts, M, step)
                    if ds == 'apple':
                        B, T = Null_Hts[0], Null_Hts[1]
                        bt = ct = 'n'
                    else:
                        B, T = base_crown_ht(Rm, N, tot_pts, M, step, Z, Null_Hts)
                        bt = ct = 'y'
                    t_base_crown = time.perf_counter() - t0

                    # --- Stage: tangent vectors ---
                    t0 = time.perf_counter()
                    gR, gz, gRB, gRC, fb, fc = surf_tangent(Rm, N, tot_pts, Z, Null_Hts,
                                                            B_Point, C_Point, B, T, bt, ct)
                    t_tangent = time.perf_counter() - t0

                    # --- Stage: output buffer allocation ---
                    t0 = time.perf_counter()
                    FR = torch.zeros((N + 1, tot_pts + 1, n2 + 1, 2), dtype=dt, device=device)
                    Fz = torch.zeros((N + 1, tot_pts + 1, n2 + 1), dtype=dt, device=device)
                    t_alloc_output = time.perf_counter() - t0

                    # --- Stage: Hermite blending kernel (the actual O(Nmn) compute) ---
                    t0 = time.perf_counter()
                    surf_pts_inplace_vectorized(Rm, N, tot_pts, Z, B_Point, C_Point, B, T,
                                                gRB, gRC, fb, fc, gR, gz, FR, Fz, bt, ct, n2)
                    t_kernel_surf_pts = time.perf_counter() - t0

                    # --- Stage: D2H transfer of the final result ---
                    t0 = time.perf_counter()
                    t_d2h_transfer = time.perf_counter() - t0

                    t_total = (t_data_create_cpu + t_h2d_transfer + t_curves + t_base_crown +
                               t_tangent + t_alloc_output + t_kernel_surf_pts + t_d2h_transfer)

                    if not is_warmup:
                        results[ds].loc[len(results[ds])] = [
                            i, ds, n1, n2, str(dt).split('.')[-1], device.type,
                            t_data_create_cpu, t_h2d_transfer, t_curves,
                            t_base_crown, t_tangent, t_alloc_output,
                            t_kernel_surf_pts, t_d2h_transfer, t_total]
                        print(f"[{ds}] dtype={str(dt).split('.')[-1]} n1={n1} n2={n2} run {i + 1}/{R}: "
                              f"total={t_total:.4f}s kernel={t_kernel_surf_pts:.4f}s "
                              f"h2d={t_h2d_transfer:.5f}s d2h={t_d2h_transfer:.5f}s ")

                    del r, Rm, B_Point, C_Point, gR, gz, gRB, gRC, fb, fc, FR, Fz
                    gc.collect()

    end_time_loops = time.perf_counter()
    time_taken = end_time_loops - start_time_loops

    for ds in objects:
        if ds == 'banana':
            results[ds].to_csv(banana_path, index=False)
        elif ds == 'apple':
            results[ds].to_csv(apple_path, index=False)
        else:
            results[ds].to_csv(vase_path, index=False)

    print(f"\nSaved: {banana_path}, {apple_path}, and {vase_path}.\n")

    return time_taken


# --- Run it ---
device = torch.device("cpu")
dtypes = [torch.float64, torch.float32]
N1 = [300, 625, 1250]
N2 = [1200, 2500, 5000]
objects = ['banana', 'apple', 'vase']

print("\n" + "=" * 75)
print("🚀 Surface Reconstruction Benchmark (PyTorch Vectorized | CPU)")
print("=" * 75)
print("🔬 Purpose  : Performance benchmarking for surface reconstruction")
print("📊 Datasets : Banana, Apple, Vase")
print("🧮 Parameters:")
print(f"   • n1 (horizontal points) : {N1}")
print(f"   • n2 (vertical points)   : {N2}")
print(f"   • Data types             : {[str(dt).split('.')[-1] for dt in dtypes]}")
print("\n⏳ This is a CPU-intensive benchmark.")

time_taken = run_pytorch_benchmark(N1, N2, dtypes, objects, device, R=10, warmup=1)

print("\n" + "-" * 75)
print("⏱️ Benchmarking Phase Completed")
print("-" * 75)
print(f"🕒 Total benchmarking time: {time_taken / 60:.2f} minutes")
print("📈 Timing data collected for all datasets and parameter combinations.")
print("-" * 75 + "\n")

# Surface reconstruction of all three shapes [apple, banana, vase] using selected parameters to visualize the final 3D shape.
print("\n" + "=" * 75)
print("🎨 Final Surface Reconstruction & Visualization")
print("=" * 75 + "\n")

for ds in ['banana', 'apple', 'vase']:
    n1 = 500  # The number of points generated horizontally on each countour = 60 * n1
    n2 = 2000  # The number of points generated in vertical directions

    dt = torch.float64

    I, Z, Null_Hts = data_3d_shape(ds, backend='torch', dtype=dt, device=device)

    tot_pts, seg_pts = t_no_pts(I, n1)

    N = len(seg_pts)
    M = 4
    step = tot_pts // M

    r = []
    for k in range(len(I)):
        r.append(curve_goodman_torch(I[k], seg_pts[k]))

    r = torch.stack(r).to(device)
    R = match_parameters_torch_seq(r, N, tot_pts, M)
    B_Point, C_Point = base_crown_pt(R, N, tot_pts, M, step)

    if ds == 'apple':
        B = Null_Hts[0]
        T = Null_Hts[1]
        bt = ct = 'n'
    else:
        B, T = base_crown_ht(R, N, tot_pts, M, step, Z, Null_Hts)
        bt = ct = 'y'

    gR, gz, gRB, gRC, fb, fc = surf_tangent(R, N, tot_pts, Z, Null_Hts, B_Point, C_Point, B, T, bt, ct)

    if ds == 'banana':
        FR_banana = torch.zeros((N + 1, tot_pts + 1, n2 + 1, 2), dtype=dt, device=device)
        Fz_banana = torch.zeros((N + 1, tot_pts + 1, n2 + 1), dtype=dt, device=device)
        surf_pts_inplace_vectorized(R, N, tot_pts, Z, B_Point, C_Point, B, T, gRB, gRC, fb, fc, gR, gz, FR_banana,
                                    Fz_banana, bt, ct, n2)
        FR_banana = FR_banana.detach().cpu().numpy()
        Fz_banana = Fz_banana.detach().cpu().numpy()
    elif ds == 'apple':
        FR_apple = torch.zeros((N + 1, tot_pts + 1, n2 + 1, 2), dtype=dt, device=device)
        Fz_apple = torch.zeros((N + 1, tot_pts + 1, n2 + 1), dtype=dt, device=device)
        surf_pts_inplace_vectorized(R, N, tot_pts, Z, B_Point, C_Point, B, T, gRB, gRC, fb, fc, gR, gz, FR_apple,
                                    Fz_apple, bt, ct, n2)
        FR_apple = FR_apple.detach().cpu().numpy()
        Fz_apple = Fz_apple.detach().cpu().numpy()
    else:
        FR_vase = torch.zeros((N + 1, tot_pts + 1, n2 + 1, 2), dtype=dt, device=device)
        Fz_vase = torch.zeros((N + 1, tot_pts + 1, n2 + 1), dtype=dt, device=device)
        surf_pts_inplace_vectorized(R, N, tot_pts, Z, B_Point, C_Point, B, T, gRB, gRC, fb, fc, gR, gz, FR_vase,
                                    Fz_vase, bt, ct, n2)
        FR_vase = FR_vase.detach().cpu().numpy()
        Fz_vase = Fz_vase.detach().cpu().numpy()

with PdfPages(fig_path) as pdf:
    # ================= FIGURE 1 =================
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(FR_banana.shape[0]):
        ax.plot_wireframe(
            FR_banana[i, :, :, 0],
            FR_banana[i, :, :, 1],
            Fz_banana[i, :, :],
            rcount=10,
            ccount=3,
            linewidth=0.7,
            edgecolor='k'
        )

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_zlim(-1.5, 1.5)
    ax.view_init(elev=10, azim=10)
    ax.axis('off')

    fig.suptitle(
        "Figure 1: Banana's reconstruction\n"
        "Interface: PyTorch (CPU), vectorized implementation",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

    # ================= FIGURE 2 =================
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0, 2):
        ax.plot_surface(
            FR_banana[i, :, :, 0],
            FR_banana[i, :, :, 1],
            Fz_banana[i, :, :],
            rcount=20,
            ccount=10,
            linewidth=0.5,
            edgecolor='k',
            cmap='cool'
        )

    ax.view_init(elev=-70, azim=70)
    ax.axis('off')

    fig.suptitle(
        "Figure 2: Banana's base\n"
        "Interface: PyTorch (CPU)",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

    # ================= FIGURE 3 =================
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(6, 8):
        ax.plot_surface(
            FR_banana[i, :, :, 0],
            FR_banana[i, :, :, 1],
            Fz_banana[i, :, :],
            rcount=20,
            ccount=10,
            linewidth=0.5,
            edgecolor='k',
            cmap='cool'
        )

    ax.view_init(elev=40, azim=20)
    ax.axis('off')

    fig.suptitle(
        "Figure 3: Banana's crown\n"
        "Interface: PyTorch (CPU)",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

    # ================= FIGURE 4 =================
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(FR_apple.shape[0]):
        ax.plot_wireframe(
            FR_apple[i, :, :, 0],
            FR_apple[i, :, :, 1],
            Fz_apple[i, :, :],
            rcount=10,
            ccount=3,
            linewidth=0.7,
            edgecolor='k'
        )

    plt.axis('equal')
    ax.view_init(elev=10, azim=10)
    ax.axis('off')

    fig.suptitle(
        "Figure 4: Apple's reconstruction\n"
        "Interface: PyTorch (CPU), vectorized implementation",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

    # ================= FIGURE 5 =================
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0, 4):
        ax.plot_surface(
            FR_apple[i, :, :, 0],
            FR_apple[i, :, :, 1],
            Fz_apple[i, :, :],
            rcount=20,
            ccount=10,
            linewidth=0.5,
            edgecolor='k',
            cmap='cool'
        )

    plt.axis('equal')
    ax.view_init(elev=60, azim=100)
    ax.axis('off')

    fig.suptitle(
        "Figure 5: Apple's base\n"
        "Interface: PyTorch (CPU)",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

    # ================= FIGURE 6 =================
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(6, 10):
        ax.plot_surface(
            FR_apple[i, :, :, 0],
            FR_apple[i, :, :, 1],
            Fz_apple[i, :, :],
            rcount=20,
            ccount=10,
            linewidth=0.5,
            edgecolor='k',
            cmap='cool'
        )

    plt.axis('equal')
    ax.view_init(elev=-50, azim=100)
    ax.axis('off')

    fig.suptitle(
        "Figure 6: Apple's crown\n"
        "Interface: PyTorch (CPU)",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

    # ================= FIGURE 7 =================
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(FR_vase.shape[0] - 1):
        ax.plot_wireframe(
            FR_vase[i, :, :, 0],
            FR_vase[i, :, :, 1],
            Fz_vase[i, :, :],
            rcount=10,
            ccount=3,
            linewidth=0.7,
            edgecolor='k'
        )

    plt.axis('square')
    ax.view_init(elev=30, azim=10)
    ax.axis('off')

    fig.suptitle(
        "Figure 7: Vase's reconstruction\n"
        "Interface: PyTorch (CPU), vectorized implementation",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

    # ================= FIGURE 8 =================
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0, 5):
        ax.plot_surface(
            FR_vase[i, :, :, 0],
            FR_vase[i, :, :, 1],
            Fz_vase[i, :, :],
            rcount=20,
            ccount=10,
            linewidth=0.5,
            edgecolor='k',
            cmap='cool'
        )

    plt.axis('square')
    ax.view_init(elev=-60, azim=100)
    ax.axis('off')

    fig.suptitle(
        "Figure 8: Vase's base\n"
        "Interface: PyTorch (CPU)",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

print("\n" + "=" * 75)
print("📄 Visualization Completed Successfully 🎉")
print("=" * 75 + "\n\n")

