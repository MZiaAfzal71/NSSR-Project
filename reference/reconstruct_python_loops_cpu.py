from curves.curves_python_loops import curve_goodman
from matplotlib.backends.backend_pdf import PdfPages
from data.shapes_3D_data import data_3d_shape
from surfaces.surfaces_python_loops import *
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import gc, os, sys
import matplotlib
import time

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
banana_path = PROJECT_ROOT / "results" / "csv files" / f"Python_CPU_stats_banana_{platform_env}.csv"
apple_path = PROJECT_ROOT / "results" / "csv files" / f"Python_CPU_stats_apple_{platform_env}.csv"
vase_path = PROJECT_ROOT / "results" / "csv files" / f"Python_CPU_stats_vase_{platform_env}.csv"
fig_path = PROJECT_ROOT / "results" / "pdf fig files" / f"serial_python_cpu_{platform_env}.pdf"

# Safely create directories
banana_path.parent.mkdir(parents=True, exist_ok=True)
apple_path.parent.mkdir(parents=True, exist_ok=True)
vase_path.parent.mkdir(parents=True, exist_ok=True)
fig_path.parent.mkdir(parents=True, exist_ok=True)

python_stats_banana = pd.DataFrame(columns=['Start time', 'Curves ST', 'Curves ET', 'Tangents ST', 'Tangents ET/Surface ST', 'Surface ET/End Time', 'n2', 'dtype'])
python_stats_apple = pd.DataFrame(columns=['Start time', 'Curves ST', 'Curves ET', 'Tangents ST', 'Tangents ET/Surface ST', 'Surface ET/End Time', 'n2', 'dtype'])
python_stats_vase = pd.DataFrame(columns=['Start time', 'Curves ST', 'Curves ET', 'Tangents ST', 'Tangents ET/Surface ST', 'Surface ET/End Time', 'n2', 'dtype'])

dtypes = [np.float64, np.float32]

N1 = [300, 625, 1250] # The number of points generated horizontally on each countour = 60 * n1
N2 = [1200, 2500, 5000] # The number of points generated in vertical directions

print("\n" + "=" * 75)
print("🚀 Surface Reconstruction Benchmark (Python (Numpy-Assisted) | CPU)")
print("=" * 75)
print("🔬 Purpose  : Performance benchmarking for surface reconstruction")
print("📊 Datasets : Banana, Apple, Vase")
print("🧮 Parameters:")
print(f"   • n1 (horizontal points) : {N1}")
print(f"   • n2 (vertical points)   : {N2}")
print(f"   • Data types             : {[dt.__name__ for dt in dtypes]}")
print("\n⏳ This is a CPU-intensive benchmark.")
print("☕ Please be patient while computations are in progress...\n")

WARMUP = 1
R = 10

start_time_loops = time.perf_counter()
for n1, n2 in zip(N1, N2):
  for dt in dtypes:
    for ds in ['banana', 'apple', 'vase']:
      for i in range(-WARMUP, R):
        is_warmup = (i < 0)

        t0 = time.perf_counter()
        I, Z, Null_Hts = data_3d_shape(ds, dtype=dt)

        tot_pts, seg_pts = t_no_pts(I, n1)

        N = len(seg_pts)
        M = 4
        step = tot_pts // M

        t1 = time.perf_counter()
        r = []
        for k in range(len(I)):
            r.append(curve_goodman(I[k], seg_pts[k]))

        r = np.stack(r)
        t2 = time.perf_counter()
        R_mat = match_parameters(r, N, tot_pts, M)
        B_Point, C_Point = base_crown_pt(R_mat, N, tot_pts, M, step)

        if ds == 'apple':
          B = Null_Hts[0]
          T = Null_Hts[1]
          bt = ct = 'n'
        else:
          B, T = base_crown_ht(R_mat, N, tot_pts, M, step, Z, Null_Hts)
          bt = ct = 'y'

        t3 = time.perf_counter()
        gR, gz, gRB, gRC, fb, fc = surf_tangent(R_mat, N, tot_pts, Z, Null_Hts, B_Point, C_Point, B, T, bt, ct)
        t4 = time.perf_counter()
        FR, Fz = surf_pts(R_mat, N, tot_pts, Z, B_Point, C_Point, B, T, gRB, gRC, fb, fc, gR, gz, bt, ct, n2)
        t5 = time.perf_counter()

        elapsed_time = t5 - t0
        if not is_warmup:
            row = [t0, t1, t2, t3, t4, t5, n2, dt.__name__]
            if ds == 'banana':
              python_stats_banana.loc[len(python_stats_banana)] = row
            elif ds == 'apple':
              python_stats_apple.loc[len(python_stats_apple)] = row
            else:
              python_stats_vase.loc[len(python_stats_vase)] = row

            print(f"Data type: {dt.__name__} n1 : {n1} n2 : {n2} \n Elapsed time {i+1}th run for {ds}: {elapsed_time:.6f} seconds")

        del r, R_mat, B_Point, C_Point, gR, gz, gRC, fb, fc, FR, Fz
        gc.collect()


end_time_loops = time.perf_counter()
time_taken = end_time_loops - start_time_loops

python_stats_banana.to_csv(banana_path, index=False)
python_stats_apple.to_csv(apple_path, index=False)
python_stats_vase.to_csv(vase_path, index=False)

print(f"\nSaved: {banana_path}, {apple_path}, and {vase_path}.\n")

print("\n" + "-" * 75)
print("⏱️ Benchmarking Phase Completed")
print("-" * 75)
print(f"🕒 Total benchmarking time: {time_taken/60:.2f} minutes")
print("📈 Timing data collected for all datasets and parameter combinations.")
print("-" * 75 + "\n")




# Surface reconstruction of all three shapes [apple, banana, vase] using selected parameters to visualize the final 3D shape.
print("\n" + "=" * 75)
print("🎨 Final Surface Reconstruction & Visualization")
print("=" * 75)
print("📐 Objective : Reconstruct final 3D shapes using selected parameters")
print("🔍 Focus     : Visual demonstration (not benchmarking)")
print("📊 Shapes    : Banana, Apple, Vase")
print("⚙️ Parameters: n1 = 500, n2 = 2000, dtype = float64")
print("🖼️ Output    : High-quality 3D figures saved to PDF")
print("=" * 75 + "\n")

for ds in ['banana', 'apple', 'vase']:
  n1 = 500 # The number of points generated horizontally on each countour = 60 * n1
  n2 = 2000 # The number of points generated in vertical directions

  dt = np.float64

  I, Z, Null_Hts = data_3d_shape(ds, dtype=dt)

  tot_pts, seg_pts = t_no_pts(I, n1)

  N = len(seg_pts)
  M = 4
  step = tot_pts // M

  r = []
  for k in range(len(I)):
      r.append(curve_goodman(I[k], seg_pts[k]))

  r = np.stack(r)
  R = match_parameters(r, N, tot_pts, M)
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
    FR_banana, Fz_banana = surf_pts(R, N, tot_pts, Z, B_Point, C_Point, B, T, gRB, gRC, fb, fc, gR, gz, bt, ct, n2)
  elif ds == 'apple':
    FR_apple, Fz_apple = surf_pts(R, N, tot_pts, Z, B_Point, C_Point, B, T, gRB, gRC, fb, fc, gR, gz, bt, ct, n2)
  else:
    FR_vase, Fz_vase = surf_pts(R, N, tot_pts, Z, B_Point, C_Point, B, T, gRB, gRC, fb, fc, gR, gz, bt, ct, n2)

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
        "Interface: Python (CPU), loop based implementation",
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
        "Interface: Python (CPU)",
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
        "Interface: Python (CPU)",
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
        "Interface: Python (CPU), loop based implementation",
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
        "Interface: Python (CPU)",
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
        "Interface: Python (CPU)",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

    # ================= FIGURE 7 =================
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(FR_vase.shape[0]-1):
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
        "Interface: Python (CPU), loop based implementation",
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
        "Interface: Python (CPU)",
        fontsize=11,
        y=0.02
    )

    pdf.savefig(fig)
    plt.close(fig)

print("\n" + "=" * 75)
print("📄 Visualization Completed Successfully 🎉")
print("=" * 75 + "\n\n")

