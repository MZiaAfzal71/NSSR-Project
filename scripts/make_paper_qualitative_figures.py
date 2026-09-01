#!/usr/bin/env python3
"""Build the three qualitative NSSR figures proposed for the paper.

The default configuration is deliberately tied to the paper's strongest
slice-count operating point, N=15:

1. ``figure_designer_comparison_N15``
   Banana, apple, and vase: input contours | classical | projected NSSR.
2. ``figure_real_comparison_N15``
   Four curated held-out real objects: GT | input | classical | projected NSSR.
3. ``figure_qualitative_combined_N15``
   A compact gallery of the final designer and real NSSR surfaces.

Designer surfaces are reconstructed with the N=15 synthetic-trained
``best.pt`` checkpoint.  Real surfaces use the N=15 real-trained ``best.pt``
checkpoint, matching the real-only qualitative results already in the repo.
Both paths use the safety-first checkpoint and inference-time projection.

When ``data/real/test_N15.pkl`` is available, the real panels are freshly
reconstructed and rendered with PyVista.  The public repository does not
currently contain that pickle or the source meshes, so ``--real-source auto``
falls back to the existing N=15 four-panel PNGs in
``results/paper_real/figures/N15``.  The fallback only affects the real panels;
designer surfaces are always freshly reconstructed and PyVista-rendered.

Typical use (from the repository root)
--------------------------------------
Install the packages used by the NSSR pipeline plus the renderer, then run:

    python scripts/make_paper_qualitative_figures.py

If the real test pickle is present:

    python scripts/make_paper_qualitative_figures.py \
        --real-source reconstruct --real-data data/real

Use one mixed-domain checkpoint for both groups if desired:

    python scripts/make_paper_qualitative_figures.py \
        --designer-ckpt runs/paper_mixed_100ep/N15/best.pt \
        --real-ckpt runs/paper_mixed_100ep/N15/best.pt \
        --real-metrics-dir results/paper_real/mixed_to_real \
        --real-source reconstruct

The script writes PNG and PDF versions plus a ready-to-edit LaTeX inclusion
snippet.  The PDF files embed the high-resolution PyVista raster panels.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# These examples were chosen after inspecting all 21 held-out N=15 renders.
# Three have strong post-projection CD improvements and distinct silhouettes;
# thingi_126195 is retained because it demonstrates tangent-stage repair on a
# geometrically complex object instead of showing only no-op projections.
DEFAULT_REAL_OBJECTS = (
    "thingi_90440",
    "thingi_62991",
    "thingi_79195",
    "thingi_126195",
)

DESIGNER_NAMES = ("banana", "apple", "vase")
DESIGNER_LABELS = {
    "banana": "Banana",
    "apple": "Apple",
    "vase": "Vase",
}

PANEL_TITLES = {
    "gt": "Ground-truth samples",
    "input": "Input contours",
    "classical": "Classical Hermite",
    "nssr": "NSSR + safety projection",
}

SURFACE_COLORS = {
    "classical": "#8CA6B8",
    "nssr": "#D87954",
    "gt": "#A8ADB3",
}

# The original designer geometry is preserved.  These paired shades only
# change its visual material: muted classical versus richer NSSR rendering.
DESIGNER_SURFACE_COLORS = {
    "banana": {"classical": "#C5A73D", "nssr": "#F2C94C"},
    "apple": {"classical": "#87352A", "nssr": "#C94732"},
    "vase": {"classical": "#3F778A", "nssr": "#5FA6B8"},
}

CONTOUR_COLORS = (
    "#3B82F6", "#06B6D4", "#10B981", "#84CC16",
    "#EAB308", "#F97316", "#EF4444", "#EC4899",
    "#A855F7", "#6366F1",
)


@dataclass
class FigureRow:
    """Images and annotations for one object row in a comparison figure."""

    key: str
    label: str
    panels: dict[str, np.ndarray]
    note: str = ""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate NSSR designer, real, and combined paper figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--N", type=int, default=15,
                    help="training/evaluation slice-count checkpoint")
    ap.add_argument("--m", type=int, default=128,
                    help="unique circumferential samples for real objects")
    ap.add_argument("--n-u", type=int, default=32,
                    help="meridional samples per patch for display")
    ap.add_argument("--designer-segment-points", type=int, default=32,
                    help="Goodman samples per designer contour segment; 32 gives m=128")
    ap.add_argument("--designer-ckpt", default="",
                    help="default: runs/paper_full_100ep/N{N}/best.pt")
    ap.add_argument("--real-ckpt", default="",
                    help="default: runs/paper_real_100ep/N{N}/best.pt")
    ap.add_argument("--real-data", default="data/real",
                    help="directory containing test_N{N}.pkl")
    ap.add_argument("--real-metrics-dir", default="results/paper_real/real_only",
                    help="directory containing N{N}_validate/projection.csv")
    ap.add_argument("--real-index-map", default="",
                    help="default: results/paper_real/base_crown_N{N}.csv")
    ap.add_argument("--existing-real-dir", default="",
                    help="default: results/paper_real/figures/N{N}")
    ap.add_argument(
        "--real-source", choices=("auto", "reconstruct", "existing"),
        default="auto",
        help="fresh reconstruction or existing four-panel repository PNGs",
    )
    ap.add_argument(
        "--real-objects", nargs="+", default=list(DEFAULT_REAL_OBJECTS),
        help="Thingiverse stems or integer test-set indices, in display order",
    )
    ap.add_argument("--device", default="auto",
                    help="auto, cpu, cuda, or a PyTorch device string")
    ap.add_argument("--c-bound", type=float, default=1.0,
                    help="must match checkpoint training")
    ap.add_argument("--max-cap-fold", type=float, default=1e-3)
    ap.add_argument("--projection-iters", type=int, default=40)
    ap.add_argument("--elev", type=float, default=16.0,
                    help="camera elevation in degrees")
    ap.add_argument("--azim", type=float, default=-55.0,
                    help="camera azimuth in degrees")
    ap.add_argument("--panel-pixels", type=int, default=760,
                    help="square PyVista render size")
    ap.add_argument("--dpi", type=int, default=400,
                    help="composite figure raster DPI")
    ap.add_argument("--formats", nargs="+", choices=("png", "pdf", "tif"),
                    default=("png", "pdf"))
    ap.add_argument("--out", default="results/paper_qualitative_figures")
    ap.add_argument(
        "--make", nargs="+", choices=("designer", "real", "combined"),
        default=("designer", "real", "combined"),
        help="which outputs to create; combined needs both object groups",
    )
    ap.add_argument("--no-panel-letters", action="store_true")
    ap.add_argument(
        "--check", action="store_true",
        help="report resolved files and dependencies without rendering",
    )
    return ap.parse_args()


def repo_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def require_modules(names: Iterable[str], purpose: str) -> None:
    missing = [name for name in names if not module_available(name)]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing package(s) for {purpose}: {joined}. "
            "Install the NSSR environment and add `pyvista vtk` for rendering."
        )


def resolved_paths(a: argparse.Namespace) -> dict[str, Path]:
    N = a.N
    return {
        "designer_ckpt": repo_path(
            a.designer_ckpt or f"runs/paper_full_100ep/N{N}/best.pt"
        ),
        "real_ckpt": repo_path(
            a.real_ckpt or f"runs/paper_real_100ep/N{N}/best.pt"
        ),
        "real_pickle": repo_path(a.real_data) / f"test_N{N}.pkl",
        "validate_csv": repo_path(a.real_metrics_dir) / f"N{N}_validate.csv",
        "projection_csv": repo_path(a.real_metrics_dir) / f"N{N}_projection.csv",
        "index_map_csv": repo_path(
            a.real_index_map or f"results/paper_real/base_crown_N{N}.csv"
        ),
        "existing_real_dir": repo_path(
            a.existing_real_dir or f"results/paper_real/figures/N{N}"
        ),
        "out": repo_path(a.out),
    }


def determine_real_source(a: argparse.Namespace, paths: Mapping[str, Path]) -> str:
    if a.real_source != "auto":
        return a.real_source
    if paths["real_pickle"].is_file():
        return "reconstruct"
    return "existing"


def checkpoint_domain_label(path: Path) -> str:
    text = str(path).lower()
    if "paper_mixed" in text or "/mixed" in text:
        return "mixed-trained"
    if "paper_real" in text or "/real" in text:
        return "real-trained"
    if "paper_full" in text or "synthetic" in text:
        return "synthetic-trained"
    return "selected"


def print_check(a: argparse.Namespace, paths: Mapping[str, Path]) -> None:
    source = determine_real_source(a, paths)
    print(f"repository: {REPO_ROOT}")
    print(f"N: {a.N}")
    print(f"real source: {source}")
    for name, path in paths.items():
        if name == "out":
            print(f"{name:18s} {path}")
        else:
            status = "FOUND" if path.exists() else "MISSING"
            print(f"{name:18s} {status:7s} {path}")
    for name in ("torch", "pyvista", "vtk", "matplotlib", "PIL"):
        print(f"module {name:10s} {'FOUND' if module_available(name) else 'MISSING'}")


def load_state_dict(path: Path, torch, device: str):
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        saved = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # older PyTorch
        saved = torch.load(path, map_location=device)
    if isinstance(saved, dict) and "state_dict" in saved:
        return saved["state_dict"]
    return saved


def choose_device(a: argparse.Namespace, torch) -> str:
    if a.device != "auto":
        return a.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def surface_grid(surface: np.ndarray) -> np.ndarray:
    """Join patch rows without duplicating meridional junctions.

    The periodic display seam is closed by appending the first unique
    circumferential sample.  This is display-only and does not alter metrics.
    """
    S = np.asarray(surface)
    if S.ndim != 4 or S.shape[-1] != 3:
        raise ValueError(f"expected (patch,n_u,m,3), got {S.shape}")
    rows = [S[0]] + [patch[1:] for patch in S[1:]]
    grid = np.concatenate(rows, axis=0)
    return np.concatenate([grid, grid[:, :1, :]], axis=1)


def contour_points(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    pts = []
    for contour, z in zip(np.asarray(R), np.asarray(Z)):
        pts.append(np.column_stack([contour, np.full(len(contour), z)]))
    return np.concatenate(pts, axis=0)


def camera_frame(points: np.ndarray, elev: float, azim: float):
    P = np.asarray(points).reshape(-1, 3)
    lo, hi = np.nanmin(P, axis=0), np.nanmax(P, axis=0)
    center = 0.5 * (lo + hi)
    radius = max(float(np.max(hi - lo) / 2.0), 1e-6)
    el = math.radians(elev)
    az = math.radians(azim)
    direction = np.array([
        math.cos(el) * math.cos(az),
        math.cos(el) * math.sin(az),
        math.sin(el),
    ])
    position = center + 4.5 * radius * direction
    return position, center, radius


def configure_pyvista() -> object:
    import pyvista as pv

    pv.OFF_SCREEN = True
    # Linux servers with an installed Xvfb often need this; EGL/OSMesa builds
    # render without it.  Failure is harmless because screenshot() will still
    # use whichever off-screen backend VTK provides.
    if not os.environ.get("DISPLAY") and hasattr(pv, "start_xvfb"):
        try:
            pv.start_xvfb(wait=0.1)
        except Exception:
            pass
    return pv


def new_plotter(pv, pixels: int):
    p = pv.Plotter(off_screen=True, window_size=(pixels, pixels))
    p.set_background("white")
    try:
        p.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    return p


def set_camera(plotter, frame) -> None:
    position, center, radius = frame
    plotter.camera_position = [position.tolist(), center.tolist(), [0.0, 0.0, 1.0]]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 1.15 * radius


def screenshot(plotter) -> np.ndarray:
    img = plotter.screenshot(
        return_img=True,
        transparent_background=False,
    )
    plotter.close()
    return np.asarray(img)[..., :3]


def pyvista_surface_panel(
    pv,
    surface: np.ndarray,
    frame,
    pixels: int,
    color: str,
) -> np.ndarray:
    G = surface_grid(surface)
    grid = pv.StructuredGrid(G[:, :, 0], G[:, :, 1], G[:, :, 2])
    mesh = grid.extract_surface().triangulate().clean()

    p = new_plotter(pv, pixels)
    p.add_mesh(
        mesh,
        color=color,
        smooth_shading=True,
        ambient=0.22,
        diffuse=0.72,
        specular=0.25,
        specular_power=24,
        show_edges=False,
    )
    try:
        p.add_silhouette(mesh, color="#263238", line_width=1.2)
    except Exception:
        pass
    set_camera(p, frame)
    return screenshot(p)


def pyvista_contour_panel(
    pv,
    R: np.ndarray,
    Z: np.ndarray,
    frame,
    pixels: int,
    poles: Sequence[np.ndarray] = (),
) -> np.ndarray:
    p = new_plotter(pv, pixels)
    for i, (contour, z) in enumerate(zip(np.asarray(R), np.asarray(Z))):
        loop2 = np.concatenate([contour, contour[:1]], axis=0)
        loop3 = np.column_stack([loop2, np.full(len(loop2), z)])
        poly = pv.lines_from_points(loop3, close=False)
        p.add_mesh(
            poly,
            color=CONTOUR_COLORS[i % len(CONTOUR_COLORS)],
            line_width=3.2,
        )
    for pole in poles:
        q = np.asarray(pole, dtype=float).reshape(1, 3)
        p.add_mesh(
            pv.PolyData(q),
            color="#374151",
            point_size=9,
            render_points_as_spheres=True,
        )
    set_camera(p, frame)
    return screenshot(p)


def pyvista_points_panel(
    pv,
    points: np.ndarray,
    frame,
    pixels: int,
) -> np.ndarray:
    P = np.asarray(points).reshape(-1, 3)
    if len(P) > 16000:
        rng = np.random.default_rng(2026)
        P = P[rng.choice(len(P), 16000, replace=False)]
    p = new_plotter(pv, pixels)
    p.add_mesh(
        pv.PolyData(P),
        color=SURFACE_COLORS["gt"],
        point_size=3.0,
        opacity=0.85,
        render_points_as_spheres=True,
    )
    set_camera(p, frame)
    return screenshot(p)


def build_designer_rows(a: argparse.Namespace, paths: Mapping[str, Path]) -> list[FigureRow]:
    require_modules(("torch", "pyvista", "vtk"), "designer reconstruction")
    import torch

    from nssr.geometry import zero_params
    from nssr.networks import ParamNet, contour_features
    from scripts.reconstruct_designer import (
        load_designer,
        project_to_safe,
        surf,
    )

    pv = configure_pyvista()
    device = choose_device(a, torch)
    dtype = torch.float64

    net = ParamNet(c_bound=a.c_bound).to(device=device, dtype=dtype)
    net.load_state_dict(load_state_dict(paths["designer_ckpt"], torch, device))
    net.eval()

    rows = []
    for name in DESIGNER_NAMES:
        obj, _ = load_designer(
            name,
            a.designer_segment_points,
            device,
            dtype,
        )
        N_obj, m_obj = obj["R"].shape[:2]
        p0 = zero_params(N_obj, m_obj, device=device, dtype=dtype)

        with torch.no_grad():
            S0 = surf(obj, p0, a.n_u)
            params = net(contour_features(
                obj["R"], obj["Z"], obj["RB"], obj["RC"],
                obj["Bh"], obj["Th"],
            ))

        projected, alpha, _, _, _, _, stage = project_to_safe(
            obj,
            params,
            a.n_u,
            surf,
            min_ratio=0.15,
            max_cap_fold=a.max_cap_fold,
            iters=a.projection_iters,
        )
        with torch.no_grad():
            S1 = surf(obj, projected, a.n_u)

        R = to_numpy(obj["R"])
        Z = to_numpy(obj["Z"])
        A0 = to_numpy(S0)
        A1 = to_numpy(S1)
        common = np.concatenate([
            surface_grid(A0).reshape(-1, 3),
            surface_grid(A1).reshape(-1, 3),
            contour_points(R, Z),
        ])
        frame = camera_frame(common, a.elev, a.azim)
        poles = (
            np.r_[to_numpy(obj["RB"]).reshape(2), float(to_numpy(obj["Bh"]))],
            np.r_[to_numpy(obj["RC"]).reshape(2), float(to_numpy(obj["Th"]))],
        )

        note = "projection not activated"
        if stage != "none" or alpha < 1.0 - 1e-10:
            note = f"{stage.replace('_', '-')} projection\nalpha={alpha:.3f}"

        rows.append(FigureRow(
            key=name,
            label=DESIGNER_LABELS[name],
            panels={
                "input": pyvista_contour_panel(
                    pv, R, Z, frame, a.panel_pixels, poles=poles
                ),
                "classical": pyvista_surface_panel(
                    pv, A0, frame, a.panel_pixels,
                    DESIGNER_SURFACE_COLORS[name]["classical"],
                ),
                "nssr": pyvista_surface_panel(
                    pv, A1, frame, a.panel_pixels,
                    DESIGNER_SURFACE_COLORS[name]["nssr"],
                ),
            },
            note=note,
        ))
    return rows


def read_csv_by(path: Path, key: str) -> dict[int, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {int(row[key]): row for row in csv.DictReader(f)}


def load_real_metadata(paths: Mapping[str, Path]) -> dict[int, dict[str, object]]:
    validate = read_csv_by(paths["validate_csv"], "idx")
    projection = read_csv_by(paths["projection_csv"], "index")
    index_map = read_csv_by(paths["index_map_csv"], "index")

    all_indices = sorted(set(validate) | set(projection) | set(index_map))
    metadata: dict[int, dict[str, object]] = {}
    for idx in all_indices:
        v = validate.get(idx, {})
        p = projection.get(idx, {})
        m = index_map.get(idx, {})
        path = str(m.get("path", ""))
        stem = Path(path).stem if path else f"object_{idx:02d}"
        classical = float(v.get("classical_chamfer_l2", "nan"))
        raw = float(v.get("learned_chamfer_l2", "nan"))
        post = float(p.get("post_chamfer_l2", raw))
        improvement = (
            100.0 * (classical - post) / classical
            if np.isfinite(classical) and np.isfinite(post) and classical > 0
            else float("nan")
        )
        metadata[idx] = {
            "idx": idx,
            "stem": stem,
            "classical": classical,
            "raw": raw,
            "post": post,
            "improvement": improvement,
            "stage": str(p.get("projection_stage", "unknown")),
            "alpha": float(p.get("alpha", "nan")),
        }
    return metadata


def normalize_object_token(token: str) -> str:
    stem = Path(str(token)).stem.lower()
    return stem.replace("thingiverse_", "thingi_")


def resolve_real_indices(
    tokens: Sequence[str],
    metadata: Mapping[int, Mapping[str, object]],
) -> list[int]:
    by_stem = {
        normalize_object_token(str(row["stem"])): idx
        for idx, row in metadata.items()
    }
    out = []
    for token in tokens:
        if re.fullmatch(r"\d+", str(token)):
            idx = int(token)
        else:
            key = normalize_object_token(str(token))
            if key not in by_stem:
                available = ", ".join(sorted(by_stem)[:12])
                raise KeyError(
                    f"real object {token!r} not found in index map; "
                    f"available examples include: {available}"
                )
            idx = by_stem[key]
        if idx not in out:
            out.append(idx)
    return out


def real_row_label(meta: Mapping[str, object]) -> str:
    stem = str(meta["stem"])
    ident = stem.replace("thingi_", "")
    return f"Thingi10K {ident}" if ident.isdigit() else stem


def real_row_note(meta: Mapping[str, object]) -> str:
    imp = float(meta.get("improvement", float("nan")))
    stage = str(meta.get("stage", "unknown"))
    alpha = float(meta.get("alpha", float("nan")))
    gain = f"CD reduction {imp:.1f}%" if np.isfinite(imp) else ""
    if stage == "none":
        repair = "raw prediction sampled-safe"
    elif stage == "unknown":
        repair = "projection metadata unavailable"
    else:
        repair = f"{stage.replace('_', '-')} repair"
        if np.isfinite(alpha):
            repair += f", alpha={alpha:.3f}"
    return "\n".join(x for x in (gain, repair) if x)


def build_real_rows_reconstructed(
    a: argparse.Namespace,
    paths: Mapping[str, Path],
    metadata: Mapping[int, Mapping[str, object]],
    indices: Sequence[int],
) -> list[FigureRow]:
    require_modules(("torch", "pyvista", "vtk"), "real-object reconstruction")
    import torch

    from nssr.networks import ParamNet
    from nssr.safety import (
        classical_geometry_and_reference,
        geometry_from_params,
        params_from_net,
        project_staged_to_safe,
    )
    from nssr.train import to_torch

    if not paths["real_pickle"].is_file():
        raise FileNotFoundError(
            f"real test pickle not found: {paths['real_pickle']}"
        )
    with paths["real_pickle"].open("rb") as f:
        samples = pickle.load(f)

    pv = configure_pyvista()
    device = choose_device(a, torch)
    dtype = torch.float32
    net = ParamNet(c_bound=a.c_bound).to(device=device, dtype=dtype)
    net.load_state_dict(load_state_dict(paths["real_ckpt"], torch, device))
    net.eval()

    rows = []
    for order, idx in enumerate(indices):
        if idx >= len(samples):
            raise IndexError(
                f"real index {idx} exceeds test pickle length {len(samples)}"
            )
        sample = samples[idx]
        obj = to_torch(
            sample,
            m=a.m,
            device=device,
            dtype=dtype,
            gt_subsample=20000,
            seed=40000 + idx,
        )

        with torch.no_grad():
            classical, reference = classical_geometry_and_reference(obj, a.n_u)
            raw_params = params_from_net(net, obj)
            projected, alpha, stage, *_ = project_staged_to_safe(
                obj,
                raw_params,
                a.n_u,
                reference,
                max_cap_fold=a.max_cap_fold,
                max_iter=a.projection_iters,
                with_metrics=False,
            )
            learned = geometry_from_params(obj, projected, a.n_u, reference)

        gt = to_numpy(obj["gt_pts"])
        R = to_numpy(obj["R"])
        Z = to_numpy(obj["Z"])
        S0 = to_numpy(classical.surface.xyz)
        S1 = to_numpy(learned.surface.xyz)
        common = np.concatenate([
            gt,
            contour_points(R, Z),
            surface_grid(S0).reshape(-1, 3),
            surface_grid(S1).reshape(-1, 3),
        ])
        frame = camera_frame(common, a.elev, a.azim)

        meta = dict(metadata.get(idx, {}))
        meta.setdefault("idx", idx)
        meta.setdefault("stem", Path(str(sample.get("path", f"object_{idx:02d}"))).stem)
        meta["stage"] = stage
        meta["alpha"] = alpha

        rows.append(FigureRow(
            key=str(meta["stem"]),
            label=real_row_label(meta),
            panels={
                "gt": pyvista_points_panel(pv, gt, frame, a.panel_pixels),
                "input": pyvista_contour_panel(pv, R, Z, frame, a.panel_pixels),
                "classical": pyvista_surface_panel(
                    pv, S0, frame, a.panel_pixels, SURFACE_COLORS["classical"]
                ),
                "nssr": pyvista_surface_panel(
                    pv, S1, frame, a.panel_pixels, SURFACE_COLORS["nssr"]
                ),
            },
            note=real_row_note(meta),
        ))
    return rows


def trim_white(image: np.ndarray, pad: int = 16) -> np.ndarray:
    A = np.asarray(image)[..., :3]
    mask = np.any(A < 248, axis=2)
    if not np.any(mask):
        return A
    ys, xs = np.where(mask)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, A.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, A.shape[1])
    return A[y0:y1, x0:x1]


def square_on_white(image: np.ndarray, pixels: int) -> np.ndarray:
    from PIL import Image

    A = np.asarray(image)[..., :3].astype(np.uint8)
    im = Image.fromarray(A)
    im.thumbnail((pixels, pixels), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (pixels, pixels), "white")
    canvas.paste(im, ((pixels - im.width) // 2, (pixels - im.height) // 2))
    return np.asarray(canvas)


def split_existing_four_panel(path: Path, pixels: int) -> dict[str, np.ndarray]:
    """Recover the four visual panels from visualize_real.py output.

    The top title band is removed before white-space trimming so the new
    composite has one consistent typography and no duplicated old labels.
    """
    from PIL import Image

    A = np.asarray(Image.open(path).convert("RGB"))
    h, w = A.shape[:2]
    y0, y1 = int(round(0.09 * h)), int(round(0.94 * h))
    keys = ("gt", "input", "classical", "nssr")
    panels = {}
    for k, key in enumerate(keys):
        x0 = int(round(k * w / 4))
        x1 = int(round((k + 1) * w / 4))
        crop = trim_white(A[y0:y1, x0:x1])
        panels[key] = square_on_white(crop, pixels)
    return panels


def locate_existing_real_png(
    directory: Path,
    idx: int,
    stem: str,
) -> Path:
    candidates = [
        directory / f"real_{idx:02d}_{stem}.png",
        *sorted(directory.glob(f"real_{idx:02d}_*.png")),
        *sorted(directory.glob(f"real_*_{stem}.png")),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"existing four-panel PNG not found for index {idx} / {stem} in {directory}"
    )


def build_real_rows_existing(
    a: argparse.Namespace,
    paths: Mapping[str, Path],
    metadata: Mapping[int, Mapping[str, object]],
    indices: Sequence[int],
) -> list[FigureRow]:
    require_modules(("PIL",), "existing real-image fallback")
    rows = []
    for idx in indices:
        meta = dict(metadata.get(idx, {}))
        meta.setdefault("idx", idx)
        meta.setdefault("stem", f"object_{idx:02d}")
        path = locate_existing_real_png(
            paths["existing_real_dir"], idx, str(meta["stem"])
        )
        rows.append(FigureRow(
            key=str(meta["stem"]),
            label=real_row_label(meta),
            panels=split_existing_four_panel(path, a.panel_pixels),
            note=real_row_note(meta),
        ))
    return rows


def panel_letter(index: int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    if index < len(letters):
        return f"({letters[index]})"
    return f"({index + 1})"


def save_figure(fig, stem: Path, formats: Sequence[str], dpi: int) -> list[Path]:
    outputs = []
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        kwargs = dict(bbox_inches="tight", pad_inches=0.04)
        if fmt in {"png", "tif"}:
            kwargs["dpi"] = dpi
        if fmt == "tif":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(path)
        print(f"wrote {path}")
    return outputs


def make_comparison_figure(
    rows: Sequence[FigureRow],
    columns: Sequence[str],
    title: str,
    stem: Path,
    formats: Sequence[str],
    dpi: int,
    panel_letters: bool,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows, ncols = len(rows), len(columns)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.75 * ncols + 1.1, 2.48 * nrows + 0.65),
        squeeze=False,
    )
    letter = 0
    for i, row in enumerate(rows):
        for j, key in enumerate(columns):
            ax = axes[i, j]
            ax.imshow(row.panels[key])
            ax.set_axis_off()
            if i == 0:
                ax.set_title(PANEL_TITLES[key], fontsize=10.5, pad=6)
            if panel_letters:
                ax.text(
                    0.015, 0.975, panel_letter(letter),
                    transform=ax.transAxes,
                    ha="left", va="top", fontsize=9.5,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82),
                )
            letter += 1

        axes[i, 0].text(
            -0.08,
            0.5,
            row.label,
            transform=axes[i, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10.2,
            fontweight="semibold",
        )
        if row.note:
            axes[i, -1].text(
                0.99,
                0.015,
                row.note,
                transform=axes[i, -1].transAxes,
                ha="right",
                va="bottom",
                fontsize=7.8,
                color="#374151",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.78),
            )

    fig.suptitle(title, y=0.995, fontsize=13, fontweight="semibold")
    fig.subplots_adjust(left=0.08, right=0.995, top=0.94, bottom=0.015,
                        wspace=0.035, hspace=0.08)
    outputs = save_figure(fig, stem, formats, dpi)
    plt.close(fig)
    return outputs


def make_combined_gallery(
    designer_rows: Sequence[FigureRow],
    real_rows: Sequence[FigureRow],
    N: int,
    stem: Path,
    formats: Sequence[str],
    dpi: int,
    panel_letters: bool,
    designer_domain: str,
    real_domain: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.0, 6.0))
    outer = fig.add_gridspec(2, 1, height_ratios=(1.0, 1.08), hspace=0.20)
    top = outer[0].subgridspec(1, len(designer_rows), wspace=0.035)
    bottom = outer[1].subgridspec(1, len(real_rows), wspace=0.035)

    letter = 0
    for j, row in enumerate(designer_rows):
        ax = fig.add_subplot(top[0, j])
        ax.imshow(row.panels["nssr"])
        ax.set_axis_off()
        ax.set_title(row.label, fontsize=10.5, pad=4)
        if panel_letters:
            ax.text(0.02, 0.97, panel_letter(letter), transform=ax.transAxes,
                    ha="left", va="top", fontsize=9.5,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82))
        letter += 1

    for j, row in enumerate(real_rows):
        ax = fig.add_subplot(bottom[0, j])
        ax.imshow(row.panels["nssr"])
        ax.set_axis_off()
        imp = ""
        match = re.search(r"reduction ([+-]?[0-9.]+)%", row.note)
        if match:
            imp = f"; CD down {match.group(1)}%"
        ax.set_title(row.label + imp, fontsize=9.5, pad=4)
        if panel_letters:
            ax.text(0.02, 0.97, panel_letter(letter), transform=ax.transAxes,
                    ha="left", va="top", fontsize=9.5,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82))
        letter += 1

    fig.text(0.015, 0.735, f"Designer shapes\n({designer_domain} checkpoint)",
             rotation=90, ha="center", va="center", fontsize=9.5,
             fontweight="semibold")
    fig.text(0.015, 0.265, f"Held-out real meshes\n({real_domain} checkpoint)",
             rotation=90, ha="center", va="center", fontsize=9.5,
             fontweight="semibold")
    fig.suptitle(
        f"NSSR qualitative operating point (N={N}, safety-first checkpoints)",
        y=0.995,
        fontsize=13,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.015)
    outputs = save_figure(fig, stem, formats, dpi)
    plt.close(fig)
    return outputs


def write_latex_snippet(
    out: Path,
    N: int,
    designer_domain: str,
    real_domain: str,
) -> Path:
    path = out / f"qualitative_figures_N{N}.tex"
    text = rf"""% Generated inclusion template for the NSSR qualitative figures.
% Adjust the relative graphics path to match the manuscript directory.

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figure_designer_comparison_N{N}.pdf}}
\caption{{Designer-shape reconstructions using the {designer_domain},
safety-first $N={N}$ checkpoint.  Each row shows the ordered input contours,
the zero-correction classical Hermite surface, and the final NSSR surface
after the shared sampled-safety projection plus the designer-specific axis
clearance check.  The computational surfaces use $m=128$ unique periodic
circumferential samples; seam duplication is used for display only.}}
\label{{fig:nssr_designer_qualitative}}
\end{{figure*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figure_real_comparison_N{N}.pdf}}
\caption{{Representative held-out real-mesh reconstructions for the
{real_domain}, safety-first $N={N}$ checkpoint.  Columns show ground-truth
samples, the sparse input contour stack, the classical Hermite surface, and
the final post-projection NSSR surface.  The examples were selected to cover
distinct silhouettes and accuracy gains and include a case requiring
tangent-stage repair; selection was not used for any aggregate metric.}}
\label{{fig:nssr_real_qualitative}}
\end{{figure*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figure_qualitative_combined_N{N}.pdf}}
\caption{{Compact qualitative gallery at the strongest slice-count operating
point, $N={N}$.  The upper row uses the {designer_domain} checkpoint on the
three hand-designed contour stacks; the lower row uses the {real_domain}
checkpoint on held-out real meshes.  Every displayed NSSR output is the final
state returned by the sampled-safety projection.}}
\label{{fig:nssr_qualitative_gallery}}
\end{{figure*}}
"""
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")
    return path


def validate_cli(a: argparse.Namespace) -> None:
    if a.N < 3:
        raise SystemExit("--N must be at least 3")
    if a.m < 8:
        raise SystemExit("--m must be at least 8")
    if a.n_u < 4:
        raise SystemExit("--n-u must be at least 4")
    if a.designer_segment_points < 2:
        raise SystemExit("--designer-segment-points must be at least 2")
    if a.panel_pixels < 256:
        raise SystemExit("--panel-pixels must be at least 256")
    if a.projection_iters < 1:
        raise SystemExit("--projection-iters must be positive")
    if a.max_cap_fold <= 0:
        raise SystemExit("--max-cap-fold must be positive")
    if "combined" in a.make and not {"designer", "real"}.issubset(set(a.make)):
        raise SystemExit("--make combined also requires designer and real")


def main() -> None:
    a = parse_args()
    validate_cli(a)
    paths = resolved_paths(a)
    if a.check:
        print_check(a, paths)
        return

    require_modules(("matplotlib",), "figure composition")
    paths["out"].mkdir(parents=True, exist_ok=True)

    designer_rows: list[FigureRow] = []
    real_rows: list[FigureRow] = []
    designer_domain = checkpoint_domain_label(paths["designer_ckpt"])
    real_domain = checkpoint_domain_label(paths["real_ckpt"])

    if "designer" in a.make or "combined" in a.make:
        designer_rows = build_designer_rows(a, paths)

    if "real" in a.make or "combined" in a.make:
        metadata = load_real_metadata(paths)
        if not metadata:
            raise SystemExit(
                "Real metrics/index files were not found; cannot resolve the "
                "requested curated objects."
            )
        indices = resolve_real_indices(a.real_objects, metadata)
        source = determine_real_source(a, paths)
        print(f"real panel source: {source}")
        if source == "reconstruct":
            real_rows = build_real_rows_reconstructed(
                a, paths, metadata, indices
            )
        else:
            if a.real_ckpt:
                print(
                    "WARNING: --real-source existing uses the repository's "
                    "pre-rendered real-only figures; --real-ckpt is ignored."
                )
            real_domain = "real-trained"
            real_rows = build_real_rows_existing(
                a, paths, metadata, indices
            )

    panel_letters = not a.no_panel_letters
    out = paths["out"]
    if "designer" in a.make:
        make_comparison_figure(
            designer_rows,
            ("input", "classical", "nssr"),
            f"Designer-shape reconstruction with the N={a.N} safety-first checkpoint",
            out / f"figure_designer_comparison_N{a.N}",
            a.formats,
            a.dpi,
            panel_letters,
        )
    if "real" in a.make:
        make_comparison_figure(
            real_rows,
            ("gt", "input", "classical", "nssr"),
            f"Held-out real-mesh reconstruction (N={a.N})",
            out / f"figure_real_comparison_N{a.N}",
            a.formats,
            a.dpi,
            panel_letters,
        )
    if "combined" in a.make:
        make_combined_gallery(
            designer_rows,
            real_rows,
            a.N,
            out / f"figure_qualitative_combined_N{a.N}",
            a.formats,
            a.dpi,
            panel_letters,
            designer_domain,
            real_domain,
        )
    write_latex_snippet(out, a.N, designer_domain, real_domain)


if __name__ == "__main__":
    main()
