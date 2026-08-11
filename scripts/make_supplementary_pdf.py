"""Build NSSR supplementary-material PDF from experiment outputs.

The script is intentionally result-driven: run the synthetic and real/mixed
experiments first, create figures with visualize_real.py (and any synthetic
figure directory), then assemble the PDF.

It includes:
  - experiment summary tables,
  - real-object reconstruction panels,
  - base/crown close-ups,
  - synthetic/designer/example panels if supplied,
  - appendix notes describing seam closure and sampled safety.

Only display/export surfaces repeat the first circumferential point to close
the visual seam; computational geometry remains unchanged.
"""
from __future__ import annotations
import argparse, csv, glob, os
from pathlib import Path


def read_rows(path):
    if not path or not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def short(v, digits=4):
    try:
        x = float(v)
    except Exception:
        return str(v)
    if abs(x) <= 1 and ("rate" in str(v).lower()):
        return f"{100*x:.1f}%"
    return f"{x:.{digits}g}"


def image_paths(roots):
    out = []
    for root in roots:
        if not root:
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            out.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--out", default="results/NSSR_supplementary.pdf")
    ap.add_argument("--title", default="NSSR Supplementary Material")
    ap.add_argument("--synthetic_summary", default="results/paper_full_100ep/summary.csv")
    ap.add_argument("--domain_summary", default="results/paper_real/domain_comparison.csv")
    ap.add_argument("--real_figs", nargs="*", default=["results/paper_real/figures"])
    ap.add_argument("--synthetic_figs", nargs="*", default=[])
    ap.add_argument("--designer_figs", nargs="*", default=[])
    ap.add_argument("--max_figures", type=int, default=40)
    a = ap.parse_args()

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, KeepTogether
    )
    from PIL import Image as PILImage

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CenterTitle", parent=styles["Title"], alignment=TA_CENTER
    ))

    doc = SimpleDocTemplate(
        a.out, pagesize=landscape(A4),
        rightMargin=12*mm, leftMargin=12*mm,
        topMargin=12*mm, bottomMargin=12*mm,
    )
    story = [
        Paragraph(a.title, styles["CenterTitle"]),
        Spacer(1, 4*mm),
        Paragraph(
            "This supplement reports synthetic, real-mesh, cross-domain and "
            "failure-aware projection results. Safety is sampled and is defined "
            "as Jacobian-valid plus cap turn-back at or below the stated threshold. "
            "The repeated first circumferential point used in figures/mesh display "
            "only closes the visual seam; it is not inserted into training, metrics "
            "or Jacobian evaluation.",
            styles["BodyText"],
        ),
        Spacer(1, 5*mm),
    ]

    syn = read_rows(a.synthetic_summary)
    if syn:
        story += [Paragraph("A. Synthetic full sweep", styles["Heading1"])]
        cols = [
            "N", "raw_j_valid_rate", "raw_cap_safe_rate", "raw_safe_rate",
            "post_safe_rate", "projection_activation_rate",
            "raw_chamfer_l2_projection_eval", "post_chamfer_l2",
            "chamfer_l2_projection_delta_pct",
        ]
        hdr = ["N","Raw J","Raw cap","Raw SAFE","Post SAFE","Proj.","Raw CD","Post CD","CD delta %"]
        data = [hdr]
        for r in syn:
            vals = []
            for c in cols:
                x = r.get(c, "")
                if "rate" in c:
                    try: x = f"{100*float(x):.1f}%"
                    except Exception: pass
                elif c.endswith("_pct"):
                    try: x = f"{float(x):+.2f}%"
                    except Exception: pass
                else:
                    try: x = f"{float(x):.6f}" if "chamfer" in c else x
                    except Exception: pass
                vals.append(x)
            data.append(vals)
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
            ("GRID",(0,0),(-1,-1),0.35,colors.grey),
            ("FONTSIZE",(0,0),(-1,-1),7.5),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ]))
        story += [t, Spacer(1,5*mm)]

    dom = read_rows(a.domain_summary)
    if dom:
        story += [Paragraph("B. Real and cross-domain evaluation", styles["Heading1"])]
        cols = [
            "train_domain","test_domain","N","learned_chamfer_l2",
            "raw_j_valid_rate","raw_cap_safe_rate","raw_safe_rate",
            "post_safe_rate","projection_activation_rate","post_chamfer_l2",
        ]
        hdr = ["Train","Test","N","Raw CD","Raw J","Raw cap","Raw SAFE","Post SAFE","Proj.","Post CD"]
        data = [hdr]
        for r in dom:
            vals = []
            for c in cols:
                x = r.get(c,"")
                if "rate" in c:
                    try: x = f"{100*float(x):.1f}%"
                    except Exception: pass
                elif "chamfer" in c:
                    try: x = f"{float(x):.6f}"
                    except Exception: pass
                vals.append(x)
            data.append(vals)
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
            ("GRID",(0,0),(-1,-1),0.35,colors.grey),
            ("FONTSIZE",(0,0),(-1,-1),7.5),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ]))
        story += [t, Spacer(1,5*mm)]

    groups = [
        ("C. Real-object reconstructions", a.real_figs),
        ("D. Synthetic / controlled examples", a.synthetic_figs),
        ("E. Designer examples", a.designer_figs),
    ]
    count = 0
    for heading, roots in groups:
        imgs = image_paths(roots)
        if not imgs:
            continue
        story += [PageBreak(), Paragraph(heading, styles["Heading1"])]
        for p in imgs:
            if count >= a.max_figures:
                break
            with PILImage.open(p) as im:
                w, h = im.size
            maxw, maxh = 250*mm, 150*mm
            scale = min(maxw/w, maxh/h)
            ri = Image(p, width=w*scale, height=h*scale)
            cap = Paragraph(
                os.path.relpath(p).replace("_","\\_"),
                styles["Caption"] if "Caption" in styles else styles["BodyText"]
            )
            story += [KeepTogether([ri, Spacer(1,2*mm),cap]), Spacer(1,5*mm)]
            count += 1

    story += [
        PageBreak(),
        Paragraph("F. Base/crown and mesh-display notes", styles["Heading1"]),
        Paragraph(
            "Base and crown close-ups should be interpreted together with the "
            "cap turn-back statistic. Exact cap poles are intentionally excluded "
            "from Jacobian degeneracy checks. For visual surface rendering, the "
            "first circumferential sample is appended after the last sample so the "
            "periodic seam is visibly closed. Face connectivity/export should wrap "
            "last-to-first in the same way. This display-only duplication is not "
            "used by the reconstruction loss or safety metrics.",
            styles["BodyText"],
        ),
        Spacer(1,4*mm),
        Paragraph(
            "The projection results are sampled guarantees under the stated "
            "Jacobian/cap criteria; they should not be described as a global "
            "analytic proof against every possible between-sample self-intersection.",
            styles["BodyText"],
        ),
    ]

    doc.build(story)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
