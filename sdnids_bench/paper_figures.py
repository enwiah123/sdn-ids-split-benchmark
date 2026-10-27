"""Figures that appear in the manuscript in a combined form: the protocol diagram and two-panel figures built
from the per-dataset figures written by figures.py."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import FancyBboxPatch         # noqa: E402
from PIL import Image, ImageDraw, ImageFont           # noqa: E402

from .figures import SERIES, INK, save                # noqa: E402


def fig_protocols(out_dir, name="fig_protocols"):
    """The three evaluation protocols and the comparisons between them."""
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")

    def box(x, y, w, h, text, edge, face, bold=False, size=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", lw=1.6,
                                    edgecolor=edge, facecolor=face))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, style="-|>", color="#3d3d3d"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.6, mutation_scale=14))

    box(36, 49, 27, 5, "Flow records (raw)", "#666666", "#f2f2f2", bold=True, size=12)
    box(54.5, 37.5, 39, 5.5, "Deduplication\n(exact duplicates, conflicting labels)", "#666666", "#f2f2f2")
    box(6.5, 24, 31, 8.5, "S-R raw\nrandom 70/15/15 split,\nduplicates kept", SERIES[1], "#fdf0e8")
    box(40, 24, 25.5, 8.5, "S-R dedup\nrandom 70/15/15 split", SERIES[2], "#e6f6ef")
    box(69, 24, 27.5, 8.5, "S-C\nper class: order by time,\n70/15/15 + embargo L", SERIES[0], "#e8f0fb")
    arrow(49.5, 48.5, 26.5, 32.8)
    arrow(56, 48.5, 70.5, 43.3)
    arrow(70.5, 37.2, 53.5, 32.8)
    arrow(79, 37.2, 82, 32.8)
    arrow(22.5, 21.5, 39.5, 21.5, "<|-|>")
    ax.text(31, 17.5, "effect of\nduplicates", ha="center", va="center", fontsize=10.5, color=INK)
    arrow(53.5, 21.5, 82.5, 21.5, "<|-|>")
    ax.text(68, 17.5, "effect of mixing early and\nlate records of each class", ha="center", va="center", fontsize=10.5,
            color=INK)
    arrow(22.5, 9.5, 82.5, 9.5, "<|-|>")
    ax.text(52.5, 7, "total inflation (H1)", ha="center", va="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.plot([82.8, 82.8], [21, 6], ls=":", color=SERIES[0], lw=1.6)
    ax.text(82.8, 3, "main benchmark:\nH2, H3, H4, B6", ha="center", va="center", fontsize=10.5, color=SERIES[0],
            fontweight="bold")
    fig.tight_layout()
    return save(fig, out_dir, name)


def stack_panels(paths, labels, out_path, pad=40, label_size=90):
    """Stack PNG figures vertically and add panel labels such as '(a) InSDN'."""
    ims = [Image.open(p).convert("RGB") for p in paths]
    w = max(i.width for i in ims)
    h = sum(i.height + label_size + pad for i in ims)
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(label_size * 0.8))
    except OSError:
        font = ImageFont.load_default()
    y = 0
    for im, lab in zip(ims, labels):
        draw.text((20, y + 5), lab, fill=INK, font=font)
        canvas.paste(im, (0, y + label_size))
        y += im.height + label_size + pad
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path, dpi=(600, 600))
    return out_path
