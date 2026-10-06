"""Draw three Valen accuracy charts from the reviewed October 6 snapshot.

Style follows plot_readme_overview.py: white canvas, gray baselines, teal Valen
bars with rounded tops, exact value labels, and the existing Valen eye mark.
Run from any directory; outputs are PNG, SVG and PDF beside the snapshot.
"""
import json
from itertools import groupby
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as PlotPath

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "assets/figures/evaluation-20261006"
TEAL, GRAY = "#009C83", "#E6E6E6"
INK, MUTED, DARK_TEAL = "#111111", "#6D6D6D", "#006C5B"


def rounded_bar(ax, x, height, color):
    # Bars extend below the visible axis range; only the top corners are rounded.
    half, rx, ry = .34, .065, .65
    left, right = x - half, x + half
    vertices = [(left, 0), (left, height - ry), (left, height),
                (left + rx, height), (right - rx, height),
                (right, height), (right, height - ry), (right, 0), (left, 0)]
    codes = [PlotPath.MOVETO, PlotPath.LINETO, PlotPath.CURVE3, PlotPath.CURVE3,
             PlotPath.LINETO, PlotPath.CURVE3, PlotPath.CURVE3, PlotPath.LINETO,
             PlotPath.CLOSEPOLY]
    ax.add_patch(PathPatch(PlotPath(vertices, codes), facecolor=color, edgecolor="none", zorder=3))


def draw(snapshot, benchmark, icon):
    info = snapshot["benchmarks"][benchmark]
    results = {row["model_id"]: row for row in snapshot["results"] if row["benchmark"] == benchmark}
    models = [m for m in snapshot["models"]
              if m["id"] in results and m["id"] != "rsi-v5-3b"]
    # Valen first within each size, then baselines by ascending benchmark accuracy.
    # Unspecified size remains a separate final group.
    models.sort(key=lambda m: (
        float("inf") if m["size_b"] is None else m["size_b"],
        not m["highlight"],
        results[m["id"]]["accuracy_pct"],
    ))
    positioned, groups, x = [], [], 0.
    for size, members in groupby(models, key=lambda m: m["size_b"]):
        start = x
        for model in members:
            positioned.append((x, model))
            x += 1
        groups.append((start, x - 1, size))
        x += .6

    fig, ax = plt.subplots(figsize=(15.8, 5.8), facecolor="white")
    fig.subplots_adjust(left=.065, right=.985, bottom=.29, top=.77)
    fig.text(.065, .94, "VALEN", fontsize=12, weight="bold", color=DARK_TEAL)
    fig.text(.525, .935, info["title"], ha="center", fontsize=24, weight="bold")
    fig.text(.525, .877, info["subtitle"], ha="center", fontsize=12, color=MUTED)
    fig.legend(handles=[Patch(facecolor=TEAL, label="Valen · SFT"),
                        Patch(facecolor=GRAY, label="Published baselines")],
               loc="upper right", bbox_to_anchor=(.985, .97), frameon=False, fontsize=10)

    ax.set_xlim(-.65, positioned[-1][0] + .65)
    ymin, ymax = {"eval_v1": (60, 85), "video": (60, 90), "jevbench": (55, 90)}[benchmark]
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Accuracy (%) ↑", fontsize=12, color=MUTED, labelpad=10)
    ax.set_yticks(range(ymin, ymax + 1, 5))
    ax.tick_params(axis="y", length=0, labelsize=10, colors=MUTED, pad=6)
    ax.grid(axis="y", color="#F0F0F0", linewidth=.8, zorder=0)
    for name, spine in ax.spines.items():
        spine.set_visible(name == "bottom")
        spine.set_color("#DCDCDC")
        spine.set_linewidth(.8)
    ax.set_xticks([p for p, _ in positioned], [m["chart_label"] for _, m in positioned])
    ax.tick_params(axis="x", length=0, pad=10, labelsize=9.5, colors=MUTED)

    for (position, model), tick in zip(positioned, ax.get_xticklabels()):
        score = results[model["id"]]["accuracy_pct"]
        if not 0 <= score <= 100:
            raise ValueError(f"Invalid accuracy for {model['id']}: {score}")
        highlighted = model["highlight"]
        rounded_bar(ax, position, score, TEAL if highlighted else GRAY)
        label_above = highlighted or (score - ymin) / (ymax - ymin) < .11
        ax.annotate(f"{score:.2f}", (position, score),
                    xytext=(0, 5 if label_above else -7), textcoords="offset points",
                    ha="center", va="bottom" if label_above else "top", fontsize=10.5,
                    weight="bold" if highlighted else "normal",
                    color=DARK_TEAL if highlighted else MUTED)
        if highlighted:
            tick.set_color(DARK_TEAL)
            tick.set_weight("bold")
            ax.add_artist(AnnotationBbox(
                OffsetImage(icon, zoom=15 / icon.shape[0], interpolation="lanczos"),
                (position, score), xybox=(0, 23), xycoords="data", boxcoords="offset points",
                box_alignment=(.5, 0), frameon=False, pad=0, annotation_clip=False))

    transform = ax.get_xaxis_transform()
    for start, end, size in groups:
        ax.plot([start - .38, end + .38], [-.22, -.22], transform=transform,
                color="#DCDCDC", linewidth=1, clip_on=False)
        label = f"{size:g}B" if size is not None else "Size n/a"
        ax.text((start + end) / 2, -.275, label, transform=transform,
                ha="center", va="top", fontsize=13, weight="bold", color=INK)
    fig.text(.525, .075, "Model size (reported parameter scale)",
             ha="center", fontsize=10, color=MUTED)
    mode = "One task per request" if benchmark == "jevbench" else "Valen: shared state · Baselines: serial"
    fig.text(.065, .02, mode + "  |  Overall accuracy, weighted by question count",
             fontsize=9, color=MUTED)
    fig.text(.985, .02, "2026-10-06", ha="right", fontsize=9, color=MUTED)
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 180} if suffix == "png" else {
            "metadata": {"Date": None} if suffix == "svg" else {"CreationDate": None, "ModDate": None}}
        path = OUTPUT / f"{benchmark}.{suffix}"
        fig.savefig(path, facecolor="white", **kwargs)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(f"{benchmark}: {len(models)} models → PNG / SVG / PDF")


def main():
    snapshot = json.loads((OUTPUT / "results.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": INK, "svg.fonttype": "path", "pdf.fonttype": 42,
                         "svg.hashsalt": "valen-2.0-results"})
    icon = plt.imread(ROOT / "assets/branding/valen-logo-v5.png")[120:562, 50:714]
    for benchmark in ("eval_v1", "video", "jevbench"):
        draw(snapshot, benchmark, icon)


if __name__ == "__main__":
    main()
