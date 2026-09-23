"""Plot the README's General / Sokoban accuracy–latency comparison.

General values come from the saved results snapshot. Sokoban values are the
published single-step results in docs/technical.md (500 questions, 100 levels).
Run from any directory with Matplotlib installed; no model weights are needed.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "assets/figures/evaluation-overview"
COLORS = {"baseline": "#8693A5", "sft100k": "#456DD4", "rlcd30k": "#008C84"}
MARKERS = {"0.8B": "o", "2B": "D"}
# size, method, accuracy (%), mean end-to-end latency (ms)
SOKOBAN = [
    ("0.8B", "baseline", 31.60, 152.54),
    ("0.8B", "sft100k", 80.60, 114.06),
    ("0.8B", "rlcd30k", 83.00, 113.95),
    ("2B", "baseline", 27.20, 155.73),
    ("2B", "sft100k", 83.60, 126.82),
    ("2B", "rlcd30k", 87.60, 127.31),
]


def label(size, method):
    if method == "baseline":
        return f"Qwen3.5-{size}"
    return f"Valen-{size}({'SFT' if method == 'sft100k' else 'RL'})"


def main():
    snapshot = json.loads((ROOT / "assets/figures/general/results.json").read_text())
    general = [(r["size"], r["method"], r["accuracy_pct"], r["mean_latency_ms"])
               for r in snapshot["models"]]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": "#233349",
        "axes.labelcolor": "#52647C", "xtick.color": "#64748B",
        "ytick.color": "#64748B", "svg.fonttype": "path",
        "svg.hashsalt": "valen-evaluation-overview", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.subplots_adjust(left=.065, right=.98, bottom=.22, top=.83, wspace=.20)
    offsets = [
        [(-12, -22), (10, -34), (8, 27), (-12, 20), (12, 24), (14, -30)],
        [(-12, 22), (8, -32), (-3, 28), (-12, -17), (18, -22), (12, 30)],
    ]
    panels = [
        ("General", "5,000 questions", general, (140, 258), (70, 82)),
        ("Sokoban", "500 single-step questions · 100 levels", SOKOBAN, (102, 172), (18, 104)),
    ]
    for i, (ax, (title, subtitle, rows, xlim, ylim)) in enumerate(zip(axes, panels)):
        ax.set_facecolor("#F7F9FC")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_axisbelow(True)
        ax.grid(color="#E2E8F0", linewidth=.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, pad=9)
        ax.set_title(title, loc="left", fontsize=19, fontweight="bold", pad=34)
        ax.text(0, 1.055, subtitle, transform=ax.transAxes, fontsize=10, color="#64748B")
        ax.set_xlabel("Mean end-to-end latency (ms) ↓", labelpad=12)
        ax.set_ylabel("Accuracy (%) ↑", labelpad=10)
        ax.set_xticks([150, 175, 200, 225, 250] if i == 0 else [110, 125, 140, 155, 170])
        ax.set_yticks([70, 72, 74, 76, 78, 80, 82] if i == 0 else [20, 40, 60, 80, 100])
        for row, offset in zip(rows, offsets[i]):
            size, method, score, latency = row
            color = COLORS[method]
            ax.scatter(latency, score, s=115, marker=MARKERS[size], color=color,
                       edgecolors="white", linewidths=1.5, zorder=3)
            ax.annotate(label(size, method), xy=(latency, score), xytext=offset,
                        textcoords="offset points", color=color, fontsize=10,
                        fontweight="semibold", ha="right" if offset[0] < -5 else "left",
                        va="center", arrowprops={"arrowstyle": "-", "color": color,
                                                  "lw": .7, "alpha": .5}, zorder=4)
    handles = [Line2D([], [], color=c, marker="o", linestyle="", markersize=7, label=n)
               for n, c in [("Qwen baseline", COLORS["baseline"]),
                            ("Valen · SFT", COLORS["sft100k"]),
                            ("Valen · RL", COLORS["rlcd30k"])]]
    handles += [Line2D([], [], color="#52647C", marker=m, linestyle="", markersize=7, label=s)
                for s, m in MARKERS.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .02),
               ncol=5, frameon=False, fontsize=10, columnspacing=2.4)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
    svg = OUTPUT.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(OUTPUT.with_suffix(".pdf"), facecolor="white",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"Wrote {OUTPUT}.png / .svg / .pdf")


if __name__ == "__main__":
    main()
