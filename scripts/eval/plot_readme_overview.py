"""Draw the README's separate General and Sokoban comparison charts.

Each figure compares six models using accuracy and mean end-to-end latency.
General values come from the saved results snapshot; Sokoban values are the
published single-step results in docs/technical.md (500 questions, 100 levels).
Style reference: https://huggingface.co/internlm/Atria-Dawn-Preview
Run with Matplotlib installed; no model weights are needed.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
COLORS = {"baseline": "#E1E3E6", "sft100k": "#83AEF5", "rlcd30k": "#3075F4"}
INK, MUTED, GRID = "#171B22", "#59616E", "#EBEDF0"
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


def draw_comparison(title, subtitle, rows, output, accuracy_range=(0, 100), latency_max=280):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), facecolor="white")
    fig.subplots_adjust(left=.055, right=.985, bottom=.21, top=.70, wspace=.16)
    fig.text(.055, .94, title, fontsize=23, weight="bold", va="top")
    fig.text(.055, .86, subtitle, fontsize=11, color=MUTED, va="top")
    handles = [Patch(facecolor=COLORS[key], label=name) for key, name in (
        ("baseline", "Qwen baseline"), ("sft100k", "Valen · SFT"),
        ("rlcd30k", "Valen · RL"))]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.989, .96),
               frameon=False, ncol=3, handlelength=1, handleheight=1,
               columnspacing=1.7, fontsize=10)
    positions = [0, 1, 2, 3.5, 4.5, 5.5]
    labels = [label(size, method).replace("(", "\n(") if method != "baseline"
              else f"Qwen3.5\n{size}" for size, method, _, _ in rows]
    accuracy_lower, accuracy_upper = accuracy_range
    accuracy_ticks = range(accuracy_lower, accuracy_upper + 1,
                           2 if accuracy_lower else 20)
    panels = [
        ("Accuracy (%) ↑", 2, accuracy_lower, accuracy_upper, accuracy_ticks),
        ("Mean end-to-end latency (ms) ↓", 3, 0, latency_max, range(0, latency_max + 1, 50)),
    ]
    for ax, (metric, index, lower, upper, ticks) in zip(axes, panels):
        values = [row[index] for row in rows]
        best = max(values) if index == 2 else min(values)
        ax.set_title(metric, loc="left", fontsize=14, weight="bold", pad=15)
        bars = ax.bar(positions, [value - lower for value in values], bottom=lower, width=.70,
                      color=[COLORS[row[1]] for row in rows], zorder=3)
        ax.set_ylim(lower, upper)
        ax.set_xlim(-.65, 6.15)
        ax.set_yticks(list(ticks))
        ax.set_xticks(positions, labels)
        ax.tick_params(axis="x", length=0, pad=10, labelsize=10)
        ax.tick_params(axis="y", length=0, pad=7, labelsize=9, colors=MUTED)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=.65)
        for name, spine in ax.spines.items():
            spine.set_visible(name == "bottom")
            spine.set_color("#D9DDE3")
            spine.set_linewidth(.8)
        for bar, row, value in zip(bars, rows, values):
            ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width()/2, value),
                        xytext=(0, 6), textcoords="offset points", ha="center",
                        va="bottom", fontsize=11.5,
                        weight="bold" if value == best else "normal",
                        color="#174B9C" if row[1] != "baseline" else MUTED)
    fig.text(.055, .045, "RL = RLCD · Bold values mark the best result for each metric.",
             fontsize=9, color=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
    svg = output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"Wrote {output.relative_to(ROOT)}.png / .svg / .pdf")


def main():
    snapshot = json.loads((ROOT / "assets/figures/general/results.json").read_text())
    general = [(row["size"], row["method"], row["accuracy_pct"], row["mean_latency_ms"])
               for row in snapshot["models"]]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
        "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": INK,
        "ytick.color": MUTED, "svg.fonttype": "path", "pdf.fonttype": 42,
        "svg.hashsalt": "valen-readme-results",
    })
    draw_comparison("General", "5,000 questions · Accuracy axis truncated to 70–82%", general,
                    ROOT / "assets/figures/general/readme-results", accuracy_range=(70, 82))
    draw_comparison("Sokoban", "500 single-step questions from 100 levels", SOKOBAN,
                    ROOT / "assets/figures/sokoban/readme-results", latency_max=200)


if __name__ == "__main__":
    main()
