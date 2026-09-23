# Eval_3 result charts

Figures and `results.json` come from the source experiment at revision `fc6210f`. The protocol, full scores and timing definitions are in the [experiment report](../../../docs/experiments/eval3.md).

| Figure | PNG | SVG | PDF |
| --- | --- | --- | --- |
| Accuracy | [PNG](accuracy.png) | [SVG](accuracy.svg) | [PDF](accuracy.pdf) |
| Mean latency | [PNG](latency.png) | [SVG](latency.svg) | [PDF](latency.pdf) |
| Latency by question type | [PNG](latency-by-task.png) | [SVG](latency-by-task.svg) | [PDF](latency-by-task.pdf) |
| Accuracy by domain | [PNG](accuracy-by-domain.png) | [SVG](accuracy-by-domain.svg) | [PDF](accuracy-by-domain.pdf) |

Accuracy bars use a 60–85% scale. Latency bars start at zero and show only SFT/RLCD models. The domain table includes all six models, highlighting the best score within each model size. Gray denotes the baseline, teal SFT, and terracotta RLCD.

The [snapshot](results.json) contains the plotted values and the original source-metric digests. It is sufficient to redraw the figures; the original per-question predictions and model weights are not included.

From the repository root, install the plotting dependency and render:

```bash
python -m pip install matplotlib
python scripts/eval/plot_eval3.py

# Render to a separate directory.
python scripts/eval/plot_eval3.py --output-dir /tmp/valen-eval3-preview
```

The renderer writes PNG, SVG and PDF, using Lato when available and DejaVu Sans otherwise. It reads the bundled snapshot without accessing the original experiment outputs.
