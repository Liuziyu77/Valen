# General experiment figures

The three PNGs used in the English and Chinese READMEs summarize the 100k training comparison on 5,000 general image questions:

| Figure | Shows |
| --- | --- |
| [Overview](overview.png) | Accuracy and mean end-to-end latency for six models |
| [Task latency](task-latency.png) | Choice, Noul and Score latency for the four trained models |
| [Breakdown](breakdown.png) | Accuracy by question type and application domain |

The [results snapshot](results.json) records the experiment values, along with the source metric paths and digests from revision `fc6210f`. Its original experiment identifier is retained for provenance. The [experiment report](../../../docs/experiments/general.md) gives the setup, full scores and timing definitions.

To draw four additional detail charts from the snapshot, install Matplotlib and choose an output directory:

```bash
python -m pip install matplotlib
python scripts/eval/plot_general.py --output-dir /tmp/valen-general-details
```

The script writes PNG, SVG and PDF files to that directory. It does not recreate the three README figures. The snapshot does not include per-question predictions or model weights.
