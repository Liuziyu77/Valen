# General experiment figures

The [results snapshot](results.json) records the experiment values, timing definition, source metric paths and digests from revision `fc6210f`. Its original experiment identifier is retained for provenance.

To draw four additional detail charts from the snapshot, install Matplotlib and choose an output directory:

```bash
python -m pip install matplotlib
python scripts/eval/plot_general.py --output-dir /tmp/valen-general-details
```

The script writes PNG, SVG and PDF files to that directory. The snapshot does not include per-question predictions or model weights.

The main READMEs use a four-panel bar chart comparing General and Sokoban accuracy and latency for three models per panel.
To regenerate its PNG, SVG and PDF versions in `assets/figures/`, run:

```bash
python scripts/eval/plot_readme_overview.py
```
