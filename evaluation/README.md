# Task evaluation

This directory contains environments for evaluating complete decision sequences. For labeled JSONL questions and probability metrics, use `python -m visualjev.evaluate`; see the [script guide](../scripts/README.md).

The [Sokoban benchmark](sokoban/README.md) includes a deterministic environment, an exact solver, synthetic data generation and validation, full-game evaluation, result comparison, and trajectory replay. It supports Visual-Jev, the original Qwen generation model, a random policy, and an oracle sanity check.

Run commands from the repository root after installing the project. Generated data and results are excluded from Git. The modules are included in the Python package; default data paths are relative to the working directory.

## Scope

Reusable evaluation components were selected from the development repository. Cluster job wrappers, training diagnostics, experiment-specific subset selection, alternate test-set variants, report snapshots and presentation scripts are not included. No historical experiment results or model checkpoints are bundled.
