# Task evaluation

This directory contains environments for evaluating complete decision sequences. For labeled JSONL questions and probability metrics, use `python -m valen.evaluate`; see the [script guide](../scripts/README.md).

The [JevBench evaluator](jevbench/README.md) runs local Valen checkpoints on a
frozen 231-task public snapshot, with upstream scoring, calibration and latency
reports. Its local experiment files live in the Git-ignored `JevBench-exp/`.

The [Sokoban benchmark](sokoban/README.md) includes a deterministic environment, an exact solver, synthetic data generation and validation, full-game evaluation, result comparison, and trajectory replay. It supports Valen, the original Qwen generation model, a random policy, and an oracle sanity check.

[game-v4 and arcade-v2](game/README.md) add labeled action scoring and complete-game evaluation for Maze, FrozenLake, Sokoban, LaneRacer and ParkourRunner. The new `game-v4/sokoban` and legacy `sokoban-v1/sokoban` results are reported separately.

Run commands from the repository root after installing the project. Generated data and results are excluded from Git. The modules are included in the Python package; default data paths are relative to the working directory.

## Scope

Reusable evaluation components were selected from the development repository. Cluster job wrappers, training diagnostics, experiment-specific subset selection, alternate test-set variants, report snapshots and presentation scripts are not included. No historical experiment results or model checkpoints are bundled.
