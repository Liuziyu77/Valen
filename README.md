<p align="center">
  <img src="assets/branding/visual-jev-banner.png" alt="Visual-Jev" width="720">
</p>

<h1 align="center">Visual-Jev</h1>

<p align="center">Candidate scoring for text, images and video.</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-0F766E?style=flat-square" alt="License: Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-24465C?style=flat-square" alt="Python 3.10+"></a>
  <a href="configs/train/"><img src="https://img.shields.io/badge/Training-SFT_%7C_RLCD-0F766E?style=flat-square" alt="SFT and experimental RLCD"></a>
</p>

<p align="center">
  <b>English</b> · <a href="README_zh.md">简体中文</a><br>
  <a href="#requirements">Requirements</a> · <a href="#quick-start">Quick start</a> · <a href="#training">Training</a> · <a href="#documentation">Documentation</a>
</p>

Visual-Jev scores user-defined candidates against text, images and video. It pairs a Qwen3.5 backbone with a shared decision head and returns a probability distribution without generating answer tokens. The repository contains the model, JSONL data pipeline, SFT and experimental RLCD training, and inference/evaluation commands.

The supplied configs and GPU smoke scripts target Qwen3.5-0.8B. A local Qwen3.5-2B snapshot can be selected through `model_path`; the download helper is specific to 0.8B. Trained decision-head weights and benchmark results are not included, so start by training a head for your task.

This is an independent research implementation inspired by [Jev and System One models](https://typesafe.ai/blog/introducing-system-one-models-and-jev). It is not an official TypeSafe implementation. The model class and response `model` field use `VisualJev`; the Python package and CLI remain `visionjev` for compatibility.

## What it returns

| Output | Use it for | Response |
| --- | --- | --- |
| **Choice** | Select among 1–255 named candidates | Selected candidate, full probability distribution, confidence |
| **Noul** | Check whether a condition holds | Probability of `true` |
| **Score** | Rate against 2–10 ordered descriptions | Expected level index, level probabilities, legend, confidence |

Candidates are supplied with each question. The head is shared across tasks and candidate counts. `confidence` describes distribution concentration; it is not a measured probability that an answer is correct.

<p align="center">
  <img src="assets/figures/readme-architecture.png" alt="Visual-Jev architecture: multimodal input and candidates pass through Qwen3.5, an added shared decision head and softmax to produce Choice, Noul or Score outputs." width="1000">
</p>

Choice and Noul score all candidates in one branch per question. Score runs a separate backbone forward for each level, then normalizes the logits together. The state is processed once by the input compiler but recomputed by the backbone in each branch. See the [architecture](docs/architecture.md) and [response fields and formulas](docs/evaluation.md).

## Requirements

- Linux, Python 3.10+, and an NVIDIA GPU supporting BF16 with a CUDA 12.4-compatible driver.
- PyTorch 2.6.0, torchvision 0.21.0, Transformers 5.4.0, and PEFT 0.18.1.
- All dependencies are declared in [pyproject.toml](pyproject.toml) and installed by the setup script.

```bash
git clone https://github.com/Liuziyu77/Visual-Jev.git Visual-Jev
cd Visual-Jev
bash scripts/setup/bootstrap.sh
source .venv/bin/activate
```

For an existing Conda or virtualenv environment, see [manual installation](scripts/README.md#installation). The setup script creates `.venv`; it does not download model weights.

## Quick start

Install the [requirements](#requirements) and keep that environment activated. Model weights are downloaded separately from the Python packages.

```bash
# Download and verify the pinned Qwen3.5-0.8B snapshot.
python scripts/setup/prepare_model.py

# Train a decision head on the bundled synthetic examples.
python -m visionjev.train \
  --config configs/train/sft_warmup.json

# Run inference with the resulting checkpoint.
python -m visionjev.inference \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/predictions.jsonl

# Check the evaluation pipeline on the same synthetic examples.
python -m visionjev.evaluate \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/smoke_eval
```

The bundled data has six records and fifteen questions, thirteen with labels. Inference writes six response lines; evaluation writes thirteen labeled-question predictions plus `metrics.json`. These examples check the pipeline, not model quality. Training stops after three epochs or 100 batches, whichever comes first; a small dataset can finish well before 100 steps.

Run the commands from the repository root. Paths in configs are relative to the working directory; media paths inside JSONL records are relative to that JSONL file. To use 2B, prepare a complete local snapshot with its processor files and change `model_path` in a config copy.

<details>
<summary>A minimal labeled record</summary>

Each JSONL line contains one record. This expanded example defines a binary question with a hard label; soft probability labels are also supported.

```json
{
  "group_id": "traffic-light-001",
  "request": {
    "state": "The traffic light is green.",
    "questions": {
      "go": {
        "type": "noul",
        "instructions": "Only a green light permits crossing. Is crossing permitted?"
      }
    }
  },
  "targets": {
    "go": {"probabilities": {"true": 1.0, "false": 0.0}}
  },
  "assets": []
}
```

`targets` are training/evaluation labels and are excluded from model input. They can be omitted for inference, which still requires `group_id`. A Noul answer has the form `{"type": "noul", "noul": 0.8}`, where the number is `P(true)`. See the [synthetic examples](data/smoke/) and [data format](docs/data-format.md) for Choice, Score and media records.

</details>

## Training

`method` selects the objective; `stage` selects the parameters to update. The stages are available configurations, not a requirement to run all four in order.

<p align="center">
  <img src="assets/figures/readme-training-workflow.png" alt="Train with SFT, optionally initialize RLCD from its checkpoint, and evaluate on held-out data." width="1000">
</p>

SFT and RLCD both use labeled training data. Evaluate either checkpoint on held-out data; loading a checkpoint also requires its base model.

| Stage | Trainable parameters | SFT | RLCD |
| --- | --- | --- | --- |
| `warmup` | Decision head | [Config](configs/train/sft_warmup.json) | [Config](configs/train/rlcd_warmup.json) |
| `text` | Language LoRA + decision head; text-only data | [Config](configs/train/sft_text.json) | [Config](configs/train/rlcd_text.json) |
| `joint` | Language LoRA + vision merger + decision head | [Config](configs/train/sft_joint.json) | [Config](configs/train/rlcd_joint.json) |
| `vision_top` | `joint` + last four vision encoder layers | [Config](configs/train/sft_vision_top.json) | [Config](configs/train/rlcd_vision_top.json) |

For a full run, copy a config and set `model_path`, `data`, `output`, `epochs` and `max_steps`. `tokens_per_step` is a **per-GPU** budget summed over all question branches in each state. Each GPU holds a complete model; this is data parallel training. Defaults and configuration examples are in the [configuration reference](docs/configuration.md).

```bash
# Eight GPUs on one machine. Use a fresh output directory for each run.
VJ_GPUS=8 bash scripts/train/launch_sft.sh configs/train/sft_warmup.json

# RLCD starts from an existing decision-head checkpoint.
VJ_GPUS=8 bash scripts/train/launch_sft.sh configs/train/rlcd_warmup.json \
  --initialize output/sft_warmup/latest
```

SFT optimizes supervised labels. RLCD uses a GRPO-style clipped objective with correctness and confidence-error rewards, a fixed-reference KL penalty, and an optional Brier loss. This project's reward formula is experimental and does not guarantee calibration. Equations, initialization and resume commands are in the [training guide](visionjev/training/README.md).

Checkpoints store trainable parameter values and training state under `<output>/latest/`; the frozen base model is loaded separately. Saving replaces `latest/` rather than keeping a history. `--initialize` starts a new experiment from those weights; `--resume` restores the optimizer, data cursor and per-rank random state and requires the same process count. `launch_sft.sh` supports both objectives and accepts `VJ_PYTHON` to select an interpreter.

## Repository layout

```text
visionjev/
  data/          Record validation, media loading and candidate compilation
  modeling/      Qwen3.5 backbone, decision head and trainable parameter groups
  training/      SFT/RLCD objectives, shared loop, gradient sync and checkpoints
  evaluation/    Inference responses, metrics and distributed evaluation
evaluation/      Task environments, Sokoban data generation and full-game evaluation
configs/train/   Eight example configs: two objectives × four stages
scripts/         Environment setup, model preparation, launchers and smoke checks
data/smoke/      Small synthetic text, image and video fixtures
tests/           CPU tests, including two-process training and resume checks
docs/            Architecture, data, configuration and evaluation references
```

The top-level `visionjev/train.py`, `inference.py` and `evaluate.py` forward to the corresponding subpackages. The [code map](visionjev/README.md) lists implementation entry points and their tests.

## Documentation

| Resource | Contents |
| --- | --- |
| [Architecture](docs/architecture.md) | Compilation, candidate scoring, branch cost and gradient synchronization |
| [Configuration reference](docs/configuration.md) | Defaults, token budgets, optimizer settings and RLCD options |
| [Training guide](visionjev/training/README.md) | SFT/RLCD objectives, distributed training and checkpoint recovery |
| [Script guide](scripts/README.md) | Setup, training, evaluation and smoke checks |
| [Code map](visionjev/README.md) | Data, modeling, training and evaluation modules |
| [Data format](docs/data-format.md) | Complete examples, labels, local media and split conventions |
| [Inference and evaluation](docs/evaluation.md) | Response fields, confidence formulas, metrics and timing |
| [Task evaluation](evaluation/README.md) | Sokoban generation, full-game evaluation, policy comparison and replay |

The detailed guides are currently in Chinese; both READMEs cover setup and training. To run the CPU suite (processor tests skip until the base-model processor is available):

```bash
python -m pytest -q
```

## Development

The current tools are file-based CLIs. There is no HTTP serving layer, validation-driven checkpoint selection or temperature-fitting command. The generic JSONL reader does not split datasets; the Sokoban generator creates its own separate training and test sets. Inference accepts a temperature file; the evaluation CLI always uses temperature 1. See the [script guide](scripts/README.md) for GPU integration checks and common errors.

For training or evaluation changes, include the config, base-model revision, dataset split and relevant test results. Report accuracy alongside NLL and Brier when changing confidence-related behavior. Comparisons between SFT and RLCD should account for actual optimizer updates: by default, RLCD performs two updates per batch and SFT performs one.

## License and acknowledgments

The code is released under [Apache 2.0](LICENSE). Base models and source datasets retain their respective licenses.

Built on [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B), with the decision interface inspired by [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). The experimental policy objective follows the GRPO approach introduced in [DeepSeekMath](https://arxiv.org/abs/2402.03300).
