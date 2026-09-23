<p align="center">
  <img src="assets/branding/visual-jev-banner.png" alt="Visual-Jev" width="720">
</p>

<h1 align="center">Visual-Jev</h1>

<p align="center">Multimodal inputs. Structured decisions. Candidate probabilities.</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-0F766E?style=flat-square" alt="License: Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-24465C?style=flat-square" alt="Python 3.10+"></a>
  <a href="configs/train/"><img src="https://img.shields.io/badge/Training-SFT_%7C_RLCD-0F766E?style=flat-square" alt="SFT and experimental RLCD"></a>
</p>

<p align="center">
  <b>English</b> · <a href="README_zh.md">简体中文</a><br>
  <a href="#requirements">Requirements</a> · <a href="#quick-start">Quick start</a> · <a href="#training">Training</a> · <a href="#documentation">Documentation</a>
</p>

Visual-Jev turns text, images and video into decisions over candidates you define. It pairs a Qwen3.5 backbone with a shared decision head and returns probabilities directly, without generating answer tokens. The repository includes SFT, experimental RLCD, single-node multi-GPU training, and evaluation tools for Qwen3.5-0.8B and 2B.

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

Choice and Noul score all candidates in one branch per question. Score encodes each level in a separate branch with shared weights. See the [architecture and output definitions](docs/architecture.md) for details.

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

For an existing Conda/virtualenv environment or optional data tools, see [manual installation](scripts/README.md#installation).

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

The bundled examples cover all three output types, text, images and video. They check the pipeline; they are not a benchmark or a useful trained model. Model weights are downloaded separately. The download helper targets 0.8B; to use 2B, prepare an official Qwen3.5-2B snapshot in `models/Qwen3.5-2B` and set `model_path` in your config.

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

`targets` are training/evaluation labels and are excluded from model input. Media paths are relative to the JSONL file. See the [synthetic examples](data/smoke/) and [data format](docs/data-format.md).

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

For a full run, set `model_path`, `data`, `output`, `epochs` and `max_steps` in your config. The stage examples use smoke data and short step limits. `tokens_per_step` is a **per-GPU** input budget.

```bash
# Eight GPUs on one machine. Use a fresh output directory for each run.
VJ_GPUS=8 bash scripts/train/launch_sft.sh configs/train/sft_warmup.json

# RLCD starts from an existing decision-head checkpoint.
VJ_GPUS=8 bash scripts/train/launch_sft.sh configs/train/rlcd_warmup.json \
  --initialize output/sft_warmup/latest
```

SFT optimizes supervised labels. RLCD uses a GRPO-style clipped objective with correctness and confidence-error rewards, a fixed-reference KL penalty, and an optional Brier loss. This project's reward formula is experimental and does not guarantee calibration. Equations, initialization and resume commands are in the [training guide](visionjev/training/README.md).

Checkpoints store trainable parameter deltas and training state; loading them also requires the original base model. `launch_sft.sh` supports both objectives and accepts `VJ_PYTHON` to select an interpreter explicitly.

## Documentation

| Resource | Contents |
| --- | --- |
| [Model design](docs/architecture.md) | Candidate scoring, media inputs and output semantics |
| [Training guide](visionjev/training/README.md) | SFT/RLCD objectives, distributed training and checkpoint recovery |
| [Script guide](scripts/README.md) | Setup, training, evaluation and smoke checks |
| [Code map](visionjev/README.md) | Data, modeling, training and evaluation modules |
| [Data notes](docs/data-format.md) | JSONL schema, media paths and split conventions |

The detailed guides are currently in Chinese; both READMEs cover setup and training. To run the CPU suite (processor tests skip until the base-model processor is available):

```bash
python -m pytest -q
```

## Roadmap and contributions

- [ ] Calibration ablations with matched optimizer budgets and independent validation data.
- [ ] Visual demos for degraded images, UI/document checks, and game decisions.
- [ ] Broader benchmarks and downloadable trained checkpoints.

Issues and pull requests are welcome. For training or evaluation changes, include the config, base-model revision, dataset split and validation results. Report accuracy alongside probability quality when changing confidence-related behavior.

## License and acknowledgments

The code is released under [Apache 2.0](LICENSE). Base models and source datasets retain their respective licenses.

Built on [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B), with the decision interface inspired by [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). The experimental policy objective follows the GRPO approach introduced in [DeepSeekMath](https://arxiv.org/abs/2402.03300).
