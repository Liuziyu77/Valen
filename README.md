<p align="center">
  <img src="assets/branding/valen-logo-v2.png" alt="Valen — a slate-blue V with flowing motion trails and a muted rose crown" width="720">
</p>

<h1 align="center"><img src="assets/branding/valen-motto.svg" alt="Valen — Run to stay in place. Evolve to move forward." width="700"></h1>

<h2 align="center">System One Model, now with vision.</h2>

<p align="center">
  A multimodal decision model inspired by <a href="https://typesafe.ai/blog/introducing-system-one-models-and-jev">Jev</a> — text, images and video in; decision probabilities out.
</p>

<p align="center">
  <a href="https://huggingface.co/Valen-Team"><img src="https://img.shields.io/badge/Hugging_Face-Valen_Team-FFD21E?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Hugging Face: Valen Team"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-80A51B?style=flat&amp;labelColor=555555" alt="License: Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-1683BB?style=flat&amp;logo=python&amp;logoColor=FFD43B&amp;labelColor=555555" alt="Python 3.10+"></a>
  <a href="configs/train/"><img src="https://img.shields.io/badge/Training-SFT_%7C_RLCD-8A63B8?style=flat&amp;labelColor=555555" alt="SFT and experimental RLCD"></a>
  <br>
  <a href="https://huggingface.co/Valen-Team/Valen-Preview-0923"><img src="https://img.shields.io/badge/Model-Preview_0923-E88B23?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Model: Valen-Preview-0923"></a>
  <a href="https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k"><img src="https://img.shields.io/badge/Train-General_100k-2185B5?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Training dataset: Valen-Training-General-100k"></a>
  <a href="https://huggingface.co/datasets/Valen-Team/Valen-Eval-General-5k"><img src="https://img.shields.io/badge/Eval-General_5k-2185B5?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Evaluation dataset: Valen-Eval-General-5k"></a>
  <a href="https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game"><img src="https://img.shields.io/badge/Eval-Game-2185B5?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Evaluation dataset: Valen-Eval-Game"></a>
</p>

<p align="center">
  <b>English</b> · <a href="README_zh.md">简体中文</a><br>
  <a href="#requirements">🛠️ Requirements</a> · <a href="#quick-start">🚀 Quick start</a> · <a href="#training">🧠 Training</a> · <a href="#results">📊 Results</a> · <a href="#demo-comparison">🎬 Demos</a> · <a href="#documentation">📚 Documentation</a>
</p>

The Red Queen hypothesis (proposed by Leigh Maiorana Van Valen) in the age of AI: intelligence must keep pace with new tasks and environments.

Valen brings visual perception to System One decision-making. Inspired by [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), it evaluates text, images and video against task instructions and returns probabilities over supplied candidates, giving software a structured decision interface. A Qwen3.5-0.8B or 2B backbone and a shared decision head score candidates without generating answer tokens. The repository includes the model implementation, data processing, SFT and experimental RLCD training, and inference and evaluation commands, with support for training on your own data.

<a id="demo-comparison"></a>

<p align="center">
  <a href="assets/demos/sokoban-model-comparison.mp4"><img src="assets/demos/sokoban-model-comparison.gif" alt="Side-by-side Sokoban demo comparing Valen-Sokoban-SFT-RLCD-2B with Qwen3.8-27B-FP8 in no-thinking and thinking modes." width="1000"></a><br>
  <sub><strong>Valen-Preview-0923 solves the puzzle with 1.13 s of cumulative decision latency; Qwen3.8-27B-FP8 takes 198.05 s with thinking and fails to solve it without thinking.</strong></sub><br>
</p>

<a id="output-types"></a>

## 🎯 Output types

| Output | Use it for | Response |
| --- | --- | --- |
| **Choice** | Select among 1–255 candidates | Selected candidate, full probability distribution, confidence |
| **Noul** | Check whether a condition holds | Probability of `true` |
| **Score** | Rate against 2–10 ordered descriptions | Expected level index, level probabilities, legend, confidence |

Candidates are supplied with each question. Tasks with different candidate counts share the same decision head. `confidence` measures how concentrated the distribution is.

<p align="center">
  <img src="assets/figures/readme-architecture.png" alt="Valen architecture: multimodal input and candidates pass through Qwen3.5, an added shared decision head and softmax to produce Choice, Noul or Score outputs." width="1000">
</p>

Choice and Noul score all candidates in one branch per question. Score runs a separate backbone forward pass for each level, then normalizes the logits together. The shared state is encoded once by the input compiler and processed by the backbone in each branch. See the [architecture](docs/architecture.md) and [response fields and formulas](docs/evaluation.md).

<a id="requirements"></a>

## 🛠️ Requirements

- Linux, Python 3.10+, and an NVIDIA GPU.
- PyTorch 2.6.0, torchvision 0.21.0, Transformers 5.4.0, and PEFT 0.18.1.
- All dependencies are declared in [pyproject.toml](pyproject.toml) and installed by the setup script.

```bash
git clone https://github.com/Liuziyu77/Valen.git Valen
cd Valen
bash scripts/setup/bootstrap.sh
source .venv/bin/activate
```

For an existing Conda or virtualenv environment, see [manual installation](scripts/README.md#installation). The setup script creates `.venv`; it does not download model weights.

<a id="quick-start"></a>

## 🚀 Quick start

Install the dependencies listed in [requirements](#requirements), then download the model weights.

```bash
# Download the Qwen3.5-2B base model.
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B

# Train a decision head on the bundled synthetic examples.
python -m valen.train \
  --config configs/train/sft_warmup.json

# Run inference with the resulting checkpoint.
python -m valen.inference \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/predictions.jsonl

# Check the evaluation pipeline on the same synthetic examples.
python -m valen.evaluate \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/smoke_eval
```

`data/smoke` contains a small set of simple questions for checking that the pipeline runs correctly.

<details>
<summary>A labeled record with an image input</summary>

Each JSONL line contains one record. The example below uses the [general experiment overview figure](assets/figures/general/overview.png) from the repository, assuming the file is saved as `example.jsonl` in the repository root.

```json
{
  "group_id": "general-overview-2b",
  "request": {
    "state": {
      "messages": [{
        "role": "user",
        "content": [
          {"type": "text", "text": "Compare the accuracy and the average latency per question of the 2B models in the figure."},
          {"type": "image_url", "image_url": {"url": "assets/figures/general/overview.png"}}
        ]
      }]
    },
    "questions": {
      "best_2b": {
        "type": "choice",
        "instructions": "Among the 2B models with average latency below 200 ms per question, which has the highest accuracy?",
        "criteria": {
          "qwen": "Qwen3.5-2B",
          "sft": "Valen-Base-SFT-2B",
          "rlcd": "Valen-Base-RLCD-2B"
        }
      }
    }
  },
  "targets": {
    "best_2b": {"probabilities": {"qwen": 0.0, "sft": 1.0, "rlcd": 0.0}}
  }
}
```
</details>

<a id="training"></a>

## 🧠 Training

Valen supports two training methods, `SFT` and `RLCD`, selected by `method`. The `stage` setting controls which parameters are updated, with four stages available.

<p align="center">
  <img src="assets/figures/readme-training-workflow.png" alt="Train with SFT, optionally initialize RLCD from its checkpoint, and evaluate on held-out data." width="1000">
</p>

SFT and RLCD both use labeled training data. Checkpoints from either method can be evaluated on held-out data and require the corresponding base model. Language-model fine-tuning currently uses LoRA; full fine-tuning is a future option.

| Stage | Trainable parameters | SFT | RLCD |
| --- | --- | --- | --- |
| `warmup` | Decision head | [Config](configs/train/sft_warmup.json) | [Config](configs/train/rlcd_warmup.json) |
| `text` | LLM LoRA + decision head; text-only data | [Config](configs/train/sft_text.json) | [Config](configs/train/rlcd_text.json) |
| `joint` | LLM LoRA + ViT merger + decision head | [Config](configs/train/sft_joint.json) | [Config](configs/train/rlcd_joint.json) |
| `vision_top` | `joint` + last four vision encoder layers | [Config](configs/train/sft_vision_top.json) | [Config](configs/train/rlcd_vision_top.json) |

For a full run, copy a config and set `model_path`, `data`, `output`, `epochs` and `max_steps`. `tokens_per_step` is a **per-GPU** budget summed over all question branches in each state. Training uses **data parallelism**, with a complete model on each GPU. Defaults and examples are in the [configuration reference](docs/configuration.md).

```bash
# Eight GPUs on one machine. Use a fresh output directory for each run.
VJ_GPUS=8 bash scripts/train/launch_sft.sh configs/train/sft_warmup.json

# RLCD starts from an SFT-trained decision-head checkpoint.
VJ_GPUS=8 bash scripts/train/launch_sft.sh configs/train/rlcd_warmup.json \
  --initialize output/sft_warmup/latest
```

SFT trains on supervised labels. RLCD uses a GRPO-style objective with rewards based on correctness and confidence error, a KL penalty against a fixed reference policy, and an optional Brier loss. Equations, initialization and resume commands are in the [training guide](valen/training/README.md).

Checkpoints store trainable parameter values and training state under `<output>/latest/`; the frozen base model is loaded separately. Each save overwrites `latest/`, without retaining previous versions. `--initialize` starts a new experiment from existing weights; `--resume` restores the optimizer, data cursor and per-rank state and requires the same process count. `launch_sft.sh` supports both SFT and RLCD; use `VJ_PYTHON` to select an interpreter.

<a id="results"></a>

## 📊 Results

### Model Card

We ran ablation experiments on training methods and training data:

| Model | Training method | Data | Datasets |
| --- | --- | --- | --- |
| Valen-Base-SFT-0.8B | SFT | General SFT 100k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) |
| Valen-Base-SFT-2B | SFT | General SFT 100k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) |
| Valen-Base-RLCD-0.8B | SFT + RLCD | General SFT 70k + General RLCD 30k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) |
| Valen-Base-RLCD-2B | SFT + RLCD | General SFT 70k + General RLCD 30k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) |
| Valen-Sokoban-SFT-2B | SFT + SFT | General SFT 100k + Sokoban SFT 30k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k)&nbsp;[![Sokoban](https://img.shields.io/badge/Sokoban-8A63B8?style=flat&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) |
| Valen-Sokoban-RLCD-2B | SFT + RLCD | General SFT 100k + Sokoban RLCD 30k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k)&nbsp;[![Sokoban](https://img.shields.io/badge/Sokoban-8A63B8?style=flat&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) |

Both the Sokoban training and evaluation sets are included in [Valen-Eval-Game](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game). Use the corresponding split for training or evaluation.

#### Training Loss

<p align="center">
  <a href="assets/figures/training/sft100k_loss.png"><img src="assets/figures/training/sft100k_loss.png" alt="Training loss for Valen-Base-SFT-0.8B and 2B on General 100k data." width="1000"></a><br>
  <sub>General SFT: training loss for the 0.8B and 2B models.</sub>
</p>

<p align="center">
  <a href="assets/figures/training/sokoban_rlcd_loss_reward.png"><img src="assets/figures/training/sokoban_rlcd_loss_reward.png" alt="Training loss and mean reward for the 0.8B and 2B Valen-Sokoban-RLCD models." width="1000"></a><br>
  <sub>Sokoban RLCD: training loss and reward for the 0.8B and 2B models.</sub>
</p>

### General: visual question answering

The experiments train on [100k records](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) and test on [5k questions](https://huggingface.co/datasets/Valen-Team/Valen-Eval-General-5k), using Qwen3.5-0.8B and 2B backbones. Each run used eight H200 GPUs and updated only the decision head: SFT used all 100k records, while the two-stage run used 70k for SFT and the remaining 30k for RLCD. The figures label the trained models `Valen-Base-*`; `Qwen3.5-*` are the original generation baselines.

<p align="center">
  <a href="assets/figures/general/overview.png"><img src="assets/figures/general/overview.png" alt="Accuracy and mean end-to-end latency for six models on 5,000 general image questions." width="1000"></a>
</p>

**Valen-Base-SFT-2B scored 79.02%**, **3.52 percentage points** above the same-size baseline. The 0.8B SFT and RLCD models averaged 156.65 and 155.82 ms per question, about 1.51 times faster than the 235.98 ms baseline. End-to-end latency includes preprocessing, model computation and output handling.

<p align="center">
  <a href="assets/figures/general/task-latency.png"><img src="assets/figures/general/task-latency.png" alt="Mean latency of the four trained models on Choice, Noul and Score questions." width="1000"></a>
</p>

Choice and Noul score candidates in one branch; Score runs a separate forward pass per level. The next chart breaks accuracy down by question type and application domain. Score has only 18 questions.

<p align="center">
  <a href="assets/figures/general/breakdown.png"><img src="assets/figures/general/breakdown.png" alt="Accuracy of six models by Choice, Noul and Score task and by documents, games, interfaces and visual QA." width="1000"></a>
</p>

See the [general experiment report](docs/experiments/general.md) for the full setup, per-source scores, P50/P95 latency and probability metrics.

### Sokoban: box-pushing puzzles

The four `Valen-Sokoban-*` models start from the general experiment's 100k SFT decision head and continue training on [Sokoban](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) data for three epochs at the `vision_top` stage. All six models were tested on the same 500 single-step questions drawn from 100 levels. **Any optimal action counts as correct.**

<p align="center">
  <a href="assets/figures/sokoban/overview.png"><img src="assets/figures/sokoban/overview.png" alt="Accuracy and mean end-to-end latency for six models on 500 Sokoban single-step questions." width="1000"></a>
</p>

| Model | Correct / 500 | Mean single-step end-to-end latency | Full games solved / 100 |
| --- | ---: | ---: | ---: |
| Qwen3.5-0.8B | 158 (31.60%) | 152.54 ms | Not evaluated |
| Valen-Sokoban-SFT-0.8B | 403 (80.60%) | 114.06 ms | 18 |
| Valen-Sokoban-RLCD-0.8B | 415 (83.00%) | 113.95 ms | 24 |
| Qwen3.5-2B | 136 (27.20%) | 155.73 ms | 0 |
| Valen-Sokoban-SFT-2B | 418 (83.60%) | 126.82 ms | 20 |
| Valen-Sokoban-RLCD-2B | 438 (87.60%) | 127.31 ms | 38 |

Single-step accuracy does not imply full-game success. Latency was measured on one H200 GPU with batch size 1 after three warmup questions; model loading is excluded. Each full game starts from the initial board, allows one attempt and stops after at most 200 moves, with no action cache, undo or solver assistance. The time panel below includes successful games only. See the [Sokoban guide](evaluation/sokoban/README.md) for the evaluation workflow.

<p align="center">
  <a href="assets/figures/sokoban/full-games.png"><img src="assets/figures/sokoban/full-games.png" alt="Games solved out of 100 Sokoban levels, with elapsed times for each model's successful games." width="1000"></a>
</p>

<a id="demos"></a>

## 🎬 Demos

### Four games in parallel

Four successful **Valen-Preview-0923** trajectories run side by side at 1× speed with no playback acceleration. Each game takes 7–10 decisions, averaging 122–128 ms per step, and all four finish within 1.24 seconds.

<p align="center">
  <a href="assets/demos/sokoban-four-game-showcase.mp4"><img src="assets/demos/sokoban-four-game-showcase.gif" alt="Four successful Valen-Preview-0923 games replayed in parallel at recorded speed." width="1000"></a><br>
</p>


<a id="repository-layout"></a>

## 📁 Repository layout

```text
valen/
  data/          Record validation, media loading and candidate compilation
  modeling/      Qwen3.5 backbone, decision head and trainable parameter groups
  training/      SFT/RLCD objectives, shared loop, gradient sync and checkpoints
  evaluation/    Inference responses, metrics and distributed evaluation
evaluation/      Task environments, Sokoban data generation and full-game evaluation
configs/train/   Eight example configs: two objectives × four stages
scripts/         Environment setup, model preparation, training launchers and GPU checks
data/smoke/      Small synthetic text, image and video fixtures
tests/           CPU tests, including two-process training and resume checks
docs/            Architecture, data, configuration and evaluation references
```

<a id="documentation"></a>

## 📚 Documentation

| Resource | Contents |
| --- | --- |
| [Architecture](docs/architecture.md) | Compilation, candidate scoring, branch cost and gradient synchronization |
| [Configuration reference](docs/configuration.md) | Defaults, token budgets, optimizer settings and RLCD options |
| [Training guide](valen/training/README.md) | SFT/RLCD objectives, distributed training and checkpoint recovery |
| [Script guide](scripts/README.md) | Setup, training, evaluation and data tools |
| [Code map](valen/README.md) | Data, modeling, training and evaluation modules |
| [Data format](docs/data-format.md) | Complete examples, labels, local media and split conventions |
| [Inference and evaluation](docs/evaluation.md) | Response fields, confidence formulas, metrics and timing |
| [Experiment report](docs/experiments/general.md) | 100k training comparison, full scores and latency protocol |
| [Task evaluation](evaluation/README.md) | Sokoban generation, full-game evaluation, policy comparison and replay |

<a id="license-and-acknowledgments"></a>

## 🤝 License and acknowledgments

The code is released under [Apache 2.0](LICENSE). Base models and source datasets retain their respective licenses.

Built on [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B), with the decision interface inspired by [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

Contributions to Valen are welcome. Open an [issue](https://github.com/Liuziyu77/Valen/issues) to report a problem, share a use case or discuss experimental results. Submit a [pull request](https://github.com/Liuziyu77/Valen/pulls) to improve the code or documentation, contribute training data or add evaluation tasks.
