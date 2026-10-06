<p align="center">
  <img src="assets/branding/valen-logo-v5-compact.svg" alt="Valen — a teal aperture with a forward arrow and uppercase wordmark" width="720"><br>
  <img src="assets/branding/valen-slogan.svg" alt="System One Model, now with vision." width="700">
</p>

<p align="center">
  A multimodal decision model inspired by <a href="https://typesafe.ai/blog/introducing-system-one-models-and-jev">Jev</a> — text, images and video in; decision probabilities out.
</p>

<p align="center">
  <a href="https://huggingface.co/Valen-Team"><img src="https://img.shields.io/badge/Hugging_Face-Valen_Team-FFD21E?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Hugging Face: Valen Team"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-80A51B?style=flat&amp;labelColor=555555" alt="License: Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-1683BB?style=flat&amp;logo=python&amp;logoColor=FFD43B&amp;labelColor=555555" alt="Python 3.10+"></a>
  <a href="configs/train/"><img src="https://img.shields.io/badge/Training-SFT_%7C_RLCD-8A63B8?style=flat&amp;labelColor=555555" alt="SFT and experimental RLCD"></a>
  <br>
  <a href="https://huggingface.co/Valen-Team/Valen-0.8B"><img src="https://img.shields.io/badge/Valen-0.8B-E88B23?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Valen 0.8B"></a>
  <a href="https://huggingface.co/Valen-Team/Valen-2B"><img src="https://img.shields.io/badge/Valen-2B-E88B23?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Valen 2B"></a>
  <a href="https://huggingface.co/Valen-Team/Valen-4B"><img src="https://img.shields.io/badge/Valen-4B-E88B23?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Valen 4B"></a>
  <a href="https://huggingface.co/spaces/yuhangzang/Valen-Preview-0923"><img src="https://img.shields.io/badge/Space-Try_Demo-009C83?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Try Valen on Hugging Face Spaces"></a>
  <a href="https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k"><img src="https://img.shields.io/badge/Train-General_100k-2185B5?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Training dataset: Valen-Training-General-100k"></a>
  <a href="https://huggingface.co/datasets/Valen-Team/VisualDecisionBench"><img src="https://img.shields.io/badge/Eval-VisualDecisionBench-2185B5?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="VisualDecisionBench"></a>
  <a href="https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game"><img src="https://img.shields.io/badge/Eval-Game-2185B5?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Evaluation dataset: Valen-Eval-Game"></a>
</p>

<p align="center">
  <b>English</b> · <a href="README_zh.md">简体中文</a><br>
  <a href="#introduction">Intro</a> · <a href="#news">News</a> · <a href="#demos">Demos</a> · <a href="#model-downloads">Models &amp; datasets</a> · <a href="#results">Eval results</a> · <a href="#quick-start">Quick start</a>
</p>

<a id="introduction"></a>

## ✨ Introduction

Valen (万澜) brings visual perception to System One decision-making. Inspired by [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), it evaluates text, images and video against task instructions and returns probabilities over supplied candidates, giving software a structured decision interface. The repository includes the model implementation, data processing, SFT and experimental RLCD training, and inference and evaluation commands, with support for training on your own data.

<a id="news"></a>

## 📰 News

- **2026-10-06**: Released **Valen models** and the [**VisualDecisionBench**](https://huggingface.co/datasets/Valen-Team/VisualDecisionBench) benchmark.
- **2026-09-23**: Open-sourced the **Valen** repository and released [**Valen-preview**](https://huggingface.co/Valen-Team/Valen-Preview-0923).

<a id="demos"></a>

## 🎬 Demos

Try **Valen-preview** in the [online demo](https://huggingface.co/spaces/yuhangzang/Valen-Preview-0923).

<a id="demo-comparison"></a>

### Against a 27B generation model

**Valen-preview** reads the board image and selects a movement direction at each step. On the same level, Valen-preview solves the puzzle in 9 decisions with 1.13 seconds of cumulative decision latency. Qwen3.8-27B-FP8 takes 198.05 seconds in thinking mode and fails to solve it in no-thinking mode.

<p align="center">
  <img src="assets/demos/sokoban-model-comparison.gif" alt="Side-by-side Sokoban demo comparing Valen-preview with Qwen3.8-27B-FP8 in no-thinking and thinking modes." width="1000"><br>
</p>

### Image blur and action confidence

Gaussian blur reveals how **Valen-preview** adapts its decisions and confidence as visual detail decreases. Decision confidence is 91.6% on the clear image and 19.2% at the strongest blur.

<p align="center">
  <img src="assets/demos/valen-preview-0923-blur-confidence.gif" alt="Valen-preview action probabilities and decision confidence across nine measured levels of Gaussian image blur." width="1000"><br>
</p>

### Four games in parallel

Four successful **Valen-preview** trajectories run side by side at 1× speed with no playback acceleration. Each game takes 7–10 decisions, averaging 122–128 ms per step, and all four finish within 1.24 seconds.

<p align="center">
  <img src="assets/demos/sokoban-four-game-showcase.gif" alt="Four successful Valen-preview games replayed in parallel at recorded speed." width="1000"><br>
</p>

<a id="model-downloads"></a>

## 📥 Models and datasets

The **Valen** models and **VisualDecisionBench** dataset are available on Hugging Face.

### Model

| Model | Download |
| --- | --- |
| Valen 0.8B | [🤗 Valen-0.8B](https://huggingface.co/Valen-Team/Valen-0.8B) |
| Valen 2B | [🤗 Valen-2B](https://huggingface.co/Valen-Team/Valen-2B) |
| Valen 4B | [🤗 Valen-4B](https://huggingface.co/Valen-Team/Valen-4B) |
| Valen-preview (earlier release) | [🤗 Hugging Face](https://huggingface.co/Valen-Team/Valen-Preview-0923) |

The demos above and quick start below use **Valen-preview**, which requires both the **Valen checkpoint** and the **Qwen3.5-2B base model**.

### Dataset

| Dataset | Purpose | Download |
| --- | --- | --- |
| VisualDecisionBench | VisualDecisionBench-Image and VisualDecisionBench-Video evaluation | [🤗 Hugging Face](https://huggingface.co/datasets/Valen-Team/VisualDecisionBench) |
| General 100k (earlier release) | Training | [🤗 Hugging Face](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) |
| General 5k (earlier release) | Evaluation | [🤗 Hugging Face](https://huggingface.co/datasets/Valen-Team/Valen-Eval-General-5k) |
| Sokoban (earlier release) | Training and evaluation | [🤗 Hugging Face](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) |

<a id="results"></a>

## 📊 Evaluation results — Valen

**Visual decisions at 0.8B, 2B and 4B.** Valen uses Qwen3.5 + Mixer with shared-state inference, trained with SFT on millions of samples. The evaluation compares these checkpoints with published decision models on **VisualDecisionBench-Image, VisualDecisionBench-Video and public JevBench**.

<p align="center">
  <a href="assets/figures/evaluation-20261006/eval_v1.png"><img src="assets/figures/evaluation-20261006/eval_v1.png" alt="Valen VisualDecisionBench-Image accuracy by model size: 0.8B 71.59%, 2B 79.18%, 4B 83.06%, compared with published baselines." width="1400"></a>
</p>

<p align="center">
  <a href="assets/figures/evaluation-20261006/video.png"><img src="assets/figures/evaluation-20261006/video.png" alt="Valen VisualDecisionBench-Video accuracy by model size: 0.8B 79.71%, 2B 82.85%, 4B 87.36%, compared with published baselines." width="1400"></a>
</p>

<p align="center">
  <a href="assets/figures/evaluation-20261006/jevbench.png"><img src="assets/figures/evaluation-20261006/jevbench.png" alt="Valen public JevBench accuracy: 0.8B 73.59%, 2B 78.79%, 4B 83.12%. InternDecision 4B scores 86.58%." width="1400"></a>
</p>

Earlier results remain in the [archived README](history/README_preview.md).

<a id="quick-start"></a>

## 🚀 Quick start

Install the dependencies listed in [requirements](docs/technical.md#环境要求), then download both the Valen-preview checkpoint and its Qwen3.5-2B base model.

```bash
# Download the Valen-preview checkpoint.
hf download Valen-Team/Valen-Preview-0923 --local-dir models/Valen-Preview-0923

# Download the Qwen3.5-2B base model.
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B

# Train the model with your own configuration.
python -m valen.train \
  --config configs/train/qwen/sft_warmup.json

# Run inference with the downloaded Valen-preview checkpoint.
python -m valen.inference \
  --checkpoint models/Valen-Preview-0923 \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/predictions.jsonl

# Check the evaluation pipeline on the same synthetic examples.
python -m valen.evaluate \
  --checkpoint models/Valen-Preview-0923 \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/smoke_eval
```

`data/smoke` contains a small set of simple questions for checking that the pipeline runs correctly.

<details>
<summary>A labeled record with an image input</summary>

Each JSONL line contains one record. The example below uses the [evaluation overview figure](assets/figures/evaluation-results.png) from the repository, assuming the file is saved as `example.jsonl` in the repository root.

```json
{
  "group_id": "evaluation-general-2b",
  "request": {
    "state": {
      "messages": [{
        "role": "user",
        "content": [
          {"type": "text", "text": "Compare the accuracy and the average latency per question of the 2B models on General in the figure."},
          {"type": "image_url", "image_url": {"url": "assets/figures/evaluation-results.png"}}
        ]
      }]
    },
    "questions": {
      "best_2b": {
        "type": "choice",
        "instructions": "On General, among the 2B models with average latency below 200 ms per question, which has the highest accuracy?",
        "criteria": {
          "qwen": "Qwen3.5-2B",
          "valen": "Valen-preview"
        }
      }
    }
  },
  "targets": {
    "best_2b": {"probabilities": {"qwen": 0.0, "valen": 1.0}}
  }
}
```
</details>

<a id="contributions"></a>

## 🤝 Contributions

Contributions to Valen are welcome. Open an [issue](https://github.com/Liuziyu77/Valen/issues) to report a problem, share a use case or discuss experimental results. Submit a [pull request](https://github.com/Liuziyu77/Valen/pulls) to improve the code or documentation, contribute training data or add evaluation tasks.

Scan the QR code below to join the Valen WeChat group, discuss the project and share your experiments.

<p align="center">
  <img src="assets/figures/wechat_1013.png" alt="QR code for the Valen WeChat discussion group" width="200">
</p>

<a id="license-and-acknowledgments"></a>

## 📄 License and acknowledgments

The code is released under [Apache 2.0](LICENSE). Base models and source datasets retain their respective licenses.

Built on [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B), with the decision interface inspired by [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
