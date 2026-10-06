<p align="center">
  <img src="assets/branding/valen-logo-v5-compact.svg" alt="Valen — 青绿色光圈、前进箭头与大写字标" width="720"><br>
  <img src="assets/branding/valen-slogan.svg" alt="System One Model, now with vision." width="700">
</p>

<p align="center">
  A multimodal decision model inspired by <a href="https://typesafe.ai/blog/introducing-system-one-models-and-jev">Jev</a> — text, images and video in; decision probabilities out.
</p>

<p align="center">
  <a href="https://huggingface.co/Valen-Team"><img src="https://img.shields.io/badge/Hugging_Face-Valen_Team-FFD21E?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Hugging Face 组织：Valen Team"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-80A51B?style=flat&amp;labelColor=555555" alt="许可证：Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-1683BB?style=flat&amp;logo=python&amp;logoColor=FFD43B&amp;labelColor=555555" alt="Python 3.10+"></a>
  <a href="configs/train/"><img src="https://img.shields.io/badge/Training-SFT_%7C_RLCD-8A63B8?style=flat&amp;labelColor=555555" alt="SFT 与实验性 RLCD"></a>
  <br>
  <a href="https://huggingface.co/Valen-Team/Valen-0.8B"><img src="https://img.shields.io/badge/Valen-0.8B-E88B23?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Valen 0.8B"></a>
  <a href="https://huggingface.co/Valen-Team/Valen-2B"><img src="https://img.shields.io/badge/Valen-2B-E88B23?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Valen 2B"></a>
  <a href="https://huggingface.co/Valen-Team/Valen-4B"><img src="https://img.shields.io/badge/Valen-4B-E88B23?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Valen 4B"></a>
  <a href="https://huggingface.co/spaces/yuhangzang/Valen-Preview-0923"><img src="https://img.shields.io/badge/Space-Try_Demo-009C83?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="在线体验 Valen：Hugging Face Spaces"></a>
  <a href="https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k"><img src="https://img.shields.io/badge/Train-General_100k-2185B5?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="训练集：Valen-Training-General-100k"></a>
  <a href="https://huggingface.co/datasets/Valen-Team/VisualDecisionBench"><img src="https://img.shields.io/badge/Eval-VisualDecisionBench-2185B5?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="VisualDecisionBench"></a>
  <a href="https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game"><img src="https://img.shields.io/badge/Eval-Game-2185B5?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="游戏评测集：Valen-Eval-Game"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b><br>
  <a href="#简介">简介</a> · <a href="#news">News</a> · <a href="#演示">演示</a> · <a href="#模型下载">模型与数据集</a> · <a href="#实验结果">评测结果</a> · <a href="#快速开始">快速开始</a>
</p>

<a id="简介"></a>

## ✨ 简介

Valen（万澜）将视觉感知引入 System One 决策。受 [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 启发，它根据任务指令，对文本、图像和视频中的信息进行判断，直接输出给定候选的概率分布，为程序提供结构化的决策接口。仓库提供模型实现、数据处理、SFT 与实验性 RLCD 训练，以及推理和评估命令，支持使用自有数据训练。

<a id="news"></a>

## 📰 News

- **2026-10-06**：发布正式版 **Valen models** 和 [**VisualDecisionBench**](https://huggingface.co/datasets/Valen-Team/VisualDecisionBench) 评测基准。
- **2026-09-23**：**Valen** 仓库开源，同时发布 [**Valen-preview**](https://huggingface.co/Valen-Team/Valen-Preview-0923)。

<a id="演示"></a>

## 🎬 演示

在 [Hugging Face Space](https://huggingface.co/spaces/yuhangzang/Valen-Preview-0923) 在线体验 **Valen-preview**。

<a id="demo-comparison"></a>

### 与 27B 生成模型对比

**Valen-preview** 读取棋盘图像，在每一步选择移动方向。在相同关卡，Valen-preview 用 9 次决策通关，累计决策耗时为 1.13 秒。Qwen3.8-27B-FP8 的 thinking 模式耗时 198.05 秒，no-thinking 模式未通关。

<p align="center">
  <img src="assets/demos/sokoban-model-comparison.gif" alt="Valen-preview 与 Qwen3.8-27B-FP8 的 no-thinking 和 thinking 模式并排对比。" width="1000"><br>
</p>

### 图像模糊与动作置信度

高斯模糊展示 **Valen-preview** 如何应对视觉细节的减少，决策与置信度动态变化的情况。清晰图像上的决策置信度为 91.6%，最强模糊下为 19.2%。

<p align="center">
  <img src="assets/demos/valen-preview-0923-blur-confidence.gif" alt="Valen-preview 在九档高斯图像模糊下的动作概率和决策置信度。" width="1000"><br>
</p>

### 四局并行能力展示

四条成功的 **Valen-preview** 轨迹以 1× 速度并排播放，不做加速。每局需要 7–10 次决策，平均每步 122–128 毫秒，四局均于 1.24 秒内完成。

<p align="center">
  <img src="assets/demos/sokoban-four-game-showcase.gif" alt="四局 Valen-preview 成功轨迹以记录速度并行播放。" width="1000"><br>
</p>

<a id="模型下载"></a>

## 📥 模型与数据集

**Valen** 模型与 **VisualDecisionBench** 数据集可从 Hugging Face 下载。

### Model（模型）

| 模型 | 下载 |
| --- | --- |
| Valen 0.8B | [🤗 Valen-0.8B](https://huggingface.co/Valen-Team/Valen-0.8B) |
| Valen 2B | [🤗 Valen-2B](https://huggingface.co/Valen-Team/Valen-2B) |
| Valen 4B | [🤗 Valen-4B](https://huggingface.co/Valen-Team/Valen-4B) |
| Valen-preview（早期版本） | [🤗 Hugging Face](https://huggingface.co/Valen-Team/Valen-Preview-0923) |

上方演示与下方快速开始使用 **Valen-preview**，该版本需同时下载 **Valen checkpoint** 和 **Qwen3.5-2B Base 模型**。

### Dataset（数据集）

| 数据集 | 用途 | 下载 |
| --- | --- | --- |
| VisualDecisionBench | VisualDecisionBench-Image 与 VisualDecisionBench-Video 评测 | [🤗 Hugging Face](https://huggingface.co/datasets/Valen-Team/VisualDecisionBench) |
| General 100k（早期版本） | 训练 | [🤗 Hugging Face](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) |
| General 5k（早期版本） | 评测 | [🤗 Hugging Face](https://huggingface.co/datasets/Valen-Team/Valen-Eval-General-5k) |
| Sokoban（早期版本） | 训练与评测 | [🤗 Hugging Face](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) |

<a id="实验结果"></a>

## 📊 评测结果 — Valen

**覆盖 0.8B、2B、4B 的视觉决策模型。** Valen 采用 Qwen3.5 + Mixer 和共享 state 推理，在 Million 级别的数据集上进行SFT。以下为 **VisualDecisionBench-Image、VisualDecisionBench-Video、公开 JevBench** 三组评测。

<p align="center">
  <a href="assets/figures/evaluation-20261006/eval_v1.png"><img src="assets/figures/evaluation-20261006/eval_v1.png" alt="Valen VisualDecisionBench-Image 准确率按模型规模排列：0.8B 71.59%、2B 79.18%、4B 83.06%，与发布基线比较。" width="1400"></a>
</p>

<p align="center">
  <a href="assets/figures/evaluation-20261006/video.png"><img src="assets/figures/evaluation-20261006/video.png" alt="Valen VisualDecisionBench-Video 准确率按模型规模排列：0.8B 79.71%、2B 82.85%、4B 87.36%，与发布基线比较。" width="1400"></a>
</p>

<p align="center">
  <a href="assets/figures/evaluation-20261006/jevbench.png"><img src="assets/figures/evaluation-20261006/jevbench.png" alt="Valen 公开 JevBench 准确率：0.8B 73.59%、2B 78.79%、4B 83.12%；InternDecision 4B 为 86.58%。" width="1400"></a>
</p>

早期结果保留在[归档 README](history/README_zh_preview.md)。

<a id="快速开始"></a>

## 🚀 快速开始

先按[环境要求](docs/technical.md#环境要求)安装依赖，再下载 Valen-preview checkpoint 及其对应的 Qwen3.5-2B Base 模型。

```bash
# 下载 Valen-preview checkpoint。
hf download Valen-Team/Valen-Preview-0923 --local-dir models/Valen-Preview-0923

# 下载 Qwen3.5-2B Base 模型。
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B

# 使用自己的配置训练模型。
python -m valen.train \
  --config configs/train/qwen/sft_warmup.json

# 加载下载好的 Valen-preview 进行推理。
python -m valen.inference \
  --checkpoint models/Valen-Preview-0923 \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/predictions.jsonl

# 用同一组合成样本检查评估流程。
python -m valen.evaluate \
  --checkpoint models/Valen-Preview-0923 \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/smoke_eval
```

`data/smoke` 有少量简单题目，仅仅用于测试功能是否正常运行。

<details>
<summary>一条带图片输入的标注记录</summary>

JSONL 每行是一条记录。下例使用仓库里的[评测总览图](assets/figures/evaluation-results.png)，假设保存为仓库根目录的 `example.jsonl`。

```json
{
  "group_id": "evaluation-general-2b",
  "request": {
    "state": {
      "messages": [{
        "role": "user",
        "content": [
          {"type": "text", "text": "比较图中 General 任务上 2B 模型的准确率和平均单题耗时。"},
          {"type": "image_url", "image_url": {"url": "assets/figures/evaluation-results.png"}}
        ]
      }]
    },
    "questions": {
      "best_2b": {
        "type": "choice",
        "instructions": "在 General 任务上，平均单题耗时低于 200 ms 的 2B 模型中，哪个准确率最高？",
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

<a id="参与贡献"></a>

## 🤝 参与贡献

欢迎一起改进 Valen。你可以通过 [Issues](https://github.com/Liuziyu77/Valen/issues) 反馈问题、分享应用场景和实验结果，也可以提交 [Pull Requests](https://github.com/Liuziyu77/Valen/pulls) 改进代码与文档、补充训练数据或评测任务。

欢迎扫描下方二维码加入 Valen 微信群，一起讨论项目、交流使用体验和实验结果。

<p align="center">
  <img src="assets/figures/wechat_1013.png" alt="Valen 微信讨论群二维码" width="200">
</p>

<a id="许可与致谢"></a>

## 📄 许可与致谢

代码使用 [Apache 2.0](LICENSE) 许可。基础模型和来源数据集遵循各自的许可。

模型基于 [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B)，决策接口受 [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 启发。
