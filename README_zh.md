<p align="center">
  <img src="assets/branding/valen-logo-v5-compact.svg" alt="Valen — 青绿色光圈、前进箭头与大写字标" width="720"><br>
  <img src="assets/branding/valen-slogan.svg" alt="System One Model, now with vision." width="700">
</p>

<p align="center">
  受 <a href="https://typesafe.ai/blog/introducing-system-one-models-and-jev">Jev</a> 启发的多模态决策模型——输入文本、图像和视频，输出决策概率。
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

- **2026-10-07**：**更快、更准、更通用。** 发布支持文本、图像和视频的最新多模态 System One 模型 **Valen（[0.8B](https://huggingface.co/Valen-Team/Valen-0.8B) / [2B](https://huggingface.co/Valen-Team/Valen-2B) / [4B](https://huggingface.co/Valen-Team/Valen-4B)）**，以及 [**VisualDecisionBench**](https://huggingface.co/datasets/Valen-Team/VisualDecisionBench) 评测基准。
- **2026-09-30**：**Training&Inference Support:** 支持 Flash Attention 2 训练与推理、多机多卡训练，并提高训练时的 GPU 显存利用率。
- **2026-09-23**：**Valen** 仓库开源，同时发布 [**Valen-preview-0923**](https://huggingface.co/Valen-Team/Valen-Preview-0923) 及其对应的[**训练数据**](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k)。

<a id="演示"></a>

## 🎬 演示

**Valen** 图像与视频决策演示，点击预览观看完整视频。

<p align="center">
  <a href="assets/demos/Valen.mp4"><img src="assets/demos/valen-demo-preview.gif" alt="Valen 演示预览，点击观看完整视频。" width="1000"></a>
</p>

**游戏决策演示**——五条由 Valen 成功完成的游戏轨迹。点击任一预览可观看完整视频。

<table>
  <tr>
    <td colspan="2" width="33%" align="center">
      <a href="assets/demos/games/valen-sokoban.mp4"><img src="assets/demos/games/valen-sokoban-preview.gif" alt="Valen 完成推箱子关卡。" width="300"></a><br>
      <b>推箱子</b><br><sub>规划 13 步最优解。</sub>
    </td>
    <td colspan="2" width="33%" align="center">
      <a href="assets/demos/games/valen-frozen-lake.mp4"><img src="assets/demos/games/valen-frozen-lake-preview.gif" alt="Valen 完成冰湖导航。" width="300"></a><br>
      <b>冰湖</b><br><sub>规划 11 步最优路线，绕开危险区域。</sub>
    </td>
    <td colspan="2" width="33%" align="center">
      <a href="assets/demos/games/valen-maze.mp4"><img src="assets/demos/games/valen-maze-preview.gif" alt="Valen 完成迷宫关卡。" width="300"></a><br>
      <b>迷宫</b><br><sub>完成 22 步长程路线规划。</sub>
    </td>
  </tr>
  <tr>
    <td colspan="3" width="50%" align="center">
      <a href="assets/demos/games/valen-lane-racer.mp4"><img src="assets/demos/games/valen-lane-racer-preview.gif" alt="Valen 完成车道竞速关卡。" width="300"></a><br>
      <b>车道竞速</b><br><sub>连续变道避障，10 步完成赛道。</sub>
    </td>
    <td colspan="3" width="50%" align="center">
      <a href="assets/demos/games/valen-parkour.mp4"><img src="assets/demos/games/valen-parkour-preview.gif" alt="Valen 完成跑酷关卡。" width="300"></a><br>
      <b>跑酷</b><br><sub>组合 14 步跳跃、冲刺、下蹲与奔跑动作。</sub>
    </td>
  </tr>
</table>

<a id="模型下载"></a>

## 📥 模型与数据集

| 模型 | 下载 |
| --- | --- |
| Valen 0.8B | [🤗 Valen-0.8B](https://huggingface.co/Valen-Team/Valen-0.8B) |
| Valen 2B | [🤗 Valen-2B](https://huggingface.co/Valen-Team/Valen-2B) |
| Valen 4B | [🤗 Valen-4B](https://huggingface.co/Valen-Team/Valen-4B) |
| Valen-preview-0923（早期版本） | [🤗 Hugging Face](https://huggingface.co/Valen-Team/Valen-Preview-0923) |

[VisualDecisionBench](https://huggingface.co/datasets/Valen-Team/VisualDecisionBench) 包含 Image 和 Video 两个子集，用于图像与视频决策评测。此前发布的 [General 100k](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) 用于训练，[General 5k](https://huggingface.co/datasets/Valen-Team/Valen-Eval-General-5k) 用于评测，[Sokoban](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) 用于游戏训练与评测。

<a id="实验结果"></a>

## 📊 评测结果 — Valen

**覆盖 0.8B、2B、4B 的视觉决策模型。** Valen 采用 Qwen3.5 + Mixer 和共享 state 推理，在百万级样本上进行 SFT。以下在 **VisualDecisionBench-Image、VisualDecisionBench-Video、公开 JevBench** 三组评测中，与已发布的决策模型进行对比。

<p align="center">
  <a href="assets/figures/evaluation-20261006/eval_v1.png"><img src="assets/figures/evaluation-20261006/eval_v1.png" alt="Valen VisualDecisionBench-Image 准确率按模型规模排列：0.8B 75.94%、2B 80.41%、4B 83.06%，与发布基线比较。" width="1400"></a>
</p>

<p align="center">
  <a href="assets/figures/evaluation-20261006/video.png"><img src="assets/figures/evaluation-20261006/video.png" alt="Valen VisualDecisionBench-Video 准确率按模型规模排列：0.8B 79.03%、2B 82.85%、4B 87.36%，与发布基线比较。" width="1400"></a>
</p>

<p align="center">
  <a href="assets/figures/evaluation-20261006/jevbench.png"><img src="assets/figures/evaluation-20261006/jevbench.png" alt="Valen 公开 JevBench 准确率：0.8B 77.92%、2B 83.55%、4B 83.12%；InternDecision 4B 为 86.58%。" width="1400"></a>
</p>

早期结果保留在[归档 README](history/README_zh_preview.md)。

<a id="快速开始"></a>

## 🚀 快速开始

先按[安装说明](scripts/README.md#installation)准备依赖；Hugging Face 仓库需要访问权限时，先运行 `hf auth login`。

### 1. 推理

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Valen-Team/Valen-2B", trust_remote_code=True, dtype="auto",
    attn_implementation="sdpa",
).to("cuda").eval()
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False

print(model.predict({
    "state": "A cat is on the sofa.",
    "questions": {
        "animal": {"type": "choice", "instructions": "Which animal is present?",
                   "criteria": {"cat": "A cat", "dog": "A dog"}},
        "on_sofa": {"type": "noul", "instructions": "The cat is on the sofa."},
    },
}, execution="shared_state"))
```

逐题推理使用 `execution="question"`；安装 Flash Attention 后可设置 `attn_implementation="flash_attention_2"`。图片和视频输入见[数据格式](docs/data-format.md)。

### 2. 评估 VisualDecisionBench

下载并解压媒体，然后运行[评估脚本](evaluation/visualdecisionbench/evaluate.py)：

```bash
hf download Valen-Team/VisualDecisionBench --repo-type dataset --local-dir data/VisualDecisionBench
python data/VisualDecisionBench/unpack_assets.py
python -m evaluation.visualdecisionbench.evaluate \
  --model Valen-Team/Valen-2B --data data/VisualDecisionBench \
  --output output/visualdecisionbench
```

添加 `--lora --revision lora-ckpt` 可评估旧 LoRA 模型。输出 `predictions.jsonl` 和 `metrics.json`，包含图像/视频、Choice/Noul/Score 分项与耗时。视频采样 16 帧；准确率仅统计硬标签题，软标签 Score 题参与概率指标计算。

### 3. 训练

共享 state 配方使用 Mixer。请安装 Flash Attention 2，或将 `model.attn_implementation` 设置为 `sdpa`。先在[配置](configs/train/qwen/)中设置数据路径、输出目录和训练预算；默认使用 smoke 数据，最多训练 100 步。

```bash
python scripts/setup/prepare_model.py

# 两阶段 SFT：先预热决策头，再联合训练 LoRA、Mixer 和视觉 merger。
python -m valen.train --config configs/train/qwen/sft_shared_state_warmup.json
python -m valen.train --config configs/train/qwen/sft_shared_state_joint.json \
  --initialize output/qwen-shared-state/warmup/latest

# RLCD：从 SFT checkpoint 开始，对候选动作进行 RL 优化。
python -m valen.train --config configs/train/qwen/rlcd_shared_state_joint.json \
  --initialize output/qwen-shared-state/joint/latest
```

切换为 **full training** 时保留预热阶段，将联合 SFT 的配置替换为 [`sft_shared_state_full_joint.json`](configs/train/qwen/sft_shared_state_full_joint.json)，继续使用相同的 `--initialize` checkpoint。该配方设置 `model.finetuning_type="full"`，更新语言主干、视觉主干和 Mixer；设为 `"lora"` 则使用 LoRA。两个阶段的模型与数据路径需保持一致。

```bash
python -m valen.train --config configs/train/qwen/sft_shared_state_full_joint.json \
  --initialize output/qwen-shared-state/warmup/latest
```

多卡使用 `VALEN_GPUS=8 bash scripts/train/launch.sh <config> [--initialize <checkpoint>]`。奖励设置与 checkpoint 恢复见[训练指南](valen/training/README.md)。

<a id="参与贡献"></a>

## 🤝 参与贡献

欢迎一起改进 Valen。你可以通过 [Issues](https://github.com/Liuziyu77/Valen/issues) 反馈问题、分享应用场景和实验结果，也可以提交 [Pull Requests](https://github.com/Liuziyu77/Valen/pulls) 改进代码与文档、补充训练数据或评测任务。积极为 Valen 作出贡献的开发者有机会加入核心开发团队。

欢迎扫描下方二维码加入 Valen 微信群，一起讨论项目、交流使用体验和实验结果。

<p align="center">
  <img src="assets/figures/wechat_1013.jpg" alt="Valen 微信讨论群二维码" width="200">
</p>

<a id="许可与致谢"></a>

## 📄 许可与致谢

代码使用 [Apache 2.0](LICENSE) 许可。基础模型和来源数据集遵循各自的许可。

模型基于 [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B)，决策接口受 [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 启发。
