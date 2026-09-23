<p align="center">
  <img src="assets/branding/visual-jev-banner.png" alt="Visual-Jev" width="720">
</p>

<h1 align="center">Visual-Jev</h1>

<p align="center">从多模态输入中作出结构化决策，并返回候选概率。</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-0F766E?style=flat-square" alt="许可证：Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-24465C?style=flat-square" alt="Python 3.10+"></a>
  <a href="configs/train/"><img src="https://img.shields.io/badge/Training-SFT_%7C_RLCD-0F766E?style=flat-square" alt="SFT 与实验性 RLCD"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b><br>
  <a href="#环境要求">环境要求</a> · <a href="#快速开始">快速开始</a> · <a href="#训练">训练</a> · <a href="#文档">文档</a>
</p>

Visual-Jev 根据文本、图片和视频，在你提供的候选中作出决策。模型由 Qwen3.5 骨干和共享决策头组成，直接返回概率，无需生成答案 token。仓库提供 SFT、实验性 RLCD、单机多卡训练和评估工具，支持 Qwen3.5-0.8B 与 2B。

这是受 [Jev 和 System One 模型](https://typesafe.ai/blog/introducing-system-one-models-and-jev)启发的独立研究实现，与 TypeSafe 无隶属关系。模型类和响应中的 `model` 字段统一使用 `VisualJev`；Python 包名和命令入口保留 `visionjev`，兼容现有使用方式。

## 输出类型

| 类型 | 用途 | 返回内容 |
| --- | --- | --- |
| **Choice** | 从 1–255 个命名候选中选择 | 所选候选、完整概率分布、confidence |
| **Noul** | 判断条件是否成立 | `true` 的概率 |
| **Score** | 按 2–10 个有序描述评分 | 等级编号期望值、等级概率、legend、confidence |

候选随问题传入，不同任务和候选数量共用决策头。`confidence` 描述分布集中度，不是经过实测的答案正确概率。

<p align="center">
  <img src="assets/figures/readme-architecture.png" alt="Visual-Jev 架构：多模态输入和候选经过 Qwen3.5 骨干、额外的共享决策头和 softmax，得到 Choice、Noul 或 Score 输出。" width="1000">
</p>

Choice 和 Noul 每道题用一个分支计算全部候选；Score 将各等级分别编码，分支间共享权重。结构和输出定义见[模型设计](docs/architecture.md)。

## 环境要求

- Linux、Python 3.10+，以及支持 BF16 的 NVIDIA GPU 和兼容 CUDA 12.4 的驱动。
- PyTorch 2.6.0、torchvision 0.21.0、Transformers 5.4.0、PEFT 0.18.1。
- 完整依赖见 [pyproject.toml](pyproject.toml)，安装脚本会自动安装。

```bash
git clone https://github.com/Liuziyu77/Visual-Jev.git Visual-Jev
cd Visual-Jev
bash scripts/setup/bootstrap.sh
source .venv/bin/activate
```

已有 Conda/虚拟环境，或需要数据工具依赖时，参见[手动安装说明](scripts/README.md#installation)。

## 快速开始

先按[环境要求](#环境要求)安装依赖，并保持该环境处于激活状态。Python 依赖安装完成后，还需单独下载模型权重。

```bash
# 下载并核验固定版本的 Qwen3.5-0.8B。
python scripts/setup/prepare_model.py

# 用随仓库提供的合成样本训练决策头。
python -m visionjev.train \
  --config configs/train/sft_warmup.json

# 加载训练得到的 checkpoint 进行推理。
python -m visionjev.inference \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/predictions.jsonl

# 用同一组合成样本检查评估流程。
python -m visionjev.evaluate \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/smoke_eval
```

合成样本覆盖三种输出及文本、图片、视频，用于检查流程，不能用于衡量模型能力，也不足以训练出实用模型。基础模型需要另行下载；下载脚本目前面向 0.8B，使用 2B 时，将官方 Qwen3.5-2B 快照放在 `models/Qwen3.5-2B`，并修改配置中的 `model_path`。

<details>
<summary>一条最小的带标签记录</summary>

JSONL 每行是一条记录。下例为展开显示的二分类问题，使用硬标签；也支持概率形式的软标签。

```json
{
  "group_id": "traffic-light-001",
  "request": {
    "state": "当前交通灯是绿灯。",
    "questions": {
      "go": {
        "type": "noul",
        "instructions": "规则：只有绿灯可以通行。当前可以通行吗？"
      }
    }
  },
  "targets": {
    "go": {"probabilities": {"true": 1.0, "false": 0.0}}
  },
  "assets": []
}
```

`targets` 用于训练和评估，不进入模型输入。媒体路径相对于 JSONL 文件所在目录。完整示例见[合成数据](data/smoke/)，字段说明见[数据文档](docs/data-format.md)。

</details>

## 训练

`method` 决定训练目标，`stage` 决定更新哪些参数。四种 stage 是可选配置，不要求每次按顺序全部执行。

<p align="center">
  <img src="assets/figures/readme-training-workflow.png" alt="先用 SFT 训练，可选择从其 checkpoint 初始化 RLCD，再用独立测试集评估。" width="1000">
</p>

SFT 和 RLCD 均使用带标签的训练数据。两种方法的 checkpoint 都可在独立测试集上评估，加载时还需对应的基础模型。

| Stage | 更新参数 | SFT | RLCD |
| --- | --- | --- | --- |
| `warmup` | 决策头 | [配置](configs/train/sft_warmup.json) | [配置](configs/train/rlcd_warmup.json) |
| `text` | 语言 LoRA、决策头；使用纯文本数据 | [配置](configs/train/sft_text.json) | [配置](configs/train/rlcd_text.json) |
| `joint` | 语言 LoRA、视觉 merger、决策头 | [配置](configs/train/sft_joint.json) | [配置](configs/train/rlcd_joint.json) |
| `vision_top` | `joint` 加视觉编码器最后 4 层 | [配置](configs/train/sft_vision_top.json) | [配置](configs/train/rlcd_vision_top.json) |

正式训练前设置配置中的 `model_path`、`data`、`output`、`epochs` 和 `max_steps`。stage 示例默认使用合成数据和短程步数限制。`tokens_per_step` 是**每张 GPU** 的输入预算。

```bash
# 单机八卡；每次独立实验使用新的输出目录。
VJ_GPUS=8 bash scripts/train/launch_sft.sh configs/train/sft_warmup.json

# RLCD 从已有决策头 checkpoint 初始化。
VJ_GPUS=8 bash scripts/train/launch_sft.sh configs/train/rlcd_warmup.json \
  --initialize output/sft_warmup/latest
```

SFT 使用标签监督；RLCD 使用 GRPO 形式的裁剪目标，奖励同时考虑正确性和置信度误差，另加固定参考策略的 KL 约束和可选的 Brier 损失。本项目的奖励公式属于实验方案，不保证概率校准。公式、初始化和断点恢复命令见[训练说明](visionjev/training/README.md)。

checkpoint 保存可训练参数增量和训练状态，加载时仍需原始基础模型。`launch_sft.sh` 共用于 SFT 和 RLCD，也可通过 `VJ_PYTHON` 显式选择解释器。

## 文档

| 文档 | 内容 |
| --- | --- |
| [模型设计](docs/architecture.md) | 候选评分、媒体输入和输出语义 |
| [训练说明](visionjev/training/README.md) | SFT/RLCD 目标、多卡训练、断点恢复 |
| [脚本说明](scripts/README.md) | 环境、训练、评估和数据工具 |
| [代码目录](visionjev/README.md) | 数据、模型、训练和评估模块 |
| [数据文档](docs/data-format.md) | JSONL 格式、媒体路径和划分约定 |
| [任务评测](evaluation/README.md) | Sokoban 数据生成、完整游戏评测、策略比较和轨迹回放 |

详细文档目前以中文为主；中英文 README 均包含安装和训练流程。CPU 测试命令（未准备基础模型处理器时，相关测试会跳过）：

```bash
python -m pytest -q
```

## 后续计划与贡献

- [ ] 对齐优化器更新预算，在独立验证集上做校准消融。
- [ ] 增加图片退化、UI／文档核验和游戏决策演示。
- [ ] 扩展评测并发布可下载的训练权重。

欢迎提交 issue 和 PR。涉及训练或评估的改动，请提供配置、基础模型 revision、数据划分及验证结果。修改置信度相关行为时，同时报告正确率和概率质量。

## 许可与致谢

代码使用 [Apache 2.0](LICENSE) 许可。基础模型和来源数据集遵循各自的许可。

模型基于 [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B)，决策接口受 [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 启发。实验性策略目标参考 [DeepSeekMath](https://arxiv.org/abs/2402.03300) 提出的 GRPO。
