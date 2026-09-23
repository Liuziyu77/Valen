<p align="center">
  <img src="assets/branding/visual-jev-banner.png" alt="Visual-Jev" width="720">
</p>

<h1 align="center">Visual-Jev</h1>

<p align="center">根据文本、图片和视频，对给定候选评分。</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-0F766E?style=flat-square" alt="许可证：Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-24465C?style=flat-square" alt="Python 3.10+"></a>
  <a href="configs/train/"><img src="https://img.shields.io/badge/Training-SFT_%7C_RLCD-0F766E?style=flat-square" alt="SFT 与实验性 RLCD"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b><br>
  <a href="#环境要求">环境要求</a> · <a href="#快速开始">快速开始</a> · <a href="#训练">训练</a> · <a href="#文档">文档</a>
</p>

Visual-Jev 可以根据文本、图片或视频，对提供的候选评分。模型由 Qwen3.5 backbone和共享decision head组成，直接返回概率分布，无需生成答案 token。仓库包含模型实现、JSONL 数据处理、SFT 与实验性 RLCD 训练，以及推理和评估命令。现有配置使用 Qwen3.5-0.8B/2B 进行训练。

本项目受 [Jev 和 System One 模型](https://typesafe.ai/blog/introducing-system-one-models-and-jev)启发。

## 输出类型

| 类型 | 用途 | 返回内容 |
| --- | --- | --- |
| **Choice** | 从 1–255 个候选中选择 | 所选候选、完整概率分布、confidence |
| **Noul** | 判断条件是否成立 | `true` 的概率 |
| **Score** | 按 2–10 个有序描述评分 | 等级编号期望值、等级概率、legend、confidence |

候选随问题传入，不同任务和候选数量共用决策头。`confidence` 描述分布集中度。

<p align="center">
  <img src="assets/figures/readme-architecture.png" alt="Visual-Jev 架构：多模态输入和候选经过 Qwen3.5 骨干、额外的共享决策头和 softmax，得到 Choice、Noul 或 Score 输出。" width="1000">
</p>

Choice 和 Noul 每道题用一个分支计算全部候选；Score 每个等级单独执行backbone forward pass，再将 logits 一起归一化。公共 state 只编码一次，但在backbone的每个分支中重复计算。详见[架构说明](docs/architecture.md)和[响应字段与公式](docs/evaluation.md)。

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

已有 Conda 或 virtualenv 环境时，参见[手动安装说明](scripts/README.md#installation)。安装脚本创建 `.venv`，模型权重需要另行下载。

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

合成集有 6 条记录、15 道题，其中 13 道有标签。推理生成 6 行响应；评估生成 13 行逐题预测和 `metrics.json`。这些样本用于检查流程。

命令均在仓库根目录执行。配置中的路径相对于当前工作目录，JSONL 中的媒体路径相对于该 JSONL 文件

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

`targets` 用于训练和评估，不进入模型输入。推理可以省略标签，但仍需 `group_id`。Noul 答案形如 `{"type": "noul", "noul": 0.8}`，其中数值为 `P(true)`。Choice、Score 及媒体记录的示例见[合成数据](data/smoke/)和[数据格式](docs/data-format.md)。

</details>

## 训练

`method` 决定训练目标，`stage` 决定更新哪些参数。可选四种 stage 是配置。

<p align="center">
  <img src="assets/figures/readme-training-workflow.png" alt="先用 SFT 训练，可选择从其 checkpoint 初始化 RLCD，再用独立测试集评估。" width="1000">
</p>

SFT 和 RLCD 均使用带标签的训练数据。两种方法的 checkpoint 都可在独立测试集上评估，加载时还需对应的base model。

| Stage | 更新参数 | SFT | RLCD |
| --- | --- | --- | --- |
| `warmup` | 决策头 | [配置](configs/train/sft_warmup.json) | [配置](configs/train/rlcd_warmup.json) |
| `text` | 语言 LoRA、决策头；使用纯文本数据 | [配置](configs/train/sft_text.json) | [配置](configs/train/rlcd_text.json) |
| `joint` | 语言 LoRA、视觉 merger、决策头 | [配置](configs/train/sft_joint.json) | [配置](configs/train/rlcd_joint.json) |
| `vision_top` | `joint` 加视觉编码器最后 4 层 | [配置](configs/train/sft_vision_top.json) | [配置](configs/train/rlcd_vision_top.json) |

正式训练前复制一份配置，设置 `model_path`、`data`、`output`、`epochs` 和 `max_steps`。`tokens_per_step` 是**每张 GPU** 的预算，按 state 内全部问题分支的 token 总量计算。每卡持有完整模型，采用数据并行。默认值与修改示例见[配置参考](docs/configuration.md)。

```bash
# 单机八卡；每次独立实验使用新的输出目录。
VJ_GPUS=8 bash scripts/train/launch_sft.sh configs/train/sft_warmup.json

# RLCD 从已有决策头 checkpoint 初始化。
VJ_GPUS=8 bash scripts/train/launch_sft.sh configs/train/rlcd_warmup.json \
  --initialize output/sft_warmup/latest
```

SFT 使用标签监督；RLCD 使用 GRPO 形式的裁剪目标，奖励同时考虑正确性和置信度误差，另加固定参考策略的 KL 约束和可选的 Brier 损失。本项目的奖励公式属于实验方案，不保证概率校准。公式、初始化和断点恢复命令见[训练说明](visionjev/training/README.md)。

checkpoint 将可训练参数的值和训练状态保存在 `<output>/latest/`，冻结的基础模型另行加载。每次保存会覆盖 `latest/`，不保留历史版本。`--initialize` 从已有权重开始新实验；`--resume` 恢复优化器、数据游标和各 rank 随机状态，要求进程数不变。`launch_sft.sh` 共用于 SFT 和 RLCD，可通过 `VJ_PYTHON` 指定解释器。

## 仓库结构

```text
visionjev/
  data/          记录校验、媒体读取、候选编译
  modeling/      Qwen3.5 骨干、决策头、可训练参数组
  training/      SFT/RLCD 目标、共用循环、梯度同步、checkpoint
  evaluation/    推理响应、指标、多卡评估
evaluation/      任务环境、Sokoban 数据生成与完整游戏评测
configs/train/   两种训练目标 × 四种 stage，共八份示例
scripts/         环境安装、模型准备、训练启动器、GPU 检查
data/smoke/      文本、图片、视频合成样例
tests/           CPU 测试，包含双进程训练与恢复检查
docs/            架构、数据、配置、评估说明
```

顶层 `visionjev/train.py`、`inference.py`、`evaluate.py` 将命令转发到相应子包。[代码目录](visionjev/README.md)列出了实现入口和对应测试。

## 文档

| 文档 | 内容 |
| --- | --- |
| [架构说明](docs/architecture.md) | 编译流程、候选评分、分支开销、梯度同步 |
| [配置参考](docs/configuration.md) | 默认值、token 预算、优化器和 RLCD 参数 |
| [训练说明](visionjev/training/README.md) | SFT/RLCD 目标、多卡训练、断点恢复 |
| [脚本说明](scripts/README.md) | 环境、训练、评估和数据工具 |
| [代码目录](visionjev/README.md) | 数据、模型、训练和评估模块 |
| [数据格式](docs/data-format.md) | 完整样例、标签、本地媒体和数据划分 |
| [推理与评估](docs/evaluation.md) | 响应字段、confidence 公式、指标和计时口径 |
| [任务评测](evaluation/README.md) | Sokoban 数据生成、完整游戏评测、策略比较和轨迹回放 |

详细文档目前以中文为主；中英文 README 均包含安装和训练流程。CPU 测试命令（未准备基础模型处理器时，相关测试会跳过）：

```bash
python -m pytest -q
```

## 开发与验证

当前工具通过文件和命令行运行，尚未提供 HTTP 服务、按验证集选取 checkpoint 或温度拟合命令。通用 JSONL 读取器不自动划分数据；Sokoban 生成器单独构建隔离的训练集和测试集。推理支持读取温度文件，评估 CLI 固定使用温度 1。GPU 集成检查和常见错误处理见[脚本说明](scripts/README.md)。

涉及训练或评估的改动，请提供配置、基础模型 revision、数据划分及相关测试结果。修改置信度行为时，同时报告正确率、NLL 和 Brier。比较 SFT 与 RLCD 时还需对齐实际优化次数：默认 RLCD 每批更新两次，SFT 每批一次。

## 许可与致谢

代码使用 [Apache 2.0](LICENSE) 许可。基础模型和来源数据集遵循各自的许可。

模型基于 [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B)，决策接口受 [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 启发。实验性策略目标参考 [DeepSeekMath](https://arxiv.org/abs/2402.03300) 提出的 GRPO。
