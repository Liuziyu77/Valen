# Valen 技术说明

[English README](../README.md) · [中文 README](../README_zh.md)

本文收录模型架构、输出协议、训练流程和完整实验结果。所有命令均在仓库根目录执行。

[输出类型](#输出类型) · [模型架构](#模型架构) · [环境要求](#环境要求) · [训练](#训练) · [完整实验](#完整实验) · [演示测量说明](#演示测量说明)

## 输出类型

| 类型 | 用途 | 返回内容 |
| --- | --- | --- |
| **Choice** | 从 1–255 个候选中选择 | 所选候选、完整概率分布、confidence |
| **Noul** | 判断条件是否成立 | `true` 的概率 |
| **Score** | 按 2–10 个有序描述评分 | 等级编号期望值、等级概率、legend、confidence |

候选随问题传入，不同任务和候选数量共用决策头。`confidence` 描述分布集中度。

## 模型架构

<p align="center">
  <img src="../assets/figures/readme-architecture.png" alt="Valen 架构：多模态输入和候选经过 Qwen3.5 骨干、额外的共享决策头和 softmax，得到 Choice、Noul 或 Score 输出。" width="1000">
</p>

Choice 和 Noul 每道题用一个分支计算全部候选；Score 每个等级单独执行backbone forward pass，再将 logits 一起归一化。公共 state 只编码一次，但在backbone的每个分支中重复计算。详见[架构说明](../docs/architecture.md)和[响应字段与公式](../docs/evaluation.md)。

<a id="环境要求"></a>

## 环境要求

- Linux、Python 3.10+，NVIDIA GPU。
- PyTorch 2.6.0、torchvision 0.21.0、Transformers 5.4.0、PEFT 0.18.1。
- 完整依赖见 [pyproject.toml](../pyproject.toml)，安装脚本会自动安装。

```bash
git clone https://github.com/Liuziyu77/Valen.git Valen
cd Valen
bash scripts/setup/bootstrap.sh
source .venv/bin/activate
```

已有 Conda 或 virtualenv 环境时，参见[手动安装说明](../scripts/README.md#installation)。安装脚本创建 `.venv`，模型权重需要另行下载。

## 本地训练与功能检查

先按[环境要求](#环境要求)安装依赖。依赖安装完成后，需下载模型权重。

```bash
# 下载 Qwen3.5-2B Base 模型。
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B

# 用随仓库提供的合成样本训练决策头。
python -m valen.train \
  --config configs/train/qwen/sft_warmup.json

# 加载训练得到的 checkpoint 进行推理。
python -m valen.inference \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/predictions.jsonl

# 用同一组合成样本检查评估流程。
python -m valen.evaluate \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/smoke_eval
```

`data/smoke` 有少量简单题目，仅仅用于测试功能是否正常运行。

<details>
<summary>一条带图片输入的标注记录</summary>

JSONL 每行是一条记录。下例使用仓库里的[评测总览图](../assets/figures/evaluation-results.png)，假设保存为仓库根目录的 `example.jsonl`。

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
          "valen": "Valen-Preview-0923"
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

<a id="训练"></a>

## 训练

当前 Valen 支持 `SFT` 和 `RLCD` 两种训练 `method`，`stage` 决定更新哪些参数，当前支持四种 stage 配置。

<p align="center">
  <img src="../assets/figures/readme-training-workflow.png" alt="先用 SFT 训练，可选择从其 checkpoint 初始化 RLCD，再用独立测试集评估。" width="1000">
</p>

SFT 和 RLCD 均使用带标签的训练数据。两种方法的 checkpoint 都可在独立测试集上评估，加载时还需对应的base model（目前实现的是LoRA，后续可考虑全量微调）。

| Stage | 更新参数 | SFT | RLCD |
| --- | --- | --- | --- |
| `warmup` | 决策头 | [配置](../configs/train/qwen/sft_warmup.json) | [配置](../configs/train/qwen/rlcd_warmup.json) |
| `text` | LLM LoRA、决策头；使用纯文本数据 | [配置](../configs/train/qwen/sft_text.json) | [配置](../configs/train/qwen/rlcd_text.json) |
| `joint` | LLM LoRA、VIT merger、决策头 | [配置](../configs/train/qwen/sft_joint.json) | [配置](../configs/train/qwen/rlcd_joint.json) |
| `vision_top` | `joint` 加视觉编码器最后 4 层 | [配置](../configs/train/qwen/sft_vision_top.json) | [配置](../configs/train/qwen/rlcd_vision_top.json) |

正式训练前复制一份配置，设置 `model_path`、`data`、`output`、`epochs` 和 `max_steps`。`tokens_per_step` 是**每张 GPU** 的预算，按 state 内全部问题分支的 token 总量计算。每卡持有完整模型，采用**数据并行**。默认值与修改示例见[配置参考](../docs/configuration.md)。

```bash
# 单机八卡；每次独立实验使用新的输出目录。
VALEN_GPUS=8 bash scripts/train/launch.sh configs/train/qwen/sft_warmup.json

# RLCD 从已训练的 SFT 决策头 checkpoint 初始化。
VALEN_GPUS=8 bash scripts/train/launch.sh configs/train/qwen/rlcd_warmup.json \
  --initialize output/sft_warmup/latest
```

目前 SFT 使用标签监督；RLCD 使用 GRPO 形式，奖励同时考虑正确性和置信度误差，另加固定参考策略的 KL 约束和可选的 Brier 损失。公式、初始化和断点恢复命令见[训练说明](../valen/training/README.md)。

checkpoint 将可训练参数的值和训练状态保存在 `<output>/latest/`，冻结的base模型另行加载。每次保存会覆盖 `latest/`，不保留历史版本。`--initialize` 从已有权重开始新实验；`--resume` 恢复优化器、数据游标和各 rank 状态，要求进程数不变。`launch.sh` 共用于 SFT 和 RLCD，可通过 `VALEN_PYTHON` 指定解释器。

<a id="完整实验"></a>

## 完整实验

表中的 `Valen-Base-*` 为通用任务模型，`Valen-Sokoban-*` 为推箱子任务模型；RLCD 为本仓库的强化学习训练方法。两组实验使用各自任务训练后的 checkpoint。

### Model Card

我们对模型的训练方法，训练数据进行了消融实验：

| 模型 | 训练方法 | 数据 | 数据集 |
| --- | --- | --- | --- |
| Valen-Base-SFT-0.8B | SFT | General SFT 100k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) |
| Valen-Base-SFT-2B | SFT | General SFT 100k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) |
| Valen-Base-RLCD-0.8B | SFT + RLCD | General SFT 70k + General RLCD 30k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) |
| Valen-Base-RLCD-2B | SFT + RLCD | General SFT 70k + General RLCD 30k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) |
| Valen-Sokoban-SFT-2B | SFT + SFT | General SFT 100k + Sokoban SFT 30k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k)&nbsp;[![Sokoban](https://img.shields.io/badge/Sokoban-8A63B8?style=flat&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) |
| Valen-Sokoban-RLCD-2B | SFT + RLCD | General SFT 100k + Sokoban RLCD 30k | [![General 100k](https://img.shields.io/badge/General-100k-2185B5?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k)&nbsp;[![Sokoban](https://img.shields.io/badge/Sokoban-8A63B8?style=flat&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) |

Sokoban 的训练集和评测集均包含在 [Valen-Eval-Game](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) 中，训练与评测分别使用对应划分。

#### Training Loss

<p align="center">
  <a href="../assets/figures/training/sft100k_loss.png"><img src="../assets/figures/training/sft100k_loss.png" alt="Valen-Base-SFT-0.8B 和 2B 在 General 100k 数据上的训练损失曲线。" width="1000"></a><br>
  <sub>General SFT：0.8B 与 2B 的训练 loss。</sub>
</p>

<p align="center">
  <a href="../assets/figures/training/sokoban_rlcd_loss_reward.png"><img src="../assets/figures/training/sokoban_rlcd_loss_reward.png" alt="Valen-Sokoban-RLCD 的 0.8B 和 2B 模型在训练过程中的 loss 与平均 reward。" width="1000"></a><br>
  <sub>Sokoban RLCD：0.8B 与 2B 的训练 loss 和 reward。</sub>
</p>
