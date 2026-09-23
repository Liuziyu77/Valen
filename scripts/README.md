# Valen 脚本

所有命令在仓库根目录执行，`python` 指已安装项目依赖的 Python 3.10+ 环境。

| 目录 | 工具 |
| --- | --- |
| `setup/` | 安装依赖，下载并核验 Qwen3.5-0.8B |
| `train/` | 单机多卡 SFT/RLCD 启动器 |
| `data/` | 生成随仓库提供的合成样例 |
| `eval/` | 从 Eval_3 结果快照重绘 README 中的实验图 |
| `smoke/` | 单卡、双卡训练与恢复集成检查 |

<a id="installation"></a>

## 安装

```bash
bash scripts/setup/bootstrap.sh
source .venv/bin/activate
```

已有 Conda 或 virtualenv 环境时，激活后手动安装：

```bash
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
python -m pip install -e '.[test]'
```

训练配置使用 BF16，需要支持 BF16 的 NVIDIA GPU。`bootstrap.sh` 默认用 `python3` 创建 `.venv`，可用 `VJ_PYTHON` 指定解释器。

## 模型与数据

```bash
python scripts/setup/prepare_model.py
```

下载脚本固定使用 `Qwen/Qwen3.5-0.8B`，默认 revision 为 `2fc06364715b967f1860aea9cf38778875588b17`。它下载快照，逐个检查 `.safetensors` 权重与 Hub 上的大小、摘要，加载本地 config/processor，然后生成 `valen_manifest.json`，记录 revision、依赖版本、权重摘要和媒体默认配置。

| 参数 | 默认值 / 用途 |
| --- | --- |
| `--output` | 默认 `models/Qwen3.5-0.8B`；自定义后需同步修改训练配置 |
| `--revision` | 指定要解析并下载的 revision |
| `--verify-local` | 核验输出目录内已有文件并重写清单，不下载权重；仍需访问 Hub 获取元数据 |

脚本没有 `--repo` 参数，不能通过修改输出目录下载 2B。2B 需另行准备包含权重、config、tokenizer 和媒体处理器配置的完整快照，再设置训练配置中的 `model_path`。下载完成后，训练使用本地文件加载，不会补齐缺失权重。

`data/smoke/` 已包含合成样例；重新生成可运行 `python scripts/data/make_smoke_data.py`，它会覆盖这些样例。自有数据格式见[数据说明](../docs/data-format.md)。

## 训练与推理

```bash
python -m valen.train --config configs/train/sft_warmup.json

VJ_GPUS=4 bash scripts/train/launch_sft.sh configs/train/rlcd_joint.json \
  --initialize output/sft_warmup/latest

python -m valen.inference \
  --checkpoint output/rlcd_joint/latest \
  --data data/smoke/train.jsonl \
  --output output/rlcd_joint/predictions.jsonl
```

`launch_sft.sh` 支持两种训练目标；`VJ_GPUS` 指定本机进程数（默认 2），`VJ_PYTHON` 指定解释器（默认 `python`），`CUDA_VISIBLE_DEVICES` 选择 GPU。不传配置时使用 `sft_joint.json`；首个配置参数之后的参数原样传给训练 CLI。`tokens_per_step` 是每卡预算。推理输出文件的父目录需已存在。

八份基础配置覆盖 SFT/RLCD × 四种 stage，均使用合成数据和短程设置。正式训练需调整 `data`、`model_path`、`epochs`、`max_steps` 和 `output`。每次独立训练使用新的输出目录；初始化与恢复见[训练指南](../valen/training/README.md)。

## 命令参数

| 命令 | 必填 | 可选 |
| --- | --- | --- |
| `python -m valen.train` | `--config` | `--method {sft,rlcd}`、`--output`、互斥的 `--initialize` / `--resume` |
| `python -m valen.inference` | `--checkpoint`、`--data`、`--output`（文件） | `--device`（默认 `cuda`）、`--calibration` |
| `python -m valen.evaluate` | `--checkpoint`、`--data`、`--output`（目录） | `--device`（默认 `cuda`） |

训练设备、学习率、数据路径等在 JSON 中设置，不能直接使用 `train --device` 或 `train --data`。详细字段见[配置参考](../docs/configuration.md)。各入口可通过 `--help` 查看参数，需要先安装 Python 依赖。

## 通用评估

```bash
# 合成样例只检查流程；正式评估替换为独立测试集。
python -m valen.evaluate \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/smoke_eval

# 本机四卡评估，同一有标签问题只处理一次。
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m valen.evaluate \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/smoke_eval_4gpu
```

输出包括逐题预测 `predictions.jsonl`、每卡分片及汇总 `metrics.json`。指标包含硬标签准确率、NLL、Brier，以及适用的二分类 F1、Score 误差和 RPS；按任务、模态、语言及领域汇总。命令始终使用显式 `--data` 参数指定的数据。评估固定使用温度 1，没有 `--calibration` 参数；指标公式和时间统计的范围见[推理与评估](../docs/evaluation.md)。

## 实验图

`scripts/eval/plot_eval3.py` 读取 `assets/figures/eval3/results.json`，输出四张图的 PNG、SVG 和 PDF。需要另行安装 Matplotlib，不需要模型、GPU 或原始评测目录。

```bash
python -m pip install matplotlib
python scripts/eval/plot_eval3.py --output-dir /tmp/valen-eval3-preview
```

省略 `--output-dir` 时覆盖 `assets/figures/eval3/` 中的配图。实验设置和结果见[Eval_3 报告](../docs/experiments/eval3.md)。

## 验证

```bash
python -m pytest -q

# 以下检查需要已下载 0.8B 模型和相应数量的 GPU。
python scripts/smoke/gpu_smoke.py
python -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node=2 --max_restarts=0 scripts/smoke/multi_gpu_smoke.py
python -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node=2 --max_restarts=0 scripts/smoke/rlcd_gpu_smoke.py
```

CPU 测试不需要模型权重；与真实处理器相关的测试在没有本地处理器文件时会跳过。GPU 检查实际执行短程训练并验证恢复结果，产物写入 `artifacts/` 或 `output/`，不衡量模型质量。

| 脚本 | GPU 数 | 主要检查 | 结果文件 |
| --- | --- | --- | --- |
| `gpu_smoke.py` | 1 | 基础 embedding 加载、三类输入/输出、参数组梯度、保存重载、stage 初始化 | `artifacts/gpu_smoke/result.json` |
| `multi_gpu_smoke.py` | 必须为 2 | SFT 梯度同步、不等分片、空闲 rank、恢复与连续训练一致 | `artifacts/multi_gpu_smoke/result.json` |
| `rlcd_gpu_smoke.py` | 必须为 2 | RLCD 更新、固定参考策略、三类决策、恢复与连续训练一致 | `output/rlcd_gpu_smoke/result.json` |

这些脚本使用固定的模型路径和输出目录，重新运行会覆盖对应产物，不接受训练 CLI 的参数。CPU 测试的文件与覆盖范围见[代码目录](../valen/README.md#测试对应关系)。

## 常见问题

| 报错或现象 | 检查与处理 |
| --- | --- |
| 找不到模型、处理器，或 `Incomplete pretrained model load` | 确认下载目录完整、`model_path` 指向该目录、依赖与 `pyproject.toml` 一致；加载器会拒绝缺失或不匹配的权重 |
| `Media hash mismatch` / 媒体文件不存在 | 路径按 JSONL 所在目录解析；确认文件内容与 `assets` 记录对应，替换素材后同步更新清单 |
| `tokens exceed max_length` | 某个分支过长；缩短 state/候选描述，或调整媒体处理参数与长度上限 |
| `State ... exceeds tokens_per_step` | 完整 state 的所有分支超过每卡预算；适当增大预算或减小单条记录的分支计算量，程序不会拆开 state |
| CUDA 显存不足 | 每卡加载完整模型，增加 GPU 数不拆分模型；检查 stage、媒体长度和 RLCD 参考模型。RLCD warmup 可用 `cache_frozen_features=true` 复用特征 |
| `Text stage requires a text-only dataset` | `text` stage 收到了有标签的媒体题；使用纯文本数据，或为多模态训练选择 `joint` |
| `Initialization would freeze previously changed parameters` | 新 stage 没有包含原 checkpoint 的全部可训练参数；例如 joint 不能初始化到 warmup |
| `Resume config/data changed` / world size 不匹配 | 保持原配置、数据和进程数；更换实验条件使用 `--initialize` 和新输出目录 |
| 恢复后没有新增 step | 已到 `epochs` 或 `max_steps` 上限；在配置副本中提高已用尽的上限 |
| `No optimizer step` | 数据没有有效标签，或 `epochs` / `max_steps` 为 0 |
| 图片和视频指标都归到 `media` | 在记录中填写 `meta.modality`；缺省时评估不区分图片与视频 |
