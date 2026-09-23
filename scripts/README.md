# Visual-Jev 脚本

所有命令在仓库根目录执行，`python` 指已安装项目依赖的 Python 3.10+ 环境。

| 目录 | 保留的工具 |
| --- | --- |
| `setup/` | 安装依赖，下载并核验 Qwen3.5-0.8B |
| `train/` | 单机多卡 SFT/RLCD 启动器 |
| `data/` | 生成随仓库提供的合成样例 |
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

下载脚本使用固定的官方 0.8B revision，核验权重并保存 `visionjev_manifest.json`。2B 需要另行准备官方快照并设置训练配置中的 `model_path`。

`data/smoke/` 已包含合成样例；重新生成可运行 `python scripts/data/make_smoke_data.py`，它会覆盖这些样例。自有数据格式见[数据说明](../docs/data-format.md)。

## 训练与推理

```bash
python -m visionjev.train --config configs/train/sft_warmup.json

VJ_GPUS=4 bash scripts/train/launch_sft.sh configs/train/rlcd_joint.json \
  --initialize output/sft_warmup/latest

python -m visionjev.inference \
  --checkpoint output/rlcd_joint/latest \
  --data data/smoke/train.jsonl \
  --output output/rlcd_joint/predictions.jsonl
```

`launch_sft.sh` 支持两种训练目标；`VJ_GPUS` 指定本机进程数，`VJ_PYTHON` 指定解释器，`CUDA_VISIBLE_DEVICES` 选择 GPU。`tokens_per_step` 是每卡预算。推理输出文件的父目录需已存在。

八份基础配置覆盖 SFT/RLCD × 四种 stage，均使用合成数据和短程设置。正式训练需调整 `data`、`model_path`、`epochs`、`max_steps` 和 `output`。每次独立训练使用新的输出目录；初始化与恢复见[训练指南](../visionjev/training/README.md)。

## 通用评估

```bash
# 合成样例只检查流程；正式评估替换为独立测试集。
python -m visionjev.evaluate \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/smoke_eval

# 本机四卡评估，同一有标签问题只处理一次。
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m visionjev.evaluate \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/smoke_eval_4gpu
```

输出包括逐题预测 `predictions.jsonl`、每卡分片及汇总 `metrics.json`。指标包含硬标签准确率、NLL、Brier，以及适用的二分类 F1、Score 误差和 RPS；按任务、模态、语言及领域汇总。命令始终使用显式 `--data` 参数指定的数据。

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
