# 训练配置参考

`python -m visualjev.train --config <file.json>` 读取一个 JSON 对象。`--method` 和 `--output` 可覆盖对应字段，其他训练参数在 JSON 中修改。配置中的相对路径以**进程工作目录**为基准；以下命令均从仓库根目录执行。

现有八份配置使用 Qwen3.5-0.8B 和合成数据集，运行 3 个 epoch、100 个 step，先达到的上限结束训练。

## 数据、模型和训练budget

下表的默认值是源码在字段缺省时使用的值，示例 JSON 可以覆盖它们。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `model_path` | 必填 | 本地 Qwen3.5 模型及处理器目录 |
| `data` | 必填 | JSONL 文件路径；每卡读取完整文件后按记录分片 |
| `output` | 必填 | 日志及 `latest/` checkpoint 的输出目录 |
| `method` | `"sft"` | `sft` 或 `rlcd` |
| `stage` | `"joint"` | `warmup`、`text`、`joint`、`vision_top` |
| `device` | `"cuda"` | 设备；多卡 CUDA 训练按 `LOCAL_RANK` 绑定 |
| `dtype` | `"bf16"` | backbone精度，可选 `bf16`、`fp32`；决策头保持 FP32 |
| `seed` | `42` | 初始化、记录排序及采样使用的种子 |
| `epochs` | `1` | 最大数据遍历轮数 |
| `max_steps` | `2**63-1` | 最大批次数 |
| `max_length` | `8192` | 单个完整分支的 token 上限，超限报错 |
| `tokens_per_step` | `16384` | 每 rank 每批的 `compute_tokens` 上限，必须为正 |
| `media_kwargs` | `{}` | 传给官方处理器的参数；随 checkpoint 保存 |
| `save_every` | `100` | 每隔多少批覆盖保存 `latest/`；应设为正整数，正常结束也会保存 |

`tokens_per_step` 控制每次更新累积多少个完整 state。state 内所有有标签题目的全部分支要一起放入预算；单个 state 超限时直接报错，不会拆分或截断。

`max_length` 与 `tokens_per_step` 是两层限制。例如，一条记录有五个长度 2000 的分支：每个分支满足 `max_length=8192`，但总量为 10000，无法放入 `tokens_per_step=8000`。

## 可训练参数与优化器

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `projection_dim` | `256` | 决策头两组投影的维度 |
| `lora_rank` | `32` | LLM LoRA rank；`warmup` 不创建 LoRA |
| `lora_alpha` | `64` | LoRA alpha；dropout 固定为 0 |
| `gradient_checkpointing` | `true` | 非 `warmup` stage 启用非重入梯度检查点 |
| `head_lr` | `2e-4` | 决策头学习率 |
| `lora_lr` | `5e-5` | LLM LoRA 学习率 |
| `merger_lr` | `1e-5` | VIT merger 学习率 |
| `vision_lr` | `2e-6` | 最后四个视觉 block 的学习率 |
| `weight_decay` | `0.01` | AdamW 的权重衰减，作用于有梯度的参数 |
| `max_grad_norm` | `1.0` | 同步后全部可训练参数的梯度裁剪阈值 |
| `rps_weight` | `0.0` | SFT 中 Score 的 RPS 权重；RLCD 必须为 0 |
| `cache_frozen_features` | `false` | 仅 RLCD 使用，要求backbone完全冻结 |

只为当前 stage 的可训练部分建立优化器组，学习率在运行中保持常数。当前示例把 `head_lr` 设为 `1e-5`，SFT 示例为 `2e-4`。LoRA 目标模块和每组参数量会写入 `run_manifest.json`。

`warmup` 指只训练决策头的 stage，不是学习率 warmup。

## RLCD 参数

设置 `method="rlcd"` 后，在 `rlcd` 对象内配置以下字段。缺省项自动补齐，未知 RLCD 字段会报错。

| 字段 | 默认值 | 含义与限制 |
| --- | --- | --- |
| `group_size` | `16` | 每题有放回采样的动作数，整数 ≥ 2 |
| `num_iterations` | `2` | 每批 rollout 的优化次数，整数 ≥ 1 |
| `correctness_weight` | `1.0` | 标签正确性奖励权重 |
| `confidence_weight` | `1.0` | 所选动作概率误差的惩罚权重 |
| `brier_weight` | `1.0` | 完整候选分布的 Brier 辅助损失权重 |
| `beta` | `0.02` | 对固定参考策略的 KL 权重；0 时不创建参考模型 |
| `clip_epsilon` | `0.2` | 策略概率比的裁剪半径，严格在 `(0, 1)` 内 |
| `advantage_epsilon` | `1e-6` | 组内优势标准化的分母稳定项，必须为正 |

所有浮点参数必须有限且非负，两个奖励权重不能同时为 0。奖励、KL 方向及损失公式见[训练说明](../visualjev/training/README.md)。

`step`、`max_steps` 和 `save_every` 均按 rollout 批次计；`optimizer_steps` 按实际更新计。默认 RLCD 每批更新两次，SFT 每批一次。对比实验除了 step，还需比较 `optimizer_steps` 和实际耗时。

## 从示例建立一个实验

下面创建配置副本，保留仓库示例。先准备好 `data/train.jsonl`，再执行训练。

```bash
mkdir -p output/experiment-configs
python - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("configs/train/sft_joint.json").read_text())
config.update(data="data/train.jsonl", output="output/traffic-joint", epochs=5, max_steps=1000)
Path("output/experiment-configs/traffic-joint.json").write_text(
    json.dumps(config, indent=2) + "\n", encoding="utf-8"
)
PY

python -m visualjev.train --config output/experiment-configs/traffic-joint.json
```

每次独立实验使用新输出目录。普通新训练会覆盖该目录的日志和 `latest/`。

## 配置校验与恢复

恢复时会比较 checkpoint 内的配置与当前配置，包括数据文件摘要。只允许调整 `output`、`epochs`、`max_steps`、`device`、`save_every`；更换数据、stage、学习率或 RLCD 参数，应使用 `--initialize` 开始新实验。

普通配置并没有统一的字段 schema，拼错的顶层字段不一定报错。`normalize_config` 会检查训练目标和已移除的蒸馏选项：非零 `kd_weight` 或有效 `teacher_checkpoint` 会被拒绝；旧 checkpoint 中关闭的蒸馏字段会被清理。`rlcd` 子对象的字段则严格校验。实现分别见 [runner.py](../visualjev/training/runner.py)、[model.py](../visualjev/modeling/model.py) 和 [rlcd.py](../visualjev/training/rlcd.py)。
