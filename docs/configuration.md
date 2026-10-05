# 训练配置参考

`python -m valen.train --config <file.json>` 读取一个 JSON 对象。`--method` 和 `--output` 可覆盖对应字段，其他训练参数在 JSON 中修改。配置中的相对路径以**进程工作目录**为基准；以下命令均从仓库根目录执行。

`configs/train/qwen/` 中的八份配置使用 Qwen3.5-2B 和合成数据集，运行 3 个 epoch、100 个 step，先达到的上限结束训练。配置按 `model`、`data`、`training`、`objective` 分区，`config_version` 为 2；`data.path` 是 JSONL 路径，其他字段沿用下表中的名称。

CLI 在应用 `--method`、`--output` 之前展开配置，checkpoint 保存展开后的字段。旧的平铺配置及原路径继续可用，同名字段出现相互冲突的值时直接报错。新旧配置归一化后相同，可以跨配置布局恢复同一次训练。双编码器的基座组合配置位于 `configs/train/dual_encoder/modernbert_dinov3b16/`。

## 数据、模型和训练budget

下表的默认值是源码在字段缺省时使用的值，示例 JSON 可以覆盖它们。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `architecture` | `"qwen"` | `qwen` 或 `dual_encoder`；下表中的 Qwen 参数只用于 Qwen 路径 |
| `model_path` | 必填 | 本地 Qwen3.5 模型及处理器目录 |
| `data` | 必填 | JSONL 文件路径；每卡读取完整文件后按记录分片 |
| `output` | 必填 | 日志及 `latest/` checkpoint 的输出目录 |
| `method` | `"sft"` | `sft` 或 `rlcd` |
| `stage` | `"joint"` | `warmup`、`text`、`joint`、`vision_top` |
| `device` | `"cuda"` | 设备；多卡 CUDA 训练按 `LOCAL_RANK` 绑定 |
| `dtype` | `"bf16"` | backbone精度，可选 `bf16`、`fp32`；决策头保持 FP32 |
| `attn_implementation` | `"eager"` | Qwen 注意力后端：`eager`、`sdpa`、`flash_attention_2` |
| `qwen_execution` | `"question"` | Qwen 执行方式；`shared_state` 让同一记录各题共享一次前向；接受 `share_state` 别名 |
| `seed` | `42` | 初始化、记录排序及采样使用的种子 |
| `epochs` | `1` | 最大数据遍历轮数 |
| `max_steps` | `2**63-1` | 最大批次数 |
| `max_length` | `8192` | 单个完整分支的 token 上限，超限报错 |
| `tokens_per_step` | `16384` | 每 rank 每批的 `compute_tokens` 上限，必须为正 |
| `microbatch_size` | `1` | Qwen 每次并行前向/反向的 QA 数量；shared_state 模式下为 state 数量，支持 SFT 和 RLCD |
| `microbatch_max_tokens` | `32768` | 每次 backbone 前向的补齐后 token 总量上限 |
| `data_prefetch` | `false` | Qwen SFT 或 shared_state RLCD 在 CPU 上预处理下一批，同时执行当前批的 GPU 计算 |
| `sft_iterations` | `1` | 同一 SFT 批次的优化次数，正整数；可与 RLCD 的 `num_iterations` 对齐进行实验对照 |
| `loss_reduction` | `"state_mean"` | 先对 state 内各题平均，再对 state 平均；`question_mean` 按全部有标签 QA 平均 |
| `media_kwargs` | `{}` | 传给官方处理器的参数；随 checkpoint 保存 |
| `save_every` | `100` | 每隔多少批覆盖保存 `latest/`；应设为正整数，正常结束也会保存 |

`tokens_per_step` 控制每次更新累积多少个完整 state。state 内所有有标签题目的全部分支要一起放入预算；单个 state 超限时直接报错，不会拆分或截断。

`max_length` 与 `tokens_per_step` 是两层限制。例如，一条记录有五个长度 2000 的分支：每个分支满足 `max_length=8192`，但总量为 10000，无法放入 `tokens_per_step=8000`。

## Qwen 共享 state 训练

`qwen_execution="shared_state"` 使用一次 backbone 前向为同一记录的全部题目评分，
单题也是同一实现的特例。`microbatch_size` 在此模式下是并行 state 数量上限；
同一 state 的图像/视频输入不会按题目重复。`loss_reduction="question_mean"` 可在
每条记录题目数量不同时保留按 QA 数指定的数据比例。模式和 loss reduction 随
checkpoint 保存，切换它们需要 `--initialize` 开始新实验。

模型结构、数据合并、两阶段 Mixer 配方和验证命令见[共享 state 说明](qwen-shared-state.md)。

## Qwen 多 QA 并行训练

`microbatch_size > 1` 会把多条 QA 的语言输入组成真正的 batch，同时拼接图像、
视频的 patch 和 grid。文本在右侧补齐，原有候选位置和角色池化区间保持有效。
每次更新内部按长度分组，减少 padding；Score 的多个分支按原顺序合并后计算 loss。
每个 state 的题目平均权重以及 `tokens_per_step` 控制的优化器更新预算保持不变。

例如，可在 `training` 中配置：

```json
{
  "tokens_per_step": 32768,
  "microbatch_size": 16,
  "microbatch_max_tokens": 65536,
  "data_prefetch": true
}
```

`microbatch_size` 是 QA 上限，实际数量还受序列长度和 Score 分支数限制。
`microbatch_max_tokens` 限制单次前向的补齐后分支 token 总数；单条分支超过该预算时
仍单独执行，原有 `max_length` 校验继续生效。多个 Score 分支可能共享一次前向，
也可能分为几次前向；它们的计算图保留至该题完成一次反向。

`data_prefetch` 使用每卡一个 CPU 工作线程。checkpoint 仅保存已消费批次的游标和
候选排列 RNG，尚在预取的数据会在恢复后重新准备。训练指标中每卡的
`prepare_wait_seconds` 记录等待预处理的时间，`cpu_prepare_seconds` 记录预处理耗时。
并行 QA 和预取目前支持 Qwen SFT。

## Qwen FlashAttention-2

在配置的 `model` 段中加入 `"attn_implementation": "flash_attention_2"`，或使用
`--attn-implementation` 覆盖。平铺配置支持同名字段。缺省值仍为 `eager`，旧
checkpoint 可以在推理或独立评估时切换后端，无需转换权重。

FlashAttention-2 要求 CUDA GPU、Ampere 或更新架构、`dtype="bf16"`，以及与
当前 PyTorch/CUDA 匹配的 `flash-attn` 扩展。先安装项目依赖，再在 GPU 环境安装
扩展；源码安装需要 CUDA toolkit 和编译工具。已验证的扩展版本为 2.7.4.post1，
框架最低要求为 2.3.3。Qwen3.5-2B 已在单卡 H200 上通过图文/视频前向、SFT/RLCD
反向、checkpoint 保存恢复及优化器恢复检查，文本与视觉实际后端均为 FlashAttention-2。

```bash
python -m pip install packaging ninja psutil
MAX_JOBS=4 python -m pip install 'flash-attn==2.7.4.post1' --no-build-isolation

python -m valen.train --config configs/train/qwen/sft_joint.json \
  --attn-implementation flash_attention_2 --output output/qwen-flash-joint

python -m valen.inference --checkpoint output/qwen-flash-joint/latest \
  --data data/smoke/train.jsonl --output output/predictions.jsonl \
  --attn-implementation flash_attention_2

python -m valen.evaluate --checkpoint output/qwen-flash-joint/latest \
  --data data/smoke/train.jsonl --output output/flash-evaluation \
  --attn-implementation flash_attention_2
```

`sdpa` 使用 PyTorch 的 scaled dot-product attention，不需要额外安装 `flash-attn`；
CUDA 上的具体融合内核由 PyTorch 根据输入与设备选择，CPU 与 FP32 也可使用。
显式选择 FlashAttention-2 时，设备、精度或扩展不兼容会直接报错。

Qwen3.5 的语言 full-attention 层及视觉注意力使用所选后端；Gated DeltaNet
linear-attention 层仍使用自身实现，其优化依赖 `flash-linear-attention` 和
`causal-conv1d`。四种 decision head 保持 FP32。训练 manifest 的
`adaptation.attention_backends` 及普通评估结果的 `attention_backends` 记录文本与视觉
的实际后端。不同后端可能产生浮点舍入差异，做速度比较时应明确记录后端。

推理/评估的 CLI 覆盖不会修改 checkpoint。Qwen 恢复训练时允许切换注意力后端、
并行 QA 数量、并行 token 预算、激活检查点和预取设置，并继续恢复优化器及 RNG。
manifest 的 `resume_runtime_changes` 记录这些运行设置的变化。模型、数据、损失和
学习率继续受配置一致性检查约束；后端及批量计算可能带来浮点舍入差异。

GPU 集成验证可复用现有 smoke 脚本，覆盖图文/视频、联合反向、保存/恢复和 RLCD：

```bash
python scripts/smoke/qwen/gpu_smoke.py --attn-implementation flash_attention_2 \
  --output artifacts/qwen-flash-smoke
```

参考：[Transformers 5.4 注意力后端](https://huggingface.co/docs/transformers/v5.4.0/en/attention_interface)、
[FlashAttention 安装要求](https://github.com/Dao-AILab/flash-attention/tree/v2.7.4.post1)。

## 可训练参数与优化器

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `head_type` | `"bilinear"` | Qwen 评分头：`bilinear`、`mlp`、`role_mlp` 或 `mixer` |
| `projection_dim` | `256` | `bilinear` 两组投影的维度 |
| `head_hidden_dim` | `512` / `1024` | `mlp` / `role_mlp` 的隐藏层维度 |
| `head_bottleneck_dim` | `128` | `mlp` 的第二个隐藏层维度 |
| `head_width` | `256` | `role_mlp` / `mixer` 的角色投影维度 |
| `head_layers` | `2` | `mixer` 的模块数 |
| `head_token_hidden_dim` | `16` | `mixer` 角色混合 MLP 的隐藏维度 |
| `head_channel_hidden_dim` | `1024` | `mixer` 通道混合 MLP 的隐藏维度 |
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
| `brier_weight` | `0.0` | SFT 中完整候选概率分布的 Brier 权重；RLCD 使用 `rlcd.brier_weight` |
| `cache_frozen_features` | `false` | 仅 RLCD 使用，要求backbone完全冻结 |

`head_*` 模型参数可放在分区配置的 `model` 内。相关维度必须为正整数；初始化时须保持 head 类型与维度一致。输入池化和各网络结构见 [Qwen 决策头](../valen/modeling/qwen/HEADS.md)。

只为当前 stage 的可训练部分建立优化器组，学习率在运行中保持常数。Qwen 的 RLCD 示例把 `head_lr` 设为 `1e-5`，SFT 示例为 `2e-4`。架构适配信息写入 `run_manifest.json` 的 `adaptation`，Qwen 的 LoRA 目标位于 `adaptation.lora_targets`；每组参数量仍在 `optimizer_groups`。

stage 由各架构的 builder 解释。Qwen 的 `text` 要求纯文本数据；双编码器的 `text`/`joint` 允许图文数据，解冻文本塔末层并保持视觉塔冻结。切换架构不能沿用另一个架构的 stage 含义或 checkpoint。

`warmup` 是冻结编码器的 stage：Qwen 只训练决策头，双编码器训练投影、融合与完整决策读取模块。它不表示学习率预热，也不是图片 + caption 的图文预训练。

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

所有浮点参数必须有限且非负，两个奖励权重不能同时为 0。奖励、KL 方向及损失公式见[训练说明](../valen/training/README.md)。

Qwen `shared_state` RLCD 支持 `microbatch_size > 1` 与 `data_prefetch`。
策略、固定参考策略以及梯度更新都按完整 state 组批；每题的采样动作与原题对应，
不会因长度排序错位。使用 `cache_frozen_features=true` 时仍要求单条 microbatch。
示例见 [rlcd_shared_state_joint.json](../configs/train/qwen/rlcd_shared_state_joint.json)，
启动时使用 `--initialize` 指向 SFT checkpoint。

`step`、`max_steps` 和 `save_every` 均按 rollout 批次计；`optimizer_steps` 按实际更新计。默认 RLCD 每批更新两次，SFT 每批一次。对比实验除了 step，还需比较 `optimizer_steps` 和实际耗时。

## 从示例建立一个实验

下面创建配置副本，保留仓库示例。先准备好 `data/train.jsonl`，再执行训练。

```bash
mkdir -p output/experiment-configs
python - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("configs/train/qwen/sft_joint.json").read_text())
config["data"]["path"] = "data/train.jsonl"
config["training"].update(output="output/traffic-joint", epochs=5, max_steps=1000)
Path("output/experiment-configs/traffic-joint.json").write_text(
    json.dumps(config, indent=2) + "\n", encoding="utf-8"
)
PY

python -m valen.train --config output/experiment-configs/traffic-joint.json
```

每次独立实验使用新输出目录。普通新训练会覆盖该目录的日志和 `latest/`。

## 配置校验与恢复

恢复时会比较 checkpoint 内的配置与当前配置，包括数据文件摘要。只允许调整 `output`、`epochs`、`max_steps`、`device`、`save_every`；更换数据、stage、学习率或 RLCD 参数，应使用 `--initialize` 开始新实验。

普通配置并没有统一的字段 schema，拼错的顶层字段不一定报错。`normalize_config` 会检查架构标识、训练目标和已移除的蒸馏选项：非零 `kd_weight` 或有效 `teacher_checkpoint` 会被拒绝；旧 checkpoint 中关闭的蒸馏字段会被清理。`rlcd` 子对象的字段则严格校验。实现分别见 [runner.py](../valen/training/runner.py)、[factory.py](../valen/modeling/factory.py) 和 [rlcd.py](../valen/training/rlcd.py)。
