# Valen：SFT 与 RLCD

`runner.py` 共用数据分片、token 预算、梯度同步和 checkpoint 流程。`sft.py` 实现标签监督，`rlcd.py` 实现基于 GRPO 的优化。四个 `stage` 决定更新哪些参数，`method` 决定训练目标，两者独立。

这里的 RLCD 是 Valen 的实验实现，奖励公式以本仓库的 [rlcd.py](rlcd.py) 为准。

## SFT loss

[sft.py](sft.py) 对候选 logits 使用分布交叉熵。设预测概率为 p、标签分布为 y，候选数为 K：

$$
\mathrm{CE}=-\sum_{k=1}^{K}y_k\log p_k.
$$

$$
\mathrm{RPS}=\frac{1}{K-1}\sum_{j=1}^{K-1}
\left(\sum_{k=1}^{j}(p_k-y_k)\right)^2.
$$

$$
L_{\mathrm{SFT}}=\mathrm{CE}+w_{\mathrm{RPS}}\mathrm{RPS}.
$$

硬标签和软标签使用同一公式。`rps_weight` 对应公式中的 RPS 权重，默认 0，此时三种任务都只优化交叉熵。Score 的 RPS 区分等级间的远近，使用前 K−1 个累积分布差；Choice 和 Noul 的 RPS 项为 0。

每个 state 内先对有标签的问题平均，再对当前批次的全局有效 state 平均。一条有三道题的记录与一条有一道题的记录权重相同。Score 的多个分支先拼成同一道题的 logits，不各自增加损失权重。

## RLCD reward与loss

每道题从旧policy的候选分布中有放回采样 `group_size` 个action，不生成文本。对采样答案 $a_i$，记其旧策略概率为 $p_i$，标签为 $y_i=y_{a_i}$：

$$
r_i = w_{\mathrm{correct}}y_i
-w_{\mathrm{confidence}}\left[y_i(1-p_i)^2+(1-y_i)p_i^2\right].
$$

硬标签下，选中答案的标签 $y_i$ 表示答对与否，第二项就是 $(p_i-y_i)^2$。默认权重均为 1：以 0.9 的概率答对，奖励 0.99；以 0.9 的概率答错，奖励 −0.81。软标签保留原分布，使用期望正确性与期望二元平方误差，不先取 argmax。

这里的“置信度”是所选候选的概率。Score 的 RL 动作是一个离散action，推理仍输出期望值。

组内优势为：

$$
A_i=\frac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)+\epsilon}.
$$

标准差使用总体定义；奖励完全相同的组，优势明确置零。采样动作、奖励、优势、旧策略 log-prob 在一次 rollout 的所有更新中固定。

$$
\rho_i=\frac{\pi_\theta(a_i\mid x)}{\pi_{\mathrm{old}}(a_i\mid x)}.
$$

$$
L_{\mathrm{policy}}=-\mathrm{mean}_i\min\left(
\rho_i A_i,\mathrm{clip}(\rho_i,1-\delta,1+\delta)A_i
\right).
$$

$$
L=L_{\mathrm{policy}}+\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
+w_{\mathrm{Brier}}\sum_k(\pi_\theta(k)-y_k)^2.
$$

参考策略固定为 RL 开始时的模型，KL 用full-distribution精确计算，约束策略更新幅度。rollout 和策略更新时关闭 dropout；训练前向仍支持 gradient checkpointing。

Brier 辅助项默认开启，设 `brier_weight=0` 可关闭该辅助项。

## 运行

先按[安装说明](../../scripts/README.md#installation)准备并激活 Python 环境，下载base model。以下命令均在仓库根目录执行，`python` 使用当前激活的环境。RLCD 必须指定初始化 checkpoint 或恢复 checkpoint，独立实验输出到新目录。

```bash
# 先训练 SFT 决策头，产生后续命令需要的 checkpoint。
python -m valen.train --config configs/train/sft_warmup.json

# 单卡 RLCD；warmup 表示只更新决策头。
python -m valen.train \
  --config configs/train/rlcd_warmup.json \
  --initialize output/sft_warmup/latest

# 四卡 joint 实验，使用与单卡 warmup 不同的输出目录。
VJ_GPUS=4 bash scripts/train/launch_sft.sh \
  configs/train/rlcd_joint.json \
  --initialize output/sft_warmup/latest

# 恢复中断的四卡 joint 实验；需要保持原卡数。
VJ_GPUS=4 bash scripts/train/launch_sft.sh \
  output/rlcd_joint/latest/config.json --resume output/rlcd_joint/latest
```

`configs/train/` 中，SFT 和 RLCD 各有四份 stage 示例。`rlcd_*.json` 已设置 `method`、独立输出目录及完整 RLCD 参数。这些参数用于运行示例，尚未经过正式数据集调优。上述命令使用随仓库提供的 smoke 数据；正式训练前设置 `data`、轮数、步数和输出目录。`text` 仅使用纯文本；RLCD 的 `rps_weight` 必须为 0。

`--initialize` 加载已有权重，开始新实验；`--resume` 恢复优化器、随机状态和进度。已达到 `epochs` 或 `max_steps` 的任务不会因恢复而自动追加训练，如需继续，应在配置副本中提高对应上限。更换 GPU 数时使用 `--initialize`。单机多卡脚本使用当前环境的 `python`，也可通过 `VJ_PYTHON` 指定解释器。全部配置字段见[配置参考](../../docs/configuration.md)。

奖励与优化参数在配置的 `rlcd` 对象中调整。示例使用以下默认值，实际值会写入 checkpoint 配置。

```json
{
  "rlcd": {
    "group_size": 16,
    "num_iterations": 2,
    "correctness_weight": 1.0,
    "confidence_weight": 1.0,
    "brier_weight": 1.0,
    "beta": 0.02,
    "clip_epsilon": 0.2,
    "advantage_epsilon": 0.000001
  }
}
```

`step`、`max_steps`、`save_every` 按 rollout 批次计；`optimizer_steps` 记录实际优化次数，默认每批更新两次。SFT 每批更新一次。`tokens_per_step` 仍是每卡输入预算，不包含重复策略前向的成本。题目在 state 内平均，再按全局 state 数平均，空闲 rank 参加相同次数的梯度同步。

`beta>0` 时，每张卡会额外驻留一份冻结参考模型。

RLCD 仅训练决策头时，可设置 `cache_frozen_features=true`：每批只提取一次backbone特征，旧策略、参考头和多次策略更新复用这些特征；每张卡首次使用时额外执行完整前向，核对 logits 完全一致。此时参考策略只需保存冻结决策头，无需额外加载一份backbone。backbone有可训练参数时拒绝启用。此开关不影响 SFT；SFT 的 `warmup` 支持 `--initialize`，可与 RLCD 共享同一初始决策头。

`metrics.jsonl` 记录奖励、采样正确性、采样概率、置信度误差、Brier、KL、裁剪比例和零优势组比例，数值在本批的策略更新间取平均；梯度范数记录最后一次更新。正式评估使用 `python -m valen.evaluate`。

## 多卡与批次

`launch_sft.sh` 启动单机 `torch.distributed.run`，默认两进程。每卡完整加载模型；CUDA 使用 NCCL，CPU 测试使用 Gloo。训练未使用 DDP 包装，而是在本地逐题累积后显式同步梯度，使图片、文本、Score 分支数不同的 rank 也能参加相同的 collective。

每轮记录排序后按 rank 分片，不补齐样本、不丢弃尾部。各卡分别按 `tokens_per_step` 打包完整 state，较早耗尽数据的 rank 继续参加梯度同步。所有 rank 都耗尽当前分片后进入下一轮。每次更新前的梯度已经按全局 state 数归一化，同步使用求和。

`tokens_per_step` 是每卡的分支 token 总量上限，不是固定 batch size；它也不限制 RLCD 的重复前向次数。一个 state 超过预算时会直接报错。具体计数见[架构说明](../../docs/architecture.md#公共上下文与计算量)。

## checkpoint 与恢复

一次训练会在 `output` 下生成：

```text
run_manifest.json       配置、依赖版本、参数组、LoRA 目标和初始化来源
metrics.jsonl           每批指标、优化次数、梯度范数、各卡工作量和耗时
media.jsonl             单进程的媒体与 state 使用记录
media.rank<N>.jsonl     多进程时每卡各自的媒体记录
latest/
  checkpoint.pt         参数、优化器、进度和随机状态
  config.json           重建模型所需的配置
```

`latest/` 在每次保存时覆盖。`checkpoint.pt` 先写临时文件再替换；随后单独写 `config.json`。需要保留某次结果时，在训练保存完成后复制整个 `latest/` 目录。

这里的“增量权重”指可训练参数的完整值，包括决策头、LoRA 和当前开放的视觉参数，并非每个参数相对基础模型的数值差。冻结参数没有写入 checkpoint。RLCD 在 `beta>0` 时另存最初的参考参数，恢复时继续使用它们。

| 行为 | `--initialize` | `--resume` |
| --- | --- | --- |
| 模型参数 | 加载已有可训练参数 | 恢复当前 stage 的全部可训练参数 |
| 优化器、随机状态、数据游标 | 重新开始 | 按 rank 恢复 |
| GPU 进程数 | 可以变化 | 必须与保存时一致 |
| stage | 新参数集合必须包含旧 checkpoint 的全部参数 | 必须保持一致 |
| 数据、学习率、训练目标 | 可作为新实验修改 | 必须保持一致 |
| RLCD 参考策略 | 以初始化后的模型建立 | 使用原始参考参数 |

例如 `warmup → joint` 可以初始化；`joint → warmup` 会因旧参数被重新冻结而拒绝。初始化要求 `model_path` 字符串和 `projection_dim` 与原配置一致，已有参数的形状也必须兼容。

恢复只允许修改 `output`、`epochs`、`max_steps`、`device`、`save_every`，其他配置及训练 JSONL 摘要需要一致。提高 `max_steps` 不会越过已用尽的 `epochs` 上限。checkpoint 保存了各 rank 的顺序、游标和 Python/PyTorch 随机状态。

基础模型目录若有 `valen_manifest.json`，checkpoint 会保存它，并在加载时比较 revision 和权重字段。加载时不会重新读取全部基础权重计算摘要。加载器使用 `torch.load(..., weights_only=False)`。

保存和恢复实现见 [checkpoint.py](checkpoint.py)，配置比较见 [runner.py](runner.py)。推理与评估共用同一加载器，详见[推理与评估](../../docs/evaluation.md)。

## 验证

```bash
python -m pytest -q

# 需要本机有两张可见 GPU，并已下载 0.8B 模型。
python -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node=2 --max_restarts=0 scripts/smoke/rlcd_gpu_smoke.py
```

CPU 测试检查奖励方向、裁剪梯度、软标签、三类输出、零优势组、参考策略与两进程恢复。GPU 短测使用 smoke 数据，覆盖文本、图片和视频，以及 LoRA、视觉 merger、决策头的更新；结果写入 `output/rlcd_gpu_smoke/result.json`。
运行日志和 checkpoint 由上述命令在本地生成。本地运行方式见[脚本说明](../../scripts/README.md)。
