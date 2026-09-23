# Visual-Jev：SFT 与 RLCD

`runner.py` 共用数据分片、token 预算、梯度同步和 checkpoint 流程。`sft.py` 实现标签监督，`rlcd.py` 实现基于 GRPO 的决策策略优化。四个 `stage` 决定更新哪些参数，`method` 决定训练目标，两者独立。

Jev 的[官方介绍](https://typesafe.ai/blog/introducing-system-one-models-and-jev)公开了校准决策的目标，没有给出可复现的奖励公式和训练算法。这里是 Visual-Jev 的实验实现；组内优势和裁剪目标参考 [DeepSeekMath 的 GRPO](https://arxiv.org/abs/2402.03300)。

## 奖励与损失

每道题从旧策略的候选分布中有放回采样 `group_size` 个决策，不生成文本。对采样答案 $a_i$，记其旧策略概率为 $p_i$，标签为 $y_i=y_{a_i}$：

$$
r_i = w_{\mathrm{correct}}y_i
-w_{\mathrm{confidence}}\left[y_i(1-p_i)^2+(1-y_i)p_i^2\right].
$$

硬标签下，$y_i$ 是答对与否，第二项就是 $(p_i-y_i)^2$。默认权重均为 1：以 0.9 的概率答对，奖励 0.99；以 0.9 的概率答错，奖励 −0.81。软标签保留原分布，使用期望正确性与期望二元平方误差，不先取 argmax。

这里的“置信度”是所选候选的概率，不是 API 的 `confidence` 字段。后者对 Choice 做了均匀基线缩放，对 Score 衡量等级分布的集中程度，不能直接当作正确概率。Score 的 RL 动作是一个离散等级，推理仍输出等级期望值。

组内优势为：

$$
A_i=\frac{r_i-\operatorname{mean}(r)}{\operatorname{std}(r)+\epsilon}.
$$

标准差使用总体定义；奖励完全相同的组，优势明确置零。采样动作、奖励、优势、旧策略 log-prob 在一次 rollout 的所有更新中固定。

$$
\begin{aligned}
\rho_i &= \pi_\theta(a_i\mid x)/\pi_{\mathrm{old}}(a_i\mid x),\\
L &= -\operatorname{mean}_i\min\left(\rho_i A_i,
\operatorname{clip}(\rho_i,1-\delta,1+\delta)A_i\right)\\
&\quad+\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
+w_{\mathrm{Brier}}\sum_k(\pi_\theta(k)-y_k)^2.
\end{aligned}
$$

参考策略固定为 RL 开始时的模型，KL 用完整候选分布精确计算，约束策略更新幅度。无需 critic。rollout 和策略更新时关闭 dropout；训练前向仍支持 gradient checkpointing。

Brier 辅助项默认开启。二分类的组内标准化可能抵消奖励幅度变化，同答案组也没有策略梯度，因此仅加置信度奖励不等于实现概率校准。Brier 提供直接的概率监督，但整个混合目标仍不保证严格校准；应在独立验证集上比较正确率、NLL 和 Brier。设 `brier_weight=0` 可关闭该辅助项。

## 运行

先按[安装说明](../../scripts/README.md#installation)准备并激活 Python 环境，下载基础模型。以下命令均在仓库根目录执行，`python` 使用当前激活的环境。RLCD 必须指定初始化 checkpoint 或恢复 checkpoint，独立实验输出到新目录。

```bash
# 先训练 SFT 决策头，产生后续命令需要的 checkpoint。
python -m visionjev.train --config configs/train/sft_warmup.json

# 单卡 RLCD；warmup 表示只更新决策头。
python -m visionjev.train \
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

`--initialize` 加载已有权重，开始新实验；`--resume` 恢复优化器、随机状态和进度。已达到 `epochs` 或 `max_steps` 的任务不会因恢复而自动追加训练，如需继续，应在配置副本中提高对应上限。更换 GPU 数时使用 `--initialize`。单机多卡脚本使用当前环境的 `python`，也可通过 `VJ_PYTHON` 指定解释器。

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

只在完整 rollout 批次结束后保存，恢复时不需要重建半批动作。checkpoint 保存各卡随机状态，以及 RL 开始时的参考增量权重；不会把恢复时的当前策略误当成新参考策略。`beta>0` 时，每张卡会额外驻留一份冻结参考模型。

仅训练决策头时，可设置 `cache_frozen_features=true`：每批只提取一次骨干特征，旧策略、参考头和多次策略更新复用这些特征；每张卡首次使用时核对与完整前向的 logits 完全一致。此时参考策略只需保存冻结决策头，无需额外加载一份骨干。骨干有可训练参数时拒绝启用。SFT 的 `warmup` 也支持 `--initialize`，可与 RLCD 共享同一初始决策头。

`metrics.jsonl` 记录奖励、采样正确性、采样概率、置信度误差、Brier、KL、裁剪比例和零优势组比例，数值在本批的策略更新间取平均；梯度范数记录最后一次更新。正式评估使用 `python -m visionjev.evaluate`。

## 验证

```bash
python -m pytest -q

# 需要本机有两张可见 GPU，并已下载 0.8B 基础模型。
python -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node=2 --max_restarts=0 scripts/smoke/rlcd_gpu_smoke.py
```

CPU 测试检查奖励方向、裁剪梯度、软标签、三类输出、零优势组、参考策略与两进程恢复。GPU 短测使用 smoke 数据，覆盖文本、图片和视频，以及 LoRA、视觉 merger、决策头的更新；结果写入 `output/rlcd_gpu_smoke/result.json`。这些检查验证实现，不衡量模型质量。

运行日志和 checkpoint 由上述命令在本地生成，不随仓库分发。本地运行方式见[脚本说明](../../scripts/README.md)。
