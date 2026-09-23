# 推理与评估

本文介绍 `visionjev` 包内的逐题推理与评估。Sokoban 的完整游戏评测使用独立的 `evaluation.sokoban` 命令，数据生成、策略比较和轨迹回放见[任务评测说明](../evaluation/sokoban/README.md)。

推理和评估都从 checkpoint 的 `config.json` 重建模型，再加载 `checkpoint.pt` 中的参数。基础模型仍需保存在配置的 `model_path` 下。两种命令的输出格式不同：推理每条记录返回一个响应，评估每道有标签题目输出一行。

## 推理命令

在仓库根目录、已激活项目环境的终端中运行：

```bash
mkdir -p output/predictions
python -m visionjev.inference \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/predictions/smoke.jsonl
```

三个路径参数均必填。`--output` 是文件路径，父目录需事先存在；已有文件会被覆盖。`--device` 默认 `cuda`，可指定如 `cuda:1`。该 CLI 是单进程推理，没有按 rank 分片的逻辑。

推理处理全部问题，输入可省略 `targets`；仍需保留 `group_id`。输出行按输入记录顺序排列，不附带 `group_id` 或 `meta`，通过行序和 `answers` 中的问题 ID 对应输入。

下面展示一道 Noul 题的响应结构。概率和 token 数仅用于说明字段，不是已训练模型的实测结果。

```json
{
  "model": "VisualJev",
  "answers": {
    "go": {"type": "noul", "noul": 0.8}
  },
  "usage": {"input_tokens": 64, "output_tokens": 0},
  "internal_usage": {"compute_tokens": 64}
}
```

`output_tokens` 固定为 0。`input_tokens` 对公共 state 只计一次，`compute_tokens` 包含分支内重复的 state，具体计算见[架构说明](architecture.md#公共上下文与计算量)。

## 三种答案的字段

设 `p = softmax(logits / temperature)`，候选数为 K，等级索引从 0 开始。实现见 [answer](../visionjev/evaluation/inference.py)。

| 类型 | 字段 |
| --- | --- |
| Choice | `type`、`choice`（最大概率候选名）、`probabilities`（候选名 → 概率）、`confidence` |
| Noul | `type`、`noul`（`true` 的概率）；不另返回 `probabilities` 或 `confidence` |
| Score | `type`、`score`（`sum(i * p[i])`）、`probabilities`（字符串等级索引 → 概率）、`legend`（索引 → 描述）、`confidence` |

Choice 的 `confidence` 将最大概率按均匀分布基线缩放：

```text
K = 1: confidence = 1
K > 1: confidence = max(0, (max(p) - 1/K) / (1 - 1/K))
```

Score 使用相对众数的平均绝对距离：

```text
mode = argmax(p)
distance = sum(p[i] * abs(i - mode))
uniform_mad = sum(abs(i - (K-1)/2)) / K
confidence = max(0, 1 - distance / uniform_mad)
```

并列最大值取当前候选顺序中的第一个。Score 的 `score` 是期望值，不是众数，也不自动缩放到 0–1。例如 `[0.1, 0.3, 0.6]` 对应 `score=1.5`、`confidence=0.25`。`confidence` 描述分布集中程度，不能直接解释成“有多少概率答对”。

### 温度设置

推理支持 `--calibration path/to/calibration.json`，文件格式为 `{"temperature": 1.0}`。温度必须有限且大于 0，对本次推理的所有问题统一使用。仓库没有拟合温度的命令；若自行拟合，应使用独立验证集。

**评估 CLI 没有 `--calibration` 参数，固定使用温度 1。** 它的结果不能作为带温度校准的推理结果来报告。

## 评估命令与产物

```bash
# 合成集只检查流程；正式评估替换为独立测试集。
python -m visionjev.evaluate \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/smoke_eval

# 同一数据按记录分到四个进程，不填充重复样本。
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m visionjev.evaluate \
  --checkpoint output/sft_warmup/latest \
  --data data/smoke/train.jsonl \
  --output output/sft_warmup/smoke_eval_4gpu
```

这里的 `--output` 是目录，程序会自动创建。输出包括：

| 文件 | 内容 |
| --- | --- |
| `predictions.rank<N>.jsonl` | 当前 rank 的逐题结果；单进程也会生成 `rank0` |
| `predictions.jsonl` | rank 0 合并的结果，按 `record_index`、`qid` 排序 |
| `metrics.json` | 指标、数据与 checkpoint 来源、计数、各卡耗时及延迟分位数 |

逐题记录包含 `record_index`、`record_id`、`group_id`、`qid`、任务和分组信息、标签分布、预测分布、结构化答案、指标及计时。合并前检查全部有标签题目是否恰好处理一次。多卡进程需能访问同一个输出目录；复用目录会覆盖同名产物。

评估数据至少要有一道有效标签题。全无标签的数据应使用推理命令；当前评估实现会在空延迟集合的分位数统计处失败。

## 指标口径

记标签分布为 y，预测分布为 p。实现见 [question_metrics / summarize](../visionjev/evaluation/metrics.py)。

| 指标 | 适用范围 | 定义 |
| --- | --- | --- |
| `nll` | 全部有标签题 | `-sum(y[k] * log(p[k]))`，使用自然对数 |
| `brier` | 全部有标签题 | `sum((p[k] - y[k])²)`，不除候选数 |
| `accuracy` | 严格 one-hot 标签 | 预测 argmax 与标签 argmax 相同记为 1；Score 也按离散等级判断 |
| `expected_score_mae` | Score，包括软标签 | `abs(sum(k*p[k]) - sum(k*y[k]))` |
| `rps` | Score，包括软标签 | 前 K−1 个累积分布差的平方平均 |
| `macro_f1` | Noul 硬标签 | true/false 两类 F1 的算术平均；某类分母为 0 时该类 F1 记为 0 |

汇总按**题目等权平均**。`metric_counts` 给出每项指标的有效题数；软标签不计 accuracy，因此该分母通常不同于 `questions`。Noul 的混淆矩阵与 `macro_f1` 只出现在 `task/noul` 和 `task_modality/noul/*` 分组，不出现在 overall 或单独的语言、领域分组。

分组键包括 `overall`、`task/*`、`modality/*`、`task_modality/*/*`、`language/*`、`domain/*`。模态优先读取 `meta.modality`；未提供时只有 `text` 和 `media` 两种分类，不自动区分图片与视频。语言读取 `meta.language_bucket`，领域读取 `meta.domain`，未提供均为 `unknown`，不自动检测。

## 如何读计时

- `compile_seconds` 是整条 state 的编译耗时，同一 state 的每道题都会记录这一个值。
- `forward_seconds` 是该题所有分支的模型前向耗时；CUDA 前后执行同步。包含首次执行开销，未做独立 warmup。
- `compile_plus_forward_seconds` 将上述两项直接相加，不按题分摊 state 编译成本，也不包含完整的结果序列化和写盘流程。
- `evaluation_wall_seconds` 取各卡评估循环的最大耗时；`load_and_evaluation_wall_seconds` 还包含初始化和模型加载。
- `questions_per_second` 为有标签题数除以 `evaluation_wall_seconds`。

延迟的 mean、p50、p95 按逐题记录计算。同一 state 的编译时间会重复计入，所以不能把逐题延迟求和当成整个数据集耗时。训练中的 state 等权损失、评估中的逐题均值和这些耗时应分别报告。
