# 代码目录

命令入口位于包顶层，实际逻辑分为数据、模型、训练、评估四部分。下面的路径可直接用于定位实现；端到端流程见[架构说明](../docs/architecture.md)。此处的 `visualjev/evaluation/` 负责通用逐题评估；仓库根目录另有[任务评测包](../evaluation/README.md)，实现 Sokoban 环境和完整游戏评测。

| 文件 | 主要对象或函数 | 职责 |
| --- | --- | --- |
| [data/schema.py](data/schema.py) | `candidates`、`target_distribution`、`read_jsonl` | 构造三类候选，校验标签，读取记录 |
| [data/compiler.py](data/compiler.py) | `Compiler`、`CompiledState`、`Question`、`Branch` | 解析媒体，编码上下文，展开问题分支，记录读取位置与 token 数 |
| [modeling/model.py](modeling/model.py) | `VisualJev`、`DecisionHead`、`build_model`、`optimizer_groups` | 加载骨干，配置 LoRA/视觉参数，计算候选 logits，建立参数组 |
| [training/runner.py](training/runner.py) | `normalize_config`、`run` | 共用训练循环、打包预算、初始化与恢复、日志和保存 |
| [training/sft.py](training/sft.py) | `SFTObjective`、`question_loss` | 分布交叉熵与可选 Score RPS |
| [training/rlcd.py](training/rlcd.py) | `RLCDObjective`、`Rollout` | 候选采样、奖励与优势、裁剪损失、固定参考策略和特征复用 |
| [training/distributed.py](training/distributed.py) | `Distributed`、`epoch_shard`、`synchronize_gradients` | 按 rank 分片、训练参数广播、累积后的梯度求和 |
| [training/checkpoint.py](training/checkpoint.py) | `capture_rank_state`、`save_checkpoint`、`load_checkpoint` | 可训练参数和各 rank 的优化进度、随机状态 |
| [evaluation/inference.py](evaluation/inference.py) | `answer`、`predict` | logits 转概率、三类响应、推理 CLI |
| [evaluation/metrics.py](evaluation/metrics.py) | `question_metrics`、`summarize` | 逐题指标与各维度汇总 |
| [evaluation/evaluate.py](evaluation/evaluate.py) | `run` | 按记录分片评估、检查覆盖、写出预测与指标 |

## 调用关系

`train.py`、`inference.py`、`evaluate.py` 只调用对应子模块的 `main()`。Python 导入使用实现所在的子包，例如 `visualjev.data.compiler.Compiler`。

训练中，`runner.run` 先构建模型、优化器和编译器，再把有标签记录编译成一个 pack。训练目标共用以下接口：

- `prepare(model, pack)`：SFT 返回空占位；RLCD 固定本批动作、旧策略概率、奖励、优势及参考概率。
- `loss(model, question, rollout)`：计算单题损失和日志项，由 runner 负责归一化、反向、同步和更新。
- `state_dict()`：SFT 返回 `None`；RLCD 返回需要随 checkpoint 保存的固定参考参数。

`num_iterations` 决定一个 pack 更新几次，`metric_names` 决定 runner 汇总哪些目标专用指标。新增训练目标时，除实现这些方法，还需在 `normalize_config` 和目标选择处注册。

推理调用 `predict → VisualJev.forward → answer`。评估直接逐题前向，再分别调用 `question_metrics` 和 `answer`；它不调用 `predict`，所以逐题输出与推理响应的结构不同。

## 测试对应关系

| 测试 | 主要覆盖 |
| --- | --- |
| [test_decisions.py](../tests/test_decisions.py) | 标签合法性、SFT/RPS、答案字段、决策头梯度 |
| [test_compiler.py](../tests/test_compiler.py) | UTF-8 读取、候选位置、Score 隔离、媒体展开、视频位置编码 |
| [test_checkpoint.py](../tests/test_checkpoint.py) | 参数保存、优化器与随机状态恢复 |
| [test_manifest.py](../tests/test_manifest.py) | 基础模型清单的优先级、旧文件读取与歧义检测 |
| [test_distributed.py](../tests/test_distributed.py) | 无填充分片、不等数量 state、空闲 rank、梯度与恢复 |
| [test_sft_training.py](../tests/test_sft_training.py) | 小模型 SFT 循环、旧配置处理与恢复 |
| [test_rlcd.py](../tests/test_rlcd.py) | 奖励、软标签、优势、裁剪梯度和参考 KL |
| [test_rlcd_training.py](../tests/test_rlcd_training.py) | 三类任务、单/双进程连续训练与恢复的一致性 |
| [test_frozen_features.py](../tests/test_frozen_features.py) | RLCD 特征复用前后的损失、梯度和更新一致性 |
| [test_evaluate.py](../tests/test_evaluate.py) | 指标公式、有效样本分母、显式数据路径 |
| [test_model_naming.py](../tests/test_model_naming.py) | 模型标识与 checkpoint 参数键兼容 |

CPU 测试主要使用小模型替身。真实处理器测试需要本地 `models/Qwen3.5-0.8B/tokenizer.json` 等处理器文件；没有时跳过。真实权重加载、视觉梯度和 NCCL 恢复由 `scripts/smoke/` 中的 GPU 检查覆盖，命令见[脚本说明](../scripts/README.md)。

模型类使用 `VisualJev`；响应的 `model` 字段来自包顶层的 `MODEL_NAME`。Python 包、命令入口和下载模型清单使用 `visualjev` 命名。checkpoint 的参数键不变；加载器也能读取模型目录中仅有一份、字段完整的旧清单文件。
