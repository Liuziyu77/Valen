# Visual-Jev 模型结构

Visual-Jev 用 Qwen3.5 多模态骨干和共享决策头，对每次请求给定的候选评分。推理直接返回概率，不生成答案 token。实现支持 0.8B 和 2B 骨干，Python 包名保留为 `visionjev`。

![模型结构](../assets/figures/readme-architecture.png)

## 输入与候选编译

每条记录包含公共 `state` 和一个或多个 `questions`。`state` 可以是文本，或含本地图片、视频的消息列表。格式见[数据说明](data-format.md)。

[Compiler](../visionjev/data/compiler.py) 使用官方处理器编码公共 state，再为问题附加任务类型、指令、候选描述和 `Decision:` 标记。问题 ID 和监督标签不进入提示词。超过 `max_length` 的输入会报错，不静默截断。

- Choice：1–255 个命名候选，同一分支编码并评分；训练时随机排列候选并同步调整标签。
- Noul：固定 true/false 两个候选，同一分支评分。
- Score：2–10 个有序等级，每个等级单独编码；各分支共享模型权重，分支不包含等级编号或其他等级描述。

公共 state 只做一次处理器编码，但每个问题或等级分支仍分别执行骨干前向。`usage.input_tokens` 按逻辑请求计数，`internal_usage.compute_tokens` 包含重复分支计算。

## 决策头与输出

[DecisionHead](../visionjev/modeling/model.py) 读取候选描述最后一个 token 和 `Decision:` 最后一个 token 的隐状态。两个独立的无偏置线性层将它们投影到同一维度，再计算缩放点积：

```text
logit[k] = dot(W_decision h_decision, W_candidate h_candidate[k]) / sqrt(projection_dim)
probabilities = softmax(logits / temperature)
```

决策头和损失使用 FP32；基础配置的骨干使用 BF16。Choice 返回最大概率候选及完整分布，Noul 返回 true 的概率，Score 返回等级索引的期望值、等级分布和图例。

Choice/Score 的 `confidence` 衡量分布集中程度，不等同于预测正确率。温度默认是 1；推理的 `--calibration` 可以读取 `{"temperature": 1.0}` 格式的文件，但本仓库不提供自动拟合温度的命令。

## 训练与 checkpoint

SFT 和实验性 RLCD 共用训练循环、分布式同步、数据分片及恢复机制。四种 stage 为 `warmup`、`text`、`joint`、`vision_top`，分别逐步开放决策头、语言 LoRA、视觉 merger 和最后四层视觉编码器。具体目标函数和命令见[训练说明](../visionjev/training/README.md)。

checkpoint 保存可训练参数增量、优化器、随机状态和训练进度；加载仍需相同基础模型。`--initialize` 用已有增量开始新的训练，`--resume` 恢复原训练且要求 GPU 进程数一致。只更新决策头时，可启用 `cache_frozen_features` 复用当前批次的冻结骨干特征。

## 代码入口

| 功能 | 实现 |
| --- | --- |
| 数据校验与编译 | [visionjev/data](../visionjev/data/) |
| 骨干、决策头和参数组 | [visionjev/modeling/model.py](../visionjev/modeling/model.py) |
| SFT、RLCD 和恢复 | [visionjev/training](../visionjev/training/) |
| 推理、指标和分布式评估 | [visionjev/evaluation](../visionjev/evaluation/) |

`python -m visionjev.train`、`python -m visionjev.inference` 和 `python -m visionjev.evaluate` 是稳定的命令入口。`VisionJev` 旧类名保留为 `VisualJev` 的兼容别名，checkpoint 参数键保持不变。
