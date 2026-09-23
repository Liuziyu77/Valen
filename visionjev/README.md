# Visual-Jev 代码目录

| 目录 | 内容 |
| --- | --- |
| `data/` | JSONL 校验、媒体处理、候选与问题编译 |
| `modeling/` | Qwen 骨干、决策头、LoRA 和参数分组 |
| `training/` | SFT、RLCD、共用训练循环、分布式同步和 checkpoint |
| `evaluation/` | 推理输出、评估指标、单卡与多卡评估流程 |

顶层 `train.py`、`inference.py`、`evaluate.py` 只保留命令入口，现有 `python -m visionjev.<入口>` 用法不变。Python 导入使用对应子目录，例如 `visionjev.data.compiler.Compiler`。

运行前在仓库根目录激活已安装依赖的 Python 环境，环境准备和可直接使用的命令见[脚本说明](../scripts/README.md)。

训练目标和启动方式见 [SFT 与 RLCD](training/README.md)。

模型类使用 `visionjev.modeling.model.VisualJev`，推理响应的 `model` 字段为 `VisualJev`，适用于 0.8B 和 2B 骨干。旧类名保留为同一类的导入别名；参数键和 checkpoint 格式不变。包名 `visionjev`、基础模型的 `visionjev_manifest.json` 及已有数据路径保留兼容。
