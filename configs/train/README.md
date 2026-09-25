# 训练配置

`qwen/` 保存 Qwen 的 SFT/RLCD 阶段配置，`dual_encoder/` 按文本和视觉基座组合保存配置。路径相对于仓库根目录解析，默认参数面向流程检查，正式实验需要修改数据、训练预算和输出目录。

新配置使用 `config_version: 2`，分为四个部分：

| 部分 | 内容 |
| --- | --- |
| `model` | architecture、基座路径、模型维度和计算精度 |
| `data` | path、序列长度、图像尺寸和输入预算 |
| `training` | stage、设备、学习率、步数、输出与保存设置 |
| `objective` | method、SFT 正则项或 rlcd 参数 |

```bash
python -m valen.train --config configs/train/qwen/sft_warmup.json
python -m valen.train --config configs/train/dual_encoder/modernbert_dinov3b16/sft_warmup.json
```

运行前会将四个部分展开为平铺配置，`data.path` 转为原有的 `data` 字符串。checkpoint 保存该运行配置，便于兼容旧 checkpoint；同名字段在多个位置给出不同值时直接报错。CLI 的 `--method` 和 `--output` 在展开后覆盖。

本目录原有的平铺 JSON 保留为兼容入口。测试会检查它们与新目录中对应配置的语义一致；调整默认配方时应同步更新两份。旧命令可以继续运行。

stage 的含义由架构决定：Qwen 的 warmup 只训练决策头，text 使用纯文本和 LoRA；双编码器的 warmup 训练投影、融合和 reader，text/joint 解冻文本塔末层，vision_top 进一步解冻视觉塔末层。每次运行的实际适配方式写入 `run_manifest.json` 的 `adaptation`。
