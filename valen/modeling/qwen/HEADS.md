# Qwen decision heads / Qwen 决策头

`head_type` selects a complete candidate scorer. The default `bilinear` preserves
existing checkpoint parameter names. All heads return one FP32 logit per candidate.

| head_type | 输入 / Inputs | 默认结构 / Defaults |
| --- | --- | --- |
| `bilinear` | Decision + candidate endpoint | Two 256-dimensional projections and dot product |
| `mlp` | Decision + candidate endpoint | Separate LayerNorm, concatenate, 512 → 128 → 1 with GELU |
| `role_mlp` | Context, instruction, candidate description, Decision | Shared projection to 256, concatenate four roles, 1024 → 1 with GELU |
| `mixer` | Same four roles | Two blocks, token MLP 4 → 16 → 4, channel MLP 256 → 1024 → 256, candidate-role readout |

四角色头先对编译器标记的上下文、问题和候选描述区间求均值，第四个角色取 Decision 位置。
上下文包含公共 state 的文本和视觉位置；这些都是 Qwen 处理后的隐状态。平均池化可能丢失空间细节。
候选名称继续进入原有提示词，候选描述池化使用描述所覆盖的 token；边界 token 可能同时覆盖空格或分隔符。
默认 `qwen_execution="question"` 保持原有分词序列和 Score 每等级独立分支。
`shared_state` 模式将所有问题及候选编入同一序列，各题保留各自的角色区间与 Decision 位置，
四种评分头均可使用。Score 的全部等级共享一次 forward，详见[共享前向说明](../../../docs/qwen-shared-state.md)。

Candidate rows share parameters and never mix inside the head. This does not make
the full Qwen model invariant to option order: causal backbone states already depend
on their prefix. Mixer token mixing operates over four semantic roles, not options.

`head_signature()` validates architecture dimensions and rejects incompatible stage
initializations. Backbone feature caching stores raw readouts before any trainable
normalization or projection, so RLCD policy/reference updates remain valid.

## Configuration / 配置

在现有 Qwen 训练配置的 `model` 段中设置 `head_type`，例如：

```json
{
  "head_type": "mixer",
  "head_width": 256,
  "head_layers": 2,
  "head_token_hidden_dim": 16,
  "head_channel_hidden_dim": 1024
}
```

这段示例只包含评分头参数，需合并到已有训练配置中。保留其余模型、数据和训练字段，然后使用通用 CLI：

```bash
python -m valen.train --config output/my_experiment/warmup.json
python -m valen.train --config output/my_experiment/joint.json \
  --initialize output/my_experiment/warmup/latest
```

`warmup` freezes Qwen and trains the selected head. `joint` trains the same head,
LLM LoRA and the visual merger. Keep the head type and dimensions identical across
stages. Training checkpoints contain trainable deltas and still require the same
base model. Full parameter descriptions are in [configuration.md](../../../docs/configuration.md).
