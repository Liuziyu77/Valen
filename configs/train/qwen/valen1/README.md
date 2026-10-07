# Valen v1 training configs

Two-stage SFT with Qwen3.5, the Mixer decision head, `shared_state`, and FlashAttention 2. Warmup trains the Mixer only; joint SFT updates the entire backbone and Mixer (`finetuning_type: full`).

两阶段 SFT：预热阶段只训练 Mixer，联合微调阶段全参数训练。配置中的路径均相对于仓库根目录。

| Model | Warmup | Joint SFT | Joint microbatch / padded tokens per GPU |
| --- | --- | --- | --- |
| 0.8B | `0.8b_warmup.json` | `0.8b_joint.json` | 32 / 131,072 |
| 2B | `2b_warmup.json` | `2b_joint.json` | 32 / 131,072 |
| 4B | `4b_warmup.json` | `4b_joint.json` | 8 / 32,768 |

The recorded runs use **32 GPUs (4 nodes × 8 H200 GPUs)**. `tokens_per_step` is a per-GPU accumulation budget: 32,768 for warmup and 65,536 for joint SFT, giving nominal global budgets of 1,048,576 and 2,097,152 tokens. These budgets count compiled tokens; microbatch limits also account for padding. Each stage runs one epoch; `max_steps` is an upper bound. The warmup dataset is a subset reused in joint SFT.

4B joint SFT initially used 16 / 65,536 and encountered an out-of-memory error. `4b_joint_initial.json` preserves that configuration; `4b_joint.json` contains the recovery settings. The completed 4B warmup used 32 / 131,072. The recovery keeps the 32-GPU world size, accumulation budget, data, and learning rates unchanged. **The 4B joint run is still in progress when these configs are recorded.**

## Run

From the repository root, prepare the matching base model at `models/Qwen3.5-{0.8B,2B,4B}` and your datasets at `data/valen1/{warmup,train}.jsonl`, or edit those paths in the configs. Relative media paths resolve against the JSONL's parent directory. Dataset contents and machine-specific launch settings are supplied separately.

Set `NODE_RANK` (0–3), `MASTER_ADDR` (node 0's reachable address), and `MASTER_PORT` consistently for all four nodes. Run each stage on all nodes, waiting for warmup to finish before starting joint SFT:

```bash
VALEN_SIZE=2b  # 0.8b / 2b / 4b

torchrun --nnodes=4 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv_id="valen1-${VALEN_SIZE}-warmup" \
  -m valen.train --config "configs/train/qwen/valen1/${VALEN_SIZE}_warmup.json"

torchrun --nnodes=4 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv_id="valen1-${VALEN_SIZE}-joint" \
  -m valen.train --config "configs/train/qwen/valen1/${VALEN_SIZE}_joint.json" \
  --initialize "output/valen1/${VALEN_SIZE}/warmup/latest"
```

To resume an interrupted joint run, replace `--initialize ...` with `--resume "output/valen1/${VALEN_SIZE}/joint/latest"`. Keep the same world size and data/config paths as the checkpoint. For an existing checkpoint created with different absolute paths, use its original runtime config with the documented microbatch changes, or use `--initialize` for a new stage with reset optimizer and progress.

断点续训使用 `--resume`；切换阶段使用 `--initialize`。公开配置已替换本地路径，不能直接作为旧训练任务的原路径断点配置。
