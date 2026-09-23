"""Two-GPU RLCD: real multimodal forwards, fixed reference and exact resume."""
import json
from pathlib import Path

import torch

from multi_gpu_smoke import weight_digest
from visualjev.training.distributed import initialize, close
from visualjev.training.runner import run


def main():
    context = initialize("cuda")
    if context.world_size != 2:
        raise ValueError("This fixture requires two GPUs")
    root = Path("output/rlcd_gpu_smoke")
    root.mkdir(parents=True, exist_ok=True)
    if context.primary:
        (root / "result.json").write_text('{"status": "RUNNING"}\n')
    config = json.loads(Path("configs/train/sft_warmup.json").read_text())
    config.update(output=str(root / "sft"), method="sft", tokens_per_step=900,
                  epochs=2, max_steps=1, save_every=1, head_lr=1e-6)
    model, _, _ = run(config)
    del model
    torch.cuda.empty_cache()
    initial = root / "sft/latest"
    config = json.loads(Path("configs/train/sft_joint.json").read_text())
    config.update(output=str(root / "resumed"), method="rlcd", tokens_per_step=900,
                  epochs=2, max_steps=1, save_every=1, head_lr=1e-5,
                  rlcd={"group_size": 16, "num_iterations": 2})
    model, _, _ = run(config, initialize=initial)
    del model
    torch.cuda.empty_cache()
    config["max_steps"] = 2
    model, compiler, progress = run(config, resume=root / "resumed/latest")
    assert progress["step"] == 2 and progress["optimizer_steps"] == 4
    digest = weight_digest(model)
    assert len(set(context.gather(digest))) == 1
    # Reloaded RLCD checkpoints remain compatible with ordinary inference.
    from visualjev.evaluation.inference import predict
    from visualjev.data.schema import read_jsonl
    records = read_jsonl(config["data"])
    prediction = predict(model, compiler.compile(records[0]))
    assert {answer["type"] for answer in prediction["answers"].values()} == {"choice", "noul", "score"}
    del model
    torch.cuda.empty_cache()
    baseline = dict(config, output=str(root / "continuous"))
    model, _, _ = run(baseline, initialize=initial)
    assert digest == weight_digest(model), "Resumed weights differ from continuous RLCD"
    assert len(set(context.gather(weight_digest(model)))) == 1
    del model
    torch.cuda.empty_cache()
    if context.primary:
        rows = [json.loads(s) for s in (root / "resumed/metrics.jsonl").read_text().splitlines()]
        assert any(rank["states"] == 0 for row in rows for rank in row["ranks"])
        assert any(row["zero_advantage_group"] < .99 for row in rows), "Fixture did not exercise policy gradients"
        assert all(rows[0]["group_grad_norms"][name] > 0 for name in ("head", "lora", "merger"))
        a = torch.load(root / "resumed/latest/checkpoint.pt", weights_only=False, map_location="cpu")
        b = torch.load(root / "continuous/latest/checkpoint.pt", weights_only=False, map_location="cpu")
        for name, value in a["training_state"]["reference_weights"].items():
            torch.testing.assert_close(value, b["training_state"]["reference_weights"][name], rtol=0, atol=0)
        report = {"status": "PASS", "gpu": torch.cuda.get_device_name(), "world_size": 2,
                  "method": "rlcd", "stage": "joint", "weight_sha256": digest,
                  "checks": ["choice_noul_score", "text_image_video", "head_lora_merger_gradients",
                             "nonzero_group_advantages", "idle_rank", "identical_rank_weights", "exact_resume", "fixed_reference", "inference"]}
        (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print("VISUALJEV_RLCD_GPU_SMOKE_PASS", flush=True)
    context.barrier()


if __name__ == "__main__":
    try:
        main()
    finally:
        close()
