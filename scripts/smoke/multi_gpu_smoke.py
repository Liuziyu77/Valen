"""Two-rank real-model checks: idle tails, synchronization, exact resume."""
import hashlib
import json
from pathlib import Path
import torch
from visionjev.training.distributed import initialize, close
from visionjev.training.runner import run


def weight_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    context = initialize("cuda")
    if context.world_size != 2:
        raise ValueError("This smoke fixture expects exactly two GPUs")
    root = Path("artifacts/multi_gpu_smoke")
    root.mkdir(parents=True, exist_ok=True)
    if context.primary:
        (root / "result.json").write_text('{"status":"RUNNING"}\n', encoding="utf-8")
    config = json.loads(Path("configs/train/sft_joint.json").read_text(encoding="utf-8"))
    config.update(output=str(root / "resumed"), tokens_per_step=900, max_steps=1,
                  epochs=2, rps_weight=.2, save_every=1)
    model, _, progress = run(config)
    assert progress["step"] == 1
    checkpoint = Path(config["output"]) / "latest"
    saved = torch.load(checkpoint / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert saved["distributed"]["world_size"] == 2
    assert saved["distributed"]["rank_states"][0]["progress"]["cursor"] != saved["distributed"]["rank_states"][1]["progress"]["cursor"]
    del model, saved
    torch.cuda.empty_cache()
    config["max_steps"] = 2
    model, _, progress = run(config, resume=checkpoint)
    assert progress["step"] == 2
    resumed_digest = weight_digest(model)
    assert len(set(context.gather(resumed_digest))) == 1
    del model
    torch.cuda.empty_cache()
    baseline = dict(config, output=str(root / "uninterrupted"))
    model, _, _ = run(baseline)
    baseline_digest = weight_digest(model)
    assert len(set(context.gather(baseline_digest))) == 1
    assert resumed_digest == baseline_digest, "Resume differs from uninterrupted training"
    del model
    torch.cuda.empty_cache()
    context.barrier()
    if context.primary:
        metrics = [json.loads(x) for x in (root / "resumed/metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(metrics) == 2
        assert any(r["states"] == 0 for row in metrics for r in row["ranks"]), "Fixture did not exercise idle rank"
        seen = []
        for rank in range(2):
            rows = [json.loads(x) for x in (root / f"resumed/media.rank{rank}.jsonl").read_text(encoding="utf-8").splitlines()]
            seen.extend(r["state_index"] for r in rows if r["epoch"] == 0)
        assert sorted(seen) == [0, 1, 2, 3, 4], f"Duplicate/dropped states: {seen}"
        report = {"status": "PASS", "world_size": 2, "gpu": torch.cuda.get_device_name(),
                  "backend": "nccl", "weight_sha256": resumed_digest,
                  "checks": ["all_ranks_identical_weights", "unequal_state_counts", "idle_rank_tail",
                             "no_duplicate_or_dropped_states", "per_rank_checkpoint", "resume_matches_uninterrupted_exactly"]}
        (root / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("VISIONJEV_MULTI_GPU_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        close()
