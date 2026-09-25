"""Actual pretrained model integration checks on one GPU."""
import argparse
import json
from pathlib import Path
import random
import torch
from valen.training.checkpoint import load_checkpoint
from valen.evaluation.inference import predict
from valen.modeling.factory import build_model
from valen.data.schema import read_jsonl
from valen.training.runner import run
from valen.configuration import flatten_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/qwen/sft_joint.json")
    parser.add_argument("--output", default="artifacts/gpu_smoke")
    args = parser.parse_args()
    root = Path(args.output)
    if (root / "result.json").exists():
        raise ValueError("Choose a fresh output directory")
    result_path = root / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text('{"status": "RUNNING"}\n', encoding="utf-8")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke requires CUDA")
    print("GPU:", torch.cuda.get_device_name(), flush=True)
    config = flatten_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
    config.update(output=str(root / "joint"), epochs=4, max_steps=2, tokens_per_step=16384,
                  rps_weight=.2, save_every=1)
    model, compiler, progress = run(config)
    assert progress["step"] == 2
    from safetensors import safe_open
    with safe_open(Path(config["model_path"]) / "model.safetensors-00001-of-00001.safetensors", framework="pt") as weights:
        expected = weights.get_slice("model.language_model.embed_tokens.weight")[:8].to(torch.bfloat16)
    actual = model.backbone.get_input_embeddings().weight[:8].detach().cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    metrics = [json.loads(line) for line in (Path(config["output"]) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    for name in ("head", "lora", "merger"):
        assert metrics[0]["group_grad_norms"][name] > 0, f"No gradients in {name}"
    records = read_jsonl(config["data"])
    model.eval()
    predictions = [predict(model, compiler.compile(r)) for r in records]
    compiled = compiler.compile(records[0])
    with torch.no_grad():
        before = [model(q).cpu() for q in compiled.questions]
    checkpoint = Path(config["output"]) / "latest"
    payload = torch.load(checkpoint / "checkpoint.pt", weights_only=False, map_location="cpu")
    # LoRA B starts at zero; this proves language adapters updated.
    assert any(t.abs().sum() > 0 for n, t in payload["weights"].items() if "lora_B" in n)
    del model
    torch.cuda.empty_cache()
    restored = build_model(config)
    load_checkpoint(checkpoint, restored)
    restored.eval()
    with torch.no_grad():
        after = [restored(q).cpu() for q in compiled.questions]
    for expected, actual in zip(before, after):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    del restored
    torch.cuda.empty_cache()
    config["max_steps"] = 3
    model, _, progress = run(config, resume=checkpoint)
    assert progress["step"] == 3
    del model
    torch.cuda.empty_cache()
    # Label-supervised stages: head warmup, text LoRA, then multimodal SFT.
    warmup = dict(config, stage="warmup", data="data/smoke/text.jsonl", output=str(root / "warmup"), max_steps=1)
    model, _, _ = run(warmup)
    assert all(n.startswith("head.") for n, p in model.named_parameters() if p.requires_grad)
    del model
    torch.cuda.empty_cache()
    text_config = dict(warmup, stage="text", output=str(root / "text"))
    model, _, _ = run(text_config, initialize=root / "warmup/latest")
    del model
    torch.cuda.empty_cache()
    joint_config = dict(config, output=str(root / "joint_initialized"), max_steps=1)
    model, _, _ = run(joint_config, initialize=root / "text/latest")
    del model
    (root / "predictions.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions), encoding="utf-8")
    torch.cuda.empty_cache()
    rl = dict(warmup, method="rlcd", output=str(root / "rlcd"), rps_weight=0.,
              rlcd={"group_size": 4, "num_iterations": 2, "beta": .02}, cache_frozen_features=True)
    model, _, _ = run(rl, initialize=root / "warmup/latest")
    del model
    torch.cuda.empty_cache()
    model, _, progress = run(dict(rl, max_steps=2), resume=root / "rlcd/latest")
    assert progress["optimizer_steps"] == 4
    del model
    report = {"status": "PASS", "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "tasks": ["choice", "noul", "score"], "modalities": ["text", "image", "video"],
              "checks": ["pretrained_embedding_exact", "joint_forward_backward", "finite_gradients", "all_trainable_groups_receive_gradients", "lora_updated", "save_reload_exact", "optimizer_resume", "head_warmup", "text_sft", "joint_sft_initialization", "rlcd_cached_reference", "rlcd_resume"]}
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("VALEN_GPU_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    main()
