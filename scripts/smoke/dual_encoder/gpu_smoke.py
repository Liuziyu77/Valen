"""Single-GPU SFT/resume/RLCD/inference check on real English training records.

--tiny uses randomly initialized small HF encoders; it tests code, not quality.
Without --tiny, the configured pretrained encoder files must be available locally.
"""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))
import torch
from select_dual_smoke import select
from valen.data.parallel_compiler import build_parallel_compiler
from valen.evaluation.inference import predict
from valen.training.runner import run
from valen.configuration import flatten_config


def random_encoders(config, root):
    from transformers import (AutoTokenizer, ModernBertConfig, ModernBertModel,
                              DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessor)
    text, vision = root / "random-text", root / "random-vision"
    tokenizer = AutoTokenizer.from_pretrained(config["text_model_path"], local_files_only=True)
    tokenizer.save_pretrained(text)
    ModernBertModel(ModernBertConfig(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                   num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=1024,
                   local_attention=128, global_attn_every_n_layers=2, pad_token_id=tokenizer.pad_token_id,
                   cls_token_id=tokenizer.cls_token_id, sep_token_id=tokenizer.sep_token_id)).save_pretrained(text)
    DINOv3ViTModel(DINOv3ViTConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, image_size=384, patch_size=16, num_register_tokens=4,
                  pos_embed_shift=None, pos_embed_jitter=None, pos_embed_rescale=None)).save_pretrained(vision)
    DINOv3ViTImageProcessor().save_pretrained(vision)
    return dict(config, text_model_path=str(text), vision_model_path=str(vision),
                model_name="Valen-DE-random-smoke-only", text_unfreeze_layers=1, vision_unfreeze_layers=1)


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/dual_encoder/modernbert_dinov3b16/sft_warmup.json")
    parser.add_argument("--source", default="data/train_v1/train.jsonl")
    parser.add_argument("--output", default="artifacts/dual_encoder_smoke")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "report.json").exists() or (root / "sft").exists():
        raise ValueError("Choose a fresh smoke output directory to preserve previous runs")
    torch.manual_seed(42)
    config = flatten_config(json.loads(Path(args.config).read_text()))
    if args.tiny:
        config = random_encoders(config, root)
    config.update(device=args.device, epochs=4, max_steps=1, save_every=1,
                  output=str(root / "sft"), method="sft")
    source = Path(args.source).resolve()
    subset = source.with_name("smoke_dual_tiny.jsonl" if args.tiny else "smoke_dual_pretrained.jsonl")
    rows = select(config, source, subset, per_type=2)
    config["data"] = str(subset)
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model, compiler, sft = run(config)
    del model, compiler
    release()
    model, compiler, sft = run(dict(config, max_steps=2), resume=root / "sft/latest")
    assert sft["step"] == 2
    del model, compiler
    release()
    rl = dict(config, output=str(root / "rlcd"), method="rlcd", brier_weight=0., rps_weight=0.,
              rlcd={"group_size": 8, "num_iterations": 2, "beta": .02, "brier_weight": 1.})
    model, compiler, progress = run(rl, initialize=root / "sft/latest")
    del model, compiler
    release()
    model, compiler, progress = run(dict(rl, max_steps=2), resume=root / "rlcd/latest")
    assert progress["optimizer_steps"] == 4
    compiler = build_parallel_compiler(config, subset.parent)
    row = deepcopy(next(r for r in rows if any(q["type"] == "noul" for q in r["request"]["questions"].values())))
    question = next(q for q in row["request"]["questions"].values() if q["type"] == "noul")
    row["targets"] = {}
    row["request"]["questions"] = {"q0": question}
    single = compiler.compile(row)
    row["request"]["questions"] = {f"q{i}": deepcopy(question) for i in range(256)}
    row["request"]["questions"]["q128"]["instructions"] = "Is the scene completely blue?"
    state = compiler.compile(row)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    inference_started = time.monotonic()
    result = predict(model, state)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    inference_seconds = time.monotonic() - inference_started
    expected = predict(model, single)["answers"]["q0"]["noul"]
    # Batched BF16 kernels need not be bitwise equal to a single-question kernel.
    difference = max(abs(result["answers"][f"q{i}"]["noul"] - expected) for i in (0, 64, 255))
    assert difference < .01, difference
    (root / "prediction_256.json").write_text(json.dumps(result, indent=2) + "\n")
    report = {"random_encoders": args.tiny, "device": args.device, "states": len(rows),
              "sft_steps": sft["step"], "rlcd_optimizer_steps": progress["optimizer_steps"],
              "questions": len(result["answers"]), "max_independence_probability_difference": difference,
              "inference_256_seconds_unwarmed": inference_seconds, "total_seconds": time.monotonic() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
              "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
              "note": "Code smoke test only; no accuracy or deployment latency claim."}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
