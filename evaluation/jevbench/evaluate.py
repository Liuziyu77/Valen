"""Evaluate one local Valen checkpoint on the frozen public JevBench tasks."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import torch

from valen.modeling.factory import build_model, build_compiler, normalize_model_config
from valen.training.checkpoint import load_checkpoint
from .dataset import (DEFAULT_FILES, DEFAULT_ROOT, MAPPING_VERSION, REVISION, file_hash,
                      load_dataset, probabilities, to_record, write_json)
from .metrics import summarize_run
from ._upstream.scoring import score_task


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "valen").rglob("*.py")) + list(Path(__file__).parent.rglob("*.py"))
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(paths)}


def runtime_versions():
    versions = {"python": platform.python_version(), "cuda": torch.version.cuda}
    for package in ("torch", "transformers", "peft", "tokenizers", "safetensors", "numpy",
                    "huggingface-hub", "flash-linear-attention", "fla-core", "causal-conv1d", "flash-attn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def read_predictions(path, task_ids):
    """Recover a interrupted JSONL append; reject corruption inside the journal."""
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    records, seen, complete_bytes = [], set(), 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                break
            raise ValueError(f"Corrupt predictions at line {index + 1}")
        if row["task_id"] not in task_ids or row["task_id"] in seen:
            raise ValueError("Duplicate or unknown task ID in predictions")
        seen.add(row["task_id"])
        records.append(row)
        complete_bytes += len(line)
    with path.open("r+b") as stream:
        stream.truncate(complete_bytes)
        if complete_bytes and not lines[len(records) - 1].endswith(b"\n"):
            stream.seek(0, 2)
            stream.write(b"\n")
    return records


def run(checkpoint, dataset, output, device="cuda", files=DEFAULT_FILES, limit=None,
        max_length=None, warmup=1, resume=False):
    if warmup < 0 or limit is not None and limit <= 0:
        raise ValueError("warmup must be nonnegative; limit must be positive")
    tasks, data_manifest = load_dataset(dataset, files)
    if limit:
        tasks = tasks[:limit]
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    config = normalize_model_config(json.loads((checkpoint / "config.json").read_text(encoding="utf-8")))
    config.update(device=device, gradient_checkpointing=False)
    if max_length is not None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        config["max_length"] = max_length
    if str(device).startswith("cpu"):
        config["dtype"] = "fp32"
    hardware = torch.cuda.get_device_name(torch.device(device)) if str(device).startswith("cuda") else platform.processor() or "cpu"
    manifest = {"benchmark_revision": REVISION, "mapping_version": MAPPING_VERSION,
                "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint / "checkpoint.pt"),
                "checkpoint_config_sha256": file_hash(checkpoint / "config.json"),
                "runtime_config": config, "dataset": data_manifest,
                "task_ids_sha256": hashlib.sha256("\n".join(t.id for t in tasks).encode()).hexdigest(),
                "planned_tasks": len(tasks), "limit": limit, "warmup_per_type": warmup,
                "temperature": 1.0, "hardware": hardware, "code_sha256": source_hashes(),
                "versions": runtime_versions(),
                "timing_scope": "local sequential compile + forward + native answer; excludes model loading and scoring"}
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError("Output already contains a run; choose a new output or use --resume")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume checkpoint, data, source, hardware or settings changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty")
        output.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
    task_ids = {t.id for t in tasks}
    path = output / "predictions.jsonl"
    records = read_predictions(path, task_ids)
    done = {row["task_id"] for row in records}
    load_started = time.perf_counter()
    torch.manual_seed(config.get("seed", 42))
    model = build_model(config)
    payload = load_checkpoint(checkpoint, model)
    model.eval().requires_grad_(False)
    compiler = build_compiler(config, ".")
    load_seconds = time.perf_counter() - load_started
    identity = getattr(model, "model_name", "Valen") + "/" + config.get("head_type", "bilinear") + "/" + config.get("stage", "joint")

    def synchronize():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))

    def infer(task):
        synchronize()
        started = time.perf_counter()
        compiled = compiler.compile(to_record(task))
        compiled_at = time.perf_counter()
        # Capture forward separately while using the same native response path.
        from valen.modeling.factory import backend_for_model
        from valen.evaluation.inference import answer
        decisions = [d for unit in backend_for_model(model).inference_units(model, compiled) for d in unit]
        synchronize()
        forwarded_at = time.perf_counter()
        response = {"answers": {d.question.qid: answer(d.question, d.logits) for d in decisions}}
        native = probabilities(task, response)
        synchronize()
        ended = time.perf_counter()
        return native, {"compile_seconds": compiled_at - started,
                        "forward_seconds": forwarded_at - compiled_at,
                        "answer_seconds": ended - forwarded_at}, ended - started, compiled

    examples = {}
    for task in tasks:
        examples.setdefault(task.question["type"], task)
    for _ in range(warmup):
        for task in examples.values():
            infer(task)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.perf_counter()
    with torch.inference_mode(), path.open("a", encoding="utf-8") as stream:
        for task in tasks:
            if task.id in done:
                continue
            item_started = time.perf_counter()
            row = {"task_id": task.id, "family": task.family, "split": task.split,
                   "group": task.group, "tier": data_manifest["tiers"][task.id],
                   "type": task.question["type"], "model": identity, "cost_usd": None,
                   "cost_basis": "unmeasured_local_gpu", "probs_source": "native",
                   "expected": task.expected}
            try:
                native, timing, latency, compiled = infer(task)
                row.update(score_task(native, task), ok=True, status="ok", latency_s=latency,
                           probs_as_returned=native, timing=timing,
                           usage={"input_tokens": compiled.logical_tokens, "output_tokens": 0},
                           internal_usage={"compute_tokens": compiled.compute_tokens})
            except Exception as exc:
                row.update(ok=False, status="failed", valid=False, strict_valid=False,
                           renormalized=False, correct=False, predicted=None, probs=None,
                           latency_s=time.perf_counter() - item_started,
                           error=type(exc).__name__ + ": " + str(exc))
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            records.append(row)
            if len(records) % 10 == 0 or len(records) == len(tasks):
                print(json.dumps({"event": "jevbench_progress", "model": identity,
                                  "attempted": len(records), "planned": len(tasks),
                                  "failures": sum(not r["ok"] for r in records)}), flush=True)
    if {r["task_id"] for r in records} != task_ids or len(records) != len(task_ids):
        raise ValueError("Evaluation coverage mismatch")
    positions = {t.id: i for i, t in enumerate(tasks)}
    records.sort(key=lambda r: positions[r["task_id"]])
    report = summarize_run(tasks, records, data_manifest["tiers"])
    report.update(model=identity, head=config.get("head_type", "bilinear"), stage=config.get("stage"),
                  checkpoint=str(checkpoint), checkpoint_sha256=manifest["checkpoint_sha256"],
                  benchmark_revision=REVISION, dataset_sha256=data_manifest["canonical_sha256"],
                  training_progress=payload.get("progress"), hardware=hardware,
                  load_seconds=load_seconds, session_evaluation_seconds=time.perf_counter() - started,
                  smoke_subset=limit is not None and limit < data_manifest["tasks"],
                  warmup_calls=warmup * len(examples), run_manifest="run_manifest.json",
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(torch.device(device)) if str(device).startswith("cuda") else None)
    write_json(output / "metrics.json", report)
    write_json(output / "complete.json", {"tasks": len(tasks), "failed": sum(not r["ok"] for r in records),
                                           "checkpoint_sha256": manifest["checkpoint_sha256"]})
    print(json.dumps({"event": "jevbench_complete", "model": identity, "accuracy": report["accuracy"],
                      "output": str(output)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--files", nargs="+", choices=DEFAULT_FILES, default=DEFAULT_FILES)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured warmup calls per question type")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.checkpoint, args.dataset, args.output, args.device, args.files, args.limit,
        args.max_length, args.warmup, args.resume)


if __name__ == "__main__":
    main()
