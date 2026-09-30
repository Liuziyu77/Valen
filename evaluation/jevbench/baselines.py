"""Native Laya and Intern-Decision HF evaluation on pinned public JevBench.

The author packages are external dependencies supplied with --source. Their
prompt compilers, probability decoding and shipped defaults are used unchanged.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

from .dataset import DEFAULT_FILES, DEFAULT_ROOT, REVISION, file_hash, load_dataset, write_json
from .evaluate import read_predictions, runtime_versions, source_hashes
from .metrics import summarize_run
from ._upstream.scoring import score_task

BASELINE_MAPPING_VERSION = "native-typed-question-v1"


def native_request(task):
    """Only inference fields enter the native API; keep evidence/option order."""
    question = {k: copy.deepcopy(task.question[k])
                for k in ("type", "instructions", "criteria", "labels") if k in task.question}
    return {"state": copy.deepcopy(task.state), "questions": {"decision": question}}


def native_probabilities(task, response):
    answer = response["answers"]["decision"]
    if answer.get("type") != task.question["type"]:
        raise ValueError("Native answer type differs from task")
    if task.question["type"] == "noul":
        p = float(answer["noul"])
        if not 0.0 <= p <= 1.0:
            raise ValueError("Native Noul probability is outside [0,1]")
        return {"no": 1.0 - p, "yes": p}
    if not isinstance(answer["probabilities"], dict):
        raise ValueError("Missing native probability object")
    return {str(k): float(v) for k, v in answer["probabilities"].items()}


def fingerprint_model(model):
    """Hash actual weights and inference metadata; verify any HF LFS hashes."""
    model = Path(model)
    paths = sorted(p for p in model.rglob("*") if p.is_file() and ".cache" not in p.parts
                   and p.suffix in {".safetensors", ".json", ".jinja", ".txt", ".py"}
                   and p.name != "download_manifest.json")
    hashes = {str(p.relative_to(model)): file_hash(p) for p in paths}
    if not any(p.endswith(".safetensors") for p in hashes):
        raise ValueError("No downloaded checkpoint weights")
    metadata = model / "download_manifest.json"
    download = json.loads(metadata.read_text()) if metadata.exists() else None
    if download:
        for item in download["files"]:
            path = model / item["path"]
            if not path.is_file() or path.stat().st_size != item["size"]:
                raise ValueError(f"Downloaded file size mismatch: {path}")
            lfs = item.get("lfs")
            if lfs and item["path"] in hashes and hashes[item["path"]] != lfs["sha256"]:
                raise ValueError(f"Downloaded file hash mismatch: {path}")
    return {"path": str(model.resolve()), "files_sha256": hashes, "download": download}


def fingerprint_source(source):
    source = Path(source).resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    return {"path": str(source), "revision": revision,
            "files_sha256": {str(p.relative_to(source)): file_hash(p)
                             for p in sorted(source.rglob("*.py")) if ".git" not in p.parts}}


def load_adapter(adapter, model, source, device):
    sys.path.insert(0, str(Path(source).resolve()))
    if adapter == "laya":
        import laya
        agent = laya.load(str(model), device=device, fast=False, compile=False)
        if str(agent.device) != str(torch.device(device)):
            raise ValueError(f"Laya fell back to {agent.device}")
        settings = {"package_version": laya.__version__, "max_length": agent.cfg["max_len"],
                    "head_max_length": agent.cfg["head_max_len"], "dtype": str(agent.dtype_for(1)),
                    "temperature": agent.temperature, "temperature_by_options": agent.temperature_by_options,
                    "fast": False, "compile": False, "truncation": "native package defaults"}

        def predict(request):
            response = agent.predict(request["state"], request["questions"])
            if agent.cpu_fallback_count:
                raise RuntimeError("Native Laya used a CPU fallback during GPU measurement")
            return response
        return predict, settings
    if adapter == "intern_hf":
        # Use the inference module shipped with these pinned weights. It includes
        # the checkpoint's own default temperature, unlike older GitHub examples.
        name = "_jevbench_intern_release"
        spec = importlib.util.spec_from_file_location(name, model / "inference.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        dtype = "bfloat16" if str(device).startswith("cuda") else "float32"
        engine = module.DecisionEngine(str(model), temperature=1.0, backend="hf", device=device,
                                       dtype=dtype, max_length=8192, attn_implementation="sdpa")

        def predict(request):
            raw = engine.predict(request)
            # Identical operation/order to the released wrapper, retaining T=1
            # for a separate report without another model forward or test fitting.
            response = module.scale_result(raw, module.DEFAULT_TEMPERATURE)
            response["uncalibrated_response"] = raw
            return response
        return predict, {"backend": "hf", "dtype": dtype, "max_length": 8192,
                         "attn_implementation": "sdpa", "temperature": module.DEFAULT_TEMPERATURE,
                         "calibration": "shipped model default; raw T=1 also retained",
                         "truncation": "forbidden by native backend"}
    raise ValueError(f"Unknown baseline adapter: {adapter}")


def run(adapter, model, source, dataset, output, device="cuda", files=DEFAULT_FILES,
        limit=None, warmup=1, resume=False):
    if warmup < 0 or limit is not None and limit <= 0:
        raise ValueError("warmup must be nonnegative; limit must be positive")
    tasks, data_manifest = load_dataset(dataset, files)
    if limit:
        tasks = tasks[:limit]
    model, source, output = map(lambda p: Path(p).resolve(), (model, source, output))
    hardware = torch.cuda.get_device_name(torch.device(device)) if str(device).startswith("cuda") else platform.processor() or "cpu"
    # Record files before loading; author packages may repair legacy tokenizer metadata.
    model_manifest = fingerprint_model(model)
    torch.manual_seed(42)
    load_started = time.perf_counter()
    predict, settings = load_adapter(adapter, model, source, device)
    load_seconds = time.perf_counter() - load_started
    after_load = fingerprint_model(model)
    if after_load != model_manifest:
        raise ValueError("Native loading changed checkpoint files; review before evaluating")
    identity = model_manifest["download"]["repo_id"] if model_manifest["download"] else model.name
    manifest = {"benchmark_revision": REVISION, "mapping_version": BASELINE_MAPPING_VERSION,
                "adapter": adapter, "model": model_manifest, "native_source": fingerprint_source(source),
                "runtime_config": settings, "dataset": data_manifest,
                "task_ids_sha256": hashlib.sha256("\n".join(t.id for t in tasks).encode()).hexdigest(),
                "planned_tasks": len(tasks), "limit": limit, "warmup_per_type": warmup,
                "hardware": hardware, "code_sha256": source_hashes(),
                "versions": runtime_versions(),
                "timing_scope": "local sequential native predict + answer conversion; excludes model loading and scoring"}
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        if not resume or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing run requires --resume and an identical manifest")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty")
        output.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
    task_ids = {t.id for t in tasks}
    path = output / "predictions.jsonl"
    records = read_predictions(path, task_ids)
    done = {r["task_id"] for r in records}

    def synchronize():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))

    def infer(task):
        synchronize()
        started = time.perf_counter()
        request = native_request(task)
        response = predict(request)
        native = native_probabilities(task, response)
        synchronize()
        return native, response, request, time.perf_counter() - started

    examples = {}
    for task in tasks:
        examples.setdefault(task.question["type"], task)
    with torch.inference_mode():
        for _ in range(warmup):
            for task in examples.values():
                infer(task)
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        started = time.perf_counter()
        with path.open("a", encoding="utf-8") as stream:
            for task in tasks:
                if task.id in done:
                    continue
                item_started = time.perf_counter()
                row = {"task_id": task.id, "family": task.family, "split": task.split,
                       "group": task.group, "tier": data_manifest["tiers"][task.id],
                       "type": task.question["type"], "model": identity, "expected": task.expected,
                       "cost_usd": None, "cost_basis": "unmeasured_local_gpu", "probs_source": "native"}
                try:
                    native, response, request, latency = infer(task)
                    row.update(score_task(native, task), ok=True, status="ok", latency_s=latency,
                               probs_as_returned=native, response=response, request=request,
                               usage=response.get("usage", {}), timing={})
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
                    print(json.dumps({"event": "jevbench_baseline_progress", "model": identity,
                                      "attempted": len(records), "planned": len(tasks),
                                      "failures": sum(not r["ok"] for r in records)}), flush=True)
    if {r["task_id"] for r in records} != task_ids or len(records) != len(task_ids):
        raise ValueError("Evaluation coverage mismatch")
    positions = {t.id: i for i, t in enumerate(tasks)}
    records.sort(key=lambda r: positions[r["task_id"]])
    report = summarize_run(tasks, records, data_manifest["tiers"])
    truncated = [r["task_id"] for r in records if r.get("usage", {}).get("truncated")]
    report.update(model=identity, adapter=adapter, checkpoint=str(model), runtime_config=settings,
                  benchmark_revision=REVISION, dataset_sha256=data_manifest["canonical_sha256"],
                  hardware=hardware, load_seconds=load_seconds,
                  session_evaluation_seconds=time.perf_counter() - started,
                  smoke_subset=limit is not None and limit < data_manifest["tasks"],
                  warmup_calls=warmup * len(examples), run_manifest="run_manifest.json",
                  truncation={"state_truncated": len(truncated), "task_ids": truncated},
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(torch.device(device)) if str(device).startswith("cuda") else None)
    write_json(output / "metrics.json", report)
    if adapter == "intern_hf":
        raw_records = []
        for task, row in zip(tasks, records):
            raw_row = copy.deepcopy(row)
            if row.get("ok"):
                response = row["response"]["uncalibrated_response"]
                native = native_probabilities(task, response)
                raw_row.update(score_task(native, task), probs_as_returned=native, response=response)
                if raw_row["predicted"] != row["predicted"]:
                    raise ValueError("Shipped calibration changed argmax")
            raw_records.append(raw_row)
        raw_report = summarize_run(tasks, raw_records, data_manifest["tiers"])
        raw_report.update({k: v for k, v in report.items() if k not in raw_report})
        raw_report.update(runtime_config={**settings, "temperature": 1.0, "calibration": None},
                          probability_variant="uncalibrated", changed_argmax=0)
        write_json(output / "uncalibrated_metrics.json", raw_report)
        (output / "uncalibrated_predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in raw_records), encoding="utf-8")
    write_json(output / "complete.json", {"tasks": len(tasks), "failed": sum(not r["ok"] for r in records)})
    print(json.dumps({"event": "jevbench_baseline_complete", "model": identity,
                      "accuracy": report["accuracy"], "output": str(output)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=("laya", "intern_hf"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--files", nargs="+", choices=DEFAULT_FILES, default=DEFAULT_FILES)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.adapter, args.model, args.source, args.dataset, args.output,
        args.device, args.files, args.limit, args.warmup, args.resume)


if __name__ == "__main__":
    main()
