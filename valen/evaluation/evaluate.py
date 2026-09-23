"""Evaluate labeled decisions once, sharded without padding across GPUs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from valen.training.checkpoint import load_checkpoint
from valen.data.compiler import Compiler
from valen.training.distributed import initialize, close
from valen.evaluation.inference import answer
from valen.modeling.model import build_model
from valen.data.schema import read_jsonl
from .metrics import question_metrics, summarize


@torch.no_grad()
def run(checkpoint, data, output, device="cuda"):
    run_started = time.monotonic()
    context = initialize(device)
    output, data, checkpoint = Path(output), Path(data), Path(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    config.update(device=context.device, gradient_checkpointing=False)
    torch.manual_seed(config.get("seed", 42))
    model = build_model(config)
    payload = load_checkpoint(checkpoint, model)
    model.eval()
    from transformers import AutoProcessor
    compiler = Compiler(AutoProcessor.from_pretrained(config["model_path"], local_files_only=True),
                        data.parent, config.get("max_length", 8192), config.get("media_kwargs"))
    records = read_jsonl(data)
    indices = list(range(context.rank, len(records), context.world_size))
    context.barrier()
    rows, started = [], time.monotonic()
    def synchronize():
        if context.device.startswith("cuda"):
            torch.cuda.synchronize()
    path = output / f"predictions.rank{context.rank}.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for local_index, index in enumerate(indices):
            record = records[index]
            compile_started = time.monotonic()
            compiled = compiler.compile(record, labeled_only=True)
            compile_seconds = time.monotonic() - compile_started
            for question in compiled.questions:
                synchronize()
                forward_started = time.monotonic()
                logits = model(question)
                synchronize()
                forward_seconds = time.monotonic() - forward_started
                meta = record.get("meta", {})
                row = {"record_index": index, "record_id": meta.get("record_id"),
                       "group_id": record["group_id"], "qid": question.qid, "task": question.kind,
                       "modality": meta.get("modality", "text" if question.is_text else "media"),
                       "language": meta.get("language_bucket", "unknown"),
                       "domain": meta.get("domain", "unknown"),
                       "timing": {"compile_seconds": compile_seconds, "forward_seconds": forward_seconds,
                                  "compile_plus_forward_seconds": compile_seconds + forward_seconds},
                       "metrics": question_metrics(question, logits)}
                rows.append(row)
                probabilities = logits.float().softmax(-1).cpu().tolist()
                prediction = dict(row, target=dict(zip(question.keys, question.target)),
                                  probabilities=dict(zip(question.keys, probabilities)),
                                  answer=answer(question, logits))
                stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            if (local_index + 1) % 50 == 0 or local_index + 1 == len(indices):
                stream.flush()
                print(json.dumps({"event": "evaluation_progress", "rank": context.rank,
                                  "records": local_index + 1, "total": len(indices),
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    rank_timing = context.gather({"rank": context.rank, "records": len(indices),
                                 "evaluation_seconds": time.monotonic() - started,
                                 "load_and_evaluation_seconds": time.monotonic() - run_started,
                                 "device": torch.cuda.get_device_name() if context.device.startswith("cuda") else "cpu"})
    gathered = context.gather(rows)
    if context.primary:
        all_rows = [row for rank_rows in gathered for row in rank_rows]
        # Check coverage without recompiling media.
        from valen.data.schema import candidates, target_distribution
        expected = {(i, qid) for i, record in enumerate(records)
                    for qid, q in record["request"]["questions"].items()
                    if target_distribution(record.get("targets", {}).get(qid), [k for k, _ in candidates(q)]) is not None}
        actual = [(row["record_index"], row["qid"]) for row in all_rows]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Evaluation duplicated or omitted labeled questions")
        report = {"checkpoint": str(checkpoint), "training_step": payload["progress"]["step"],
                  "base_model_path": config["model_path"], "seed": config.get("seed", 42),
                  "training_epoch": payload["progress"]["epoch"], "data": str(data),
                  "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                  "checkpoint_sha256": hashlib.sha256((checkpoint / "checkpoint.pt").read_bytes()).hexdigest(),
                  "records": len(records), "questions": len(all_rows), "world_size": context.world_size,
                  "temperature": 1.0, "elapsed_seconds": time.monotonic() - started,
                  "rank_timing": rank_timing,
                  "evaluation_wall_seconds": max(r["evaluation_seconds"] for r in rank_timing),
                  "load_and_evaluation_wall_seconds": max(r["load_and_evaluation_seconds"] for r in rank_timing),
                  "metrics": summarize(all_rows)}
        report["questions_per_second"] = len(all_rows) / report["evaluation_wall_seconds"]
        report["latency_seconds"] = {}
        for name in ("compile_seconds", "forward_seconds", "compile_plus_forward_seconds"):
            values = torch.tensor([row["timing"][name] for row in all_rows], dtype=torch.float64)
            report["latency_seconds"][name] = {"mean": float(values.mean()), "p50": float(values.quantile(.5)),
                                               "p95": float(values.quantile(.95))}
        (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        predictions = []
        for rank in range(context.world_size):
            predictions.extend(json.loads(line) for line in (output / f"predictions.rank{rank}.jsonl").read_text(encoding="utf-8").splitlines())
        predictions.sort(key=lambda row: (row["record_index"], row["qid"]))
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in predictions:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"event": "evaluation_complete", "questions": len(all_rows),
                          "metrics_path": str(output / "metrics.json")}), flush=True)
    context.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        run(args.checkpoint, args.data, args.output, args.device)
    finally:
        close()


if __name__ == "__main__":
    main()
