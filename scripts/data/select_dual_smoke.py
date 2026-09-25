"""Select a deterministic English, single-image/text training smoke subset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from valen.data.parallel_compiler import build_parallel_compiler


def select(config, source, output, per_type=4):
    source, output = Path(source), Path(output)
    if per_type <= 0:
        raise ValueError("per_type must be positive")
    if source.resolve() == output.resolve():
        raise ValueError("Subset output must not overwrite the source")
    if source.resolve().parent != output.resolve().parent:
        raise ValueError("Keep the subset beside its source so media paths remain valid")
    compiler = build_parallel_compiler(config, source.parent)
    counts, rejected, selected, groups = Counter(), Counter(), [], set()
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("meta", {}).get("language_bucket") != "en" or row["group_id"] in groups:
                continue
            kinds = {q["type"] for q in row["request"]["questions"].values()}
            if not any(counts[k] < per_type for k in kinds):
                continue
            try:
                state = compiler.compile(row, labeled_only=True)
            except (ValueError, FileNotFoundError) as exc:
                rejected[type(exc).__name__] += 1
                continue
            if not state.questions:
                continue
            selected.append(row)
            groups.add(row["group_id"])
            counts.update(q.kind for q in state.questions)
            if all(counts[k] >= per_type for k in ("choice", "noul", "score")):
                break
    if not all(counts[k] >= per_type for k in ("choice", "noul", "score")):
        raise ValueError(f"Insufficient coverage: {counts}")
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    report = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "subset_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "states": len(selected),
              "questions_by_type": dict(counts), "rejected": dict(rejected), "purpose": "training code smoke test, not evaluation"}
    output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/dual_encoder_sft.json")
    parser.add_argument("--source", default="data/train_v1/train.jsonl")
    parser.add_argument("--output", default="data/train_v1/smoke_dual.jsonl")
    parser.add_argument("--per-type", type=int, default=4)
    args = parser.parse_args()
    select(json.loads(Path(args.config).read_text()), args.source, args.output, args.per_type)


if __name__ == "__main__":
    main()
