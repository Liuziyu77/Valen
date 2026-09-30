"""Compare local runs made on the same frozen public task set and hardware."""
import argparse
from itertools import combinations
import json
from pathlib import Path

from .dataset import write_json


def compare(runs, output):
    rows, manifests, predictions = [], [], []
    for path in map(Path, runs):
        report = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        by_id = {r["task_id"]: r for r in records}
        if not report["complete"] or report["smoke_subset"] or len(records) != len(by_id):
            raise ValueError(f"Incomplete, duplicate or smoke run: {path}")
        if len(records) != manifest["planned_tasks"]:
            raise ValueError(f"Prediction count differs from manifest: {path}")
        rows.append(dict(report, run=str(path)))
        manifests.append(manifest)
        predictions.append(by_id)
    if not rows or len({r["model"] for r in rows}) != len(rows):
        raise ValueError("Provide distinct model runs")
    config_keys = ("model_path", "architecture", "dtype", "stage", "max_length", "media_kwargs")
    def conditions(manifest):
        config = {k: manifest["runtime_config"].get(k) for k in config_keys}
        if config["model_path"] is not None:
            config["model_path"] = str(Path(config["model_path"]).resolve())
        return {key: manifest[key] for key in ("benchmark_revision", "mapping_version", "dataset",
                "task_ids_sha256", "warmup_per_type", "temperature", "hardware", "code_sha256", "versions")} | {
                "model_conditions": config}
    if any(conditions(m) != conditions(manifests[0]) for m in manifests[1:]):
        raise ValueError("Runs used different datasets, hardware, source or inference conditions")
    if any(set(p) != set(predictions[0]) for p in predictions[1:]):
        raise ValueError("Runs have different task coverage")
    pairs = []
    for i, j in combinations(range(len(rows)), 2):
        scorable = [key for key in predictions[i] if predictions[i][key].get("expected") is not None]
        pairs.append({"first": rows[i]["head"], "second": rows[j]["head"],
                      "first_only_correct": sum(bool(predictions[i][k]["correct"]) and not predictions[j][k]["correct"] for k in scorable),
                      "second_only_correct": sum(bool(predictions[j][k]["correct"]) and not predictions[i][k]["correct"] for k in scorable)})
    result = {"scope": "public_subset", "benchmark_revision": rows[0]["benchmark_revision"],
              "hardware": rows[0]["hardware"], "tasks": rows[0]["n_planned"],
              "models": rows, "paired_outcomes": pairs, "official_leaderboard_score": None}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", result)
    def value(number, percentage=False, milliseconds=False):
        if number is None:
            return "—"
        return f"{number * 100:.2f}%" if percentage else f"{number * 1000:.1f}" if milliseconds else f"{number:.4f}"
    lines = ["# Qwen four-head JevBench public evaluation", "",
             f"Snapshot: `{result['benchmark_revision']}`; {result['tasks']} public tasks; hardware: {result['hardware']}.", "",
             "| Head | Accuracy | Correct | Brier | ECE | NLL | Mean ms | p50 ms | p95 ms | Failures |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['head']} | {value(row['accuracy'], True)} | {row['n_correct']}/{row['n_scorable']} | "
                     f"{value(row['brier_mean'])} | {value(row['ece']['ece'] if row['ece'] else None)} | "
                     f"{value(row['nll']['mean'])} | {value(row['latency']['mean_s'], milliseconds=True)} | "
                     f"{value(row['latency']['p50_s'], milliseconds=True)} | {value(row['latency']['p95_s'], milliseconds=True)} | "
                     f"{row['n_attempted'] - row['n_valid']} |")
    for name, field in (("Tiers", "per_tier"), ("Question types", "per_type")):
        keys = sorted(rows[0][field])
        lines += ["", "## " + name, "", "| Head | " + " | ".join(keys) + " |",
                  "| --- | " + " | ".join("---:" for _ in keys) + " |"]
        for row in rows:
            lines.append("| " + row["head"] + " | " + " | ".join(value(row[field][k]["accuracy"], True) for k in keys) + " |")
    lines += ["", "## Measurement", "",
              "Each task is a separate request, in the frozen source order. Choice option order is unchanged. "
              "Structured state is serialized as JSON; Noul rubric descriptions are appended to instructions. "
              "Native probabilities use temperature 1, without fitting calibration to these answers.", "",
              "Latency includes local input compilation, model forward and answer conversion. GPU work is synchronized. "
              "Model loading, warmup and scoring are excluded. These measurements have no network hop or production-load adjustment.", "",
              "Score accuracy uses the most probable level, following the pinned upstream scorer; expected-value MAE is separate. "
              "Brier and ECE follow upstream hard-label conventions; ten hard items additionally have exact gold distributions, "
              "whose TVD and distribution Brier are in the JSON report.", "",
              "Public coverage is 48 easy + 72 standard + 111 hard. The sealed items and judge tier are unavailable. "
              "No official leaderboard composite or serving-cost estimate is computed.", "",
              "The bilinear model continues an older warmup checkpoint with a different warmup recipe. "
              "Its difference from the other heads cannot be attributed solely to head architecture."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compare(args.runs, args.output)


if __name__ == "__main__":
    main()
