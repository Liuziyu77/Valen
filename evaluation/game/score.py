"""Score labeled game actions, including ties. / 任一等长最优动作都算正确。"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score(data, predictions, revision):
    records, values = rows(data), rows(predictions)
    expected = {(i, qid) for i, record in enumerate(records) for qid in record.get("targets", {})}
    identities = [(row["record_index"], row["qid"]) for row in values]
    if len(set(identities)) != len(identities) or set(identities) != expected:
        raise ValueError("Game predictions duplicated or omitted labeled questions")
    buckets = defaultdict(list)
    for row in values:
        record = records[row["record_index"]]
        target = record["targets"][row["qid"]]["probabilities"]
        probabilities = row["probabilities"]
        if set(target) != set(probabilities) or any(not math.isfinite(p) or p < 0 for p in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1., abs_tol=1e-5):
            raise ValueError("Invalid game probability distribution")
        if row["target"] != target:
            raise ValueError("Prediction labels differ from game dataset")
        chosen = max(probabilities, key=probabilities.get)
        optimal = {key for key, value in target.items() if value > 0}
        if not optimal:
            raise ValueError("Game target has no optimal action")
        result = {"optimal_action_accuracy": float(chosen in optimal),
                  "optimal_probability_mass": sum(probabilities[key] for key in optimal),
                  "brier": sum((probabilities[key] - target[key])**2 for key in target),
                  "cross_entropy": -sum(target[key] * math.log(max(probabilities[key], 1e-30)) for key in target),
                  "tied_optimum": float(len(optimal) > 1)}
        environment = record["meta"]["environment"]
        buckets[environment].append(result)
        buckets["overall"].append(result)
    def average(items):
        return {"questions": len(items), **{key: sum(item[key] for item in items) / len(items) for key in items[0]}}
    return {"revision": revision, "protocol": "labeled single-step optimal-action-set",
            "data_sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest(),
            "prediction_sha256": hashlib.sha256(Path(predictions).read_bytes()).hexdigest(),
            "metrics": {name: average(items) for name, items in sorted(buckets.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", required=True, choices=("game-v4", "arcade-v2", "sokoban-v1"))
    args = parser.parse_args()
    report = score(args.data, args.predictions, args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
