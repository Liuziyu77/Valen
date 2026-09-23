"""Merge disjoint game shards and compare policies on the same evaluation set."""
import argparse
import json
import math
from pathlib import Path
from .build_dataset import dump_json, dump_jsonl
from .dataset import sha256_file
from .evaluate import full_report
from .validate_dataset import read_jsonl, require


def normalized_media(settings):
    """Transformers may add its output container type to processor_kwargs.

    Both policy adapters explicitly request PyTorch tensors. This bookkeeping
    field is not an image transform; preserve every actual image/video setting.
    """
    result = dict(settings)
    require(result.get("return_tensors", "pt") == "pt", "Different processor output type")
    result.pop("return_tensors", None)
    return result


def merge(directory, eval_dir):
    directory, eval_dir = Path(directory), Path(eval_dir)
    levels = read_jsonl(eval_dir / "levels.jsonl")
    expected = [l["level_id"] for l in levels]
    manifests = sorted(directory.glob("shard_*/metrics.json"))
    require(bool(manifests), "No finished shards: " + str(directory))
    rows, configs = [], []
    for path in manifests:
        metrics = json.loads(path.read_text())
        config = metrics["config"]
        require(not config["smoke_subset"], "Cannot merge smoke subsets into a full evaluation")
        require(config["dataset_manifest_sha256"] == sha256_file(eval_dir / "manifest.json"), "Dataset version mismatch")
        shard_rows = read_jsonl(path.parent / "episodes.jsonl")
        require([r["level_id"] for r in shard_rows] == expected[config["shard_index"]::config["num_shards"]], "Shard coverage mismatch")
        configs.append(config)
        rows.extend(shard_rows)
    require(len(configs) == configs[0]["num_shards"], "Missing shards")
    require({c["shard_index"] for c in configs} == set(range(len(configs))), "Duplicate/missing shard indices")
    fields = ("policy", "checkpoint_sha256", "model_config_sha256", "model_weights_sha256", "media_kwargs",
              "temperature", "memoize", "num_shards", "runner_sha256", "wall_timeout", "seed")
    require(all(all(c.get(k) == configs[0].get(k) for k in fields) for c in configs), "Inconsistent shard settings")
    require(len(rows) == len(expected) and {r["level_id"] for r in rows} == set(expected), "Incomplete/duplicate games")
    rows.sort(key=lambda row: expected.index(row["level_id"]))
    merged_config = dict(configs[0], shard_index=None, merged_shards=len(configs),
                         memoization_repeat_verified=(all(c.get("memoization_repeat_verified") for c in configs)
                                                       if configs[0]["memoize"] else None))
    report = full_report(merged_config, rows)
    dump_jsonl(directory / "episodes.jsonl", rows)
    dump_json(directory / "metrics.json", report)
    return rows, report


def load_run(directory, eval_dir):
    """Read a complete unsharded run or merge all of a run's shards."""
    directory, eval_dir = Path(directory), Path(eval_dir)
    if any(directory.glob("shard_*")):
        return merge(directory, eval_dir)
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    config = report["config"]
    require(not config["smoke_subset"], "Cannot compare a smoke subset as a full evaluation")
    require(config["num_shards"] == 1 and config["shard_index"] == 0, "Expected an unsharded run")
    require(config["dataset_manifest_sha256"] == sha256_file(eval_dir / "manifest.json"), "Dataset version mismatch")
    rows = read_jsonl(directory / "episodes.jsonl")
    expected = [level["level_id"] for level in read_jsonl(eval_dir / "levels.jsonl")]
    require([row["level_id"] for row in rows] == expected, "Incomplete/duplicate games or incorrect order")
    return rows, full_report(config, rows)


def compare(first, second, eval_dir, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite comparison: " + str(output))
    first_rows, first_report = load_run(first, eval_dir)
    second_rows, second_report = load_run(second, eval_dir)
    a_config, b_config = first_report["config"], second_report["config"]
    for field in ("dataset_manifest_sha256", "memoize", "wall_timeout", "runner_sha256"):
        require(a_config[field] == b_config[field], "Comparison settings differ: " + field)
    require(normalized_media(a_config.get("media_kwargs", {})) == normalized_media(b_config.get("media_kwargs", {})),
            "Image processing differs")
    if not a_config["memoize"]:
        for row in first_rows + second_rows:
            require(row["cache_hits"] == 0, "Unexpected cache hit with memoization disabled")
            if row["termination"] in ("solved", "step_limit"):
                require(row["model_calls"] == row["steps"], "Every executed step must call the policy")
    pairs = dict.fromkeys(("both_solved", "first_only", "second_only", "both_failed"), 0)
    for a, b in zip(first_rows, second_rows):
        require(a["level_id"] == b["level_id"], "Unpaired levels")
        key = "both_solved" if a["solved"] and b["solved"] else "first_only" if a["solved"] else "second_only" if b["solved"] else "both_failed"
        pairs[key] += 1
    discordant = pairs["first_only"] + pairs["second_only"]
    pvalue = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(pairs["first_only"], pairs["second_only"]) + 1)) / (2 ** discordant)) if discordant else 1.
    a, b = first_report["overall"], second_report["overall"]
    comparison = {"first": first_report, "second": second_report,
                  "first_directory": str(Path(first).resolve()), "second_directory": str(Path(second).resolve()),
                  "paired_outcomes": pairs, "mcnemar_exact_two_sided_p": pvalue,
                  "success_rate_difference_first_minus_second": a["success_rate"] - b["success_rate"]}
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / "comparison.json", comparison)
    lines = ["# Sokoban policy comparison", "",
             f"Paired evaluation on {len(first_rows)} levels. Step limits come from the dataset.", "",
             f"First policy: `{a_config['policy']}`. Second policy: `{b_config['policy']}`.",
             "Oracle is a privileged reference check, not a visual model result.", "",
             "| Metric | First | Second |", "| --- | ---: | ---: |",
             f"| Solved | {a['solved']}/{a['games']} | {b['solved']}/{b['games']} |"]
    for key in ("success_rate", "invalid_action_rate", "static_deadlock_rate_lower_bound",
                "mean_success_steps", "mean_success_steps_over_optimal",
                "mean_uncached_decision_seconds", "model_calls", "cache_hits"):
        def fmt(value):
            return "—" if value is None else str(value) if isinstance(value, int) else f"{value:.4f}"
        lines.append(f"| {key} | {fmt(a[key])} | {fmt(b[key])} |")
    lines += ["", "Paired outcomes: `" + json.dumps(pairs) + "`.",
              f"Exact two-sided McNemar p = {pvalue:.4g}.", "",
              "Dataset, checkpoint, preprocessing and runner identities are recorded in comparison.json.",
              "Timing excludes cache hits; inspect memoize and privileged_reference before interpreting results."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    merge_parser = commands.add_parser("merge", help="Merge shard_*/ outputs from one policy")
    merge_parser.add_argument("--directory", type=Path, required=True)
    compare_parser = commands.add_parser("compare", help="Compare two complete or sharded policy runs")
    compare_parser.add_argument("--first", type=Path, required=True)
    compare_parser.add_argument("--second", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    for command in (merge_parser, compare_parser):
        command.add_argument("--eval-dir", type=Path, default=Path("data/eval_sokoban"))
    args = parser.parse_args()
    if args.command == "merge":
        _, report = merge(args.directory, args.eval_dir)
        print(json.dumps(report["overall"], indent=2))
    else:
        report = compare(args.first, args.second, args.eval_dir, args.output)
        print(json.dumps(report["paired_outcomes"], indent=2))


if __name__ == "__main__":
    main()
