"""Complete games on published eval maps, with separately named legacy Sokoban."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import time

from valen.training.distributed import initialize, close
from evaluation.sokoban.evaluate import ValenPolicy, MemoizedPolicy, run_episode as legacy_episode, summarize as legacy_summary
from .dataset import eval_maps, make_game, validate_initial_frame, action_request, solution


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run_episode(policy, record, output):
    output.mkdir(parents=True, exist_ok=True)
    env = make_game(record)
    validate_initial_frame(env, record)
    trace, calls, hits, optimal, decisions = [], 0, 0, 0, 0
    initial_length = len(solution(env))
    if initial_length < 1:
        raise ValueError("Game initial state has no winning solution")
    started = time.monotonic()
    while not env.done:
        path = output / "current.png"
        env.observation().save(path, format="PNG", compress_level=1)
        optimal_before = env.optimal_actions()
        result = policy.choose(action_request(record, env, path))
        action = result["action"]
        optimal += int(action in optimal_before)
        decisions += 1
        calls += result.get("model_calls", 1)
        hits += int(result.get("cache_hit", False))
        _, reward, _, info = env.step(action)
        # Trace is evaluator-only; symbolic info is never passed to the policy.
        trace.append({"step": env.steps, "action": action, "probabilities": result.get("probabilities"),
                      "reward": reward, "info": info})
    env.observation().save(output / "final.png", format="PNG", compress_level=1)
    with (output / "trajectory.jsonl").open("w") as stream:
        for row in trace:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "current.png").unlink(missing_ok=True)
    result = {"map_id": record["meta"]["map_id"], "environment": record["meta"]["environment"],
              "revision": record["meta"]["source_revision"], "success": env.success,
              "steps": env.steps, "optimal_distance": initial_length,
              "steps_over_optimal": env.steps / initial_length if env.success else None,
              "optimal_action_matches": optimal, "decisions": decisions, "model_calls": calls,
              "cache_hits": hits, "seconds": time.monotonic()-started}
    dump(output / "result.json", result)
    return result


def summarize(items):
    successful = [r for r in items if r["success"]]
    ratios = [r["steps_over_optimal"] for r in successful]
    decisions = sum(r["decisions"] for r in items)
    return {"games": len(items), "solved": len(successful), "success_rate": len(successful)/len(items),
            "mean_success_steps_over_optimal": sum(ratios)/len(ratios) if ratios else None,
            "optimal_action_accuracy": sum(r["optimal_action_matches"] for r in items)/decisions if decisions else None,
            "decisions": decisions, "model_calls": sum(r["model_calls"] for r in items),
            "cache_hits": sum(r["cache_hits"] for r in items)}


def run(checkpoint, datasets, output, legacy_eval_dir=None, device="cuda", oracle=False):
    output = output.resolve()
    datasets = [p.resolve() for p in datasets]
    checkpoint = checkpoint.resolve() if checkpoint else None
    legacy_eval_dir = legacy_eval_dir.resolve() if legacy_eval_dir else None
    context = initialize(device)
    output.mkdir(parents=True, exist_ok=True)
    cases = eval_maps(datasets)
    legacy = []
    references = {}
    if legacy_eval_dir:
        legacy = [json.loads(line) for line in (legacy_eval_dir / "levels.jsonl").read_text().splitlines() if line.strip()]
        references = {r["level_id"]: r for r in (json.loads(line) for line in (legacy_eval_dir/"reference.jsonl").read_text().splitlines() if line.strip())}
        if {r["level_id"] for r in legacy} != set(references):
            raise ValueError("Legacy Sokoban level/reference coverage mismatch")
    policy = MemoizedPolicy(ValenPolicy(checkpoint, context.device)) if not oracle else None
    identity = {"protocol": "complete published eval maps from initial state", "world_size": context.world_size,
                "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256((checkpoint/"checkpoint.pt").read_bytes()).hexdigest() if checkpoint else None,
                "datasets_sha256": {p.name+":"+str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in datasets},
                "privileged_oracle": oracle, "memoize": not oracle,
                "memoization_key": "visible image SHA and full request only",
                "legacy_eval_dir": str(legacy_eval_dir) if legacy_eval_dir else None}
    manifest = output / "run_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Game evaluation resume identity changed")
    if context.primary:
        dump(manifest, identity)
    context.barrier()
    started, rows, legacy_rows = time.monotonic(), [], []
    for index in range(context.rank, len(cases), context.world_size):
        record = cases[index]
        destination = output / "episodes" / record["meta"]["map_id"]
        if (destination / "result.json").exists():
            rows.append(json.loads((destination / "result.json").read_text()))
            continue
        if oracle:
            env = make_game(record)
            class OraclePolicy:
                def __init__(self):
                    self.actions = iter(solution(env))
                def choose(self, request):
                    action = next(self.actions)
                    return {"action": action, "probabilities": {a: float(a == action) for a in env.actions}}
            policy = OraclePolicy()
        rows.append(run_episode(policy, record, destination))
        print(json.dumps({"event": "game_episode", "rank": context.rank, **rows[-1]}), flush=True)
    for index in range(context.rank, len(legacy), context.world_size):
        level = legacy[index]
        destination = output / "sokoban-v1" / level["level_id"]
        if (destination / "result.json").exists():
            legacy_rows.append(json.loads((destination / "result.json").read_text()))
            continue
        if destination.exists():
            shutil.rmtree(destination)  # Retry only an incomplete episode. / 仅清理未完成的一局。
        if oracle:
            from evaluation.sokoban.evaluate import ReferencePolicy
            policy = ReferencePolicy(references[level["level_id"]]["solution_actions"])
        result = legacy_episode(policy, level, destination, references[level["level_id"]])
        if result["error"]:
            raise RuntimeError(result["error"])
        legacy_rows.append(result)
    gathered, old_gathered = context.gather(rows), context.gather(legacy_rows)
    timings = context.gather({"rank": context.rank, "seconds": time.monotonic()-started})
    if context.primary:
        all_rows = [r for values in gathered for r in values]
        old_rows = [r for values in old_gathered for r in values]
        if len(all_rows) != len(cases) or len({r["map_id"] for r in all_rows}) != len(cases) or len(old_rows) != len(legacy):
            raise ValueError("Game episode coverage mismatch")
        buckets = defaultdict(list)
        for row in all_rows:
            buckets[row["revision"]+"/"+row["environment"]].append(row)
        report = {**identity, "rank_timing": timings, "wall_seconds": max(r["seconds"] for r in timings),
                  "metrics": {key: summarize(values) for key, values in sorted(buckets.items())}}
        if old_rows:
            report["metrics"]["sokoban-v1/sokoban"] = legacy_summary(old_rows)
        dump(output / "metrics.json", report)
        dump(output / "complete.json", {"games": len(all_rows), "legacy_games": len(old_rows), "failed": 0})
    context.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--datasets", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--legacy-eval-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()
    if not args.oracle and not args.checkpoint:
        parser.error("--checkpoint is required for model evaluation")
    try:
        run(args.checkpoint, args.datasets, args.output, args.legacy_eval_dir, args.device, args.oracle)
    finally:
        close()


if __name__ == "__main__":
    main()
