"""Build 30k next-action records and 100 frozen complete-game test cases."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import time

from . import VERSION
from .dataset import make_record, sha256_file
from .env import canonical_hash, transition
from .generator import SOURCES, choose_initial, difficulty, generate_candidate, sample_training
from .render import render

ROOT = Path.cwd()


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dump_jsonl(path, values):
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _prepare(args):
    seed, max_states = args
    result = generate_candidate(seed, max_states=max_states)
    if result is None:
        return None
    board, index = result
    rng = random.Random(seed ^ 0x534F4B4F)
    # Prefer longer initial trajectories, but retain a range of difficulties.
    nonterminal = [s for s, d in index.distances.items() if d >= max(4, max(index.distances.values()) // 2)]
    initial = rng.choice(nonterminal)
    samples = sample_training(index, initial, rng, dict(zip(SOURCES, (8, 2, 1))))
    # Return only chosen states and their certificates, not a large search index.
    prepared = []
    for source, state, provenance in samples:
        order_rng = random.Random(seed * 1009 + len(prepared))
        record = make_record(board, state, index, "placeholder.png", "placeholder", "placeholder",
                             source, provenance, seed, 0, order_rng, "classic", 44)
        prepared.append((source, state, provenance, record))
    return seed, board, initial, index.solution(initial), prepared, {
        "indexed_states": len(index.distances), "reverse_graph_complete": index.complete,
        "max_indexed_distance": max(index.distances.values()), "expanded": index.expanded}


def _render_and_finalize(prepared, board, group_id, index_number, train_dir, rng, tile_size):
    source, state, provenance, record = prepared
    language = "zh" if index_number % 2 == 0 else "en"
    # Rebuild wording/auxiliary tasks using a tiny certified adapter, reusing
    # exactly the optimal labels proven in the worker (no second search).
    class Certificate:
        distances = {state: record["meta"]["optimal_distance"]}

        def optimal_actions(self, candidate):
            assert candidate == state
            return record["meta"]["question_annotations"]["next_action"]["optimal_actions"]

        def solution(self, candidate):
            assert candidate == state
            return record["meta"]["solution_actions"]
    relative = "assets/{:06d}.png".format(index_number)
    theme = rng.choice(("classic", "warm"))
    image_path = train_dir / relative
    render(board, state, theme, tile_size).save(image_path)
    result = make_record(board, state, Certificate(), relative, sha256_file(image_path), group_id,
                         source, provenance, record["meta"]["generator_seed"], index_number, rng, theme, tile_size)
    assert result["meta"]["language_bucket"] == language
    return result


def build(train_dir, eval_dir, train_count=30000, test_count=100, seed=20260922,
          workers=4, max_states=20000, tile_size=44, max_attempts=200000):
    train_dir, eval_dir = Path(train_dir).resolve(), Path(eval_dir).resolve()
    if train_count <= 0 or train_count % 20 or test_count < 1 or workers < 1:
        raise ValueError("Training count must be a positive multiple of 20; test count/workers must be positive")
    if train_dir == eval_dir or train_dir in eval_dir.parents or eval_dir in train_dir.parents:
        raise ValueError("Training and test directories must be separate")
    if not 24 <= tile_size <= 128 or max_states < 100 or max_attempts < 1:
        raise ValueError("Invalid generation or rendering budget")
    for path in (train_dir, eval_dir):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError("Refusing to overwrite nonempty dataset: " + str(path))
    for path in (train_dir, eval_dir):
        path.mkdir(parents=True, exist_ok=True)
        (path / "BUILDING").write_text("Incomplete until validated and manifest.json exists.\n")
    (train_dir / "assets").mkdir()
    started = time.monotonic()
    rng = random.Random(seed)
    seen_layouts, seen_states = set(), set()
    test_levels, references = [], []
    attempted, rejected = 0, Counter()
    search_stats = Counter()
    # Reserve test groups first; all symmetric copies are excluded from train.
    while len(test_levels) < test_count:
        if attempted >= max_attempts:
            raise RuntimeError("Could not fill test quotas; partial data retained with BUILDING marker")
        candidate_seed = seed + attempted
        attempted += 1
        result = generate_candidate(candidate_seed, max_states=max_states)
        if result is None:
            rejected["test_unsuitable"] += 1
            continue
        board, index = result
        identity = canonical_hash(board)
        if identity in seen_layouts:
            rejected["duplicate_layout"] += 1
            continue
        slot = len(test_levels) % 10
        bucket = "easy" if slot < 2 else "medium" if slot < 7 else "hard"
        initial = choose_initial(index, rng, bucket)
        if initial is None:
            rejected["test_difficulty_unavailable"] += 1
            continue
        seen_layouts.add(identity)
        solution = index.solution(initial)
        level_id = "sokoban-test-{:03d}".format(len(test_levels))
        group_id = "sokoban:" + identity
        test_levels.append({"level_id": level_id, "group_id": group_id, "split": "test",
                            "board": board.to_dict(), "initial_state": initial.to_dict(),
                            "generator_seed": candidate_seed, "rules_version": VERSION,
                            "render": {"theme": rng.choice(("classic", "warm")), "tile_size": tile_size},
                            "language": "zh" if len(test_levels) % 2 == 0 else "en", "max_steps": 200})
        state, pushes = initial, 0
        for action in solution:
            state, moved, pushed = transition(board, state, action)
            if not moved:
                raise AssertionError("Invalid reference action")
            pushes += int(pushed)
        assert board.solved(state)
        references.append({"level_id": level_id, "optimal_distance": len(solution), "difficulty": bucket,
                           "solution_actions": solution, "reference_pushes": pushes,
                           "label_method": "reverse_multisource_bfs", "solver_status": "optimal_proven"})
        if len(test_levels) % 25 == 0 or len(test_levels) == test_count:
            print(json.dumps({"event": "test_generation", "games": len(test_levels),
                              "attempts": attempted, "seconds": round(time.monotonic() - started, 1)}), flush=True)

    dump_jsonl(eval_dir / "levels.jsonl", test_levels)
    dump_jsonl(eval_dir / "reference.jsonl", references)
    quotas = dict(zip(SOURCES, (train_count * 80 // 100, train_count * 15 // 100, train_count * 5 // 100)))
    counts, records, train_levels = Counter(), [], []
    # Process deterministic batches in seed order; worker scheduling never
    # changes sample identities. Bound prefetch to keep memory predictable.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while len(records) < train_count:
            if attempted >= max_attempts:
                raise RuntimeError("Could not fill training quotas; partial data retained with BUILDING marker")
            batch_count = min(workers * 8, max_attempts - attempted)
            seeds = list(range(seed + attempted, seed + attempted + batch_count))
            attempted += batch_count
            for result in pool.map(_prepare, [(s, max_states) for s in seeds]):
                if result is None:
                    rejected["train_unsuitable"] += 1
                    continue
                candidate_seed, board, initial, solution, samples, stats = result
                identity = canonical_hash(board)
                if identity in seen_layouts:
                    rejected["duplicate_layout"] += 1
                    continue
                chosen = []
                for prepared in samples:
                    source, state, _, _ = prepared
                    state_identity = canonical_hash(board, state)
                    if counts[source] >= quotas[source] or state_identity in seen_states:
                        continue
                    seen_states.add(state_identity)
                    chosen.append(prepared)
                    counts[source] += 1
                if not chosen:
                    continue
                seen_layouts.add(identity)
                group_id = "sokoban:" + identity
                train_levels.append({"level_id": "sokoban-train-{:05d}".format(len(train_levels)),
                                     "group_id": group_id, "split": "train", "board": board.to_dict(),
                                     "initial_state": initial.to_dict(), "generator_seed": candidate_seed,
                                     "solution_actions": solution, "search": stats})
                search_stats["indexed_states"] += stats["indexed_states"]
                search_stats["complete_layouts"] += int(stats["reverse_graph_complete"])
                for prepared in chosen:
                    records.append(_render_and_finalize(prepared, board, group_id, len(records), train_dir, rng, tile_size))
                if len(train_levels) % 100 == 0 or len(records) == train_count:
                    print(json.dumps({"event": "train_generation", "records": len(records), "sources": dict(counts),
                                      "levels": len(train_levels), "seconds": round(time.monotonic() - started, 1)}), flush=True)
                if len(records) == train_count:
                    break
    assert dict(counts) == quotas
    rng.shuffle(records)
    dump_jsonl(train_dir / "train.jsonl", records)
    dump_jsonl(train_dir / "levels.jsonl", train_levels)
    task_counts = Counter({"next_action": len(records)})
    common = {"version": VERSION, "seed": seed, "objective": "minimum_moves",
              "generator": "connected_room_reverse_multisource_bfs", "max_search_states": max_states,
              "max_distance": 80, "split_policy": "canonical_layout_disjoint_d4",
              "attempted_candidates": attempted, "rejections": dict(rejected)}
    train_manifest = dict(common, split="train", records=len(records), unique_states=len(seen_states),
                          levels=len(train_levels), questions=sum(task_counts.values()),
                          source_counts=dict(counts), task_counts=dict(task_counts),
                          languages=dict(Counter(r["meta"]["language_bucket"] for r in records)),
                          distance_buckets=dict(Counter(difficulty(r["meta"]["optimal_distance"]) for r in records)),
                          box_counts=dict(Counter(str(len(r["meta"]["state"]["boxes"])) for r in records)),
                          search_statistics=dict(search_stats),
                          files={name: sha256_file(train_dir / name) for name in ("train.jsonl", "levels.jsonl")})
    eval_manifest = dict(common, split="test", games=len(test_levels),
                         difficulty_counts=dict(Counter(r["difficulty"] for r in references)),
                         max_steps=200, temperature=1.0,
                         files={name: sha256_file(eval_dir / name) for name in ("levels.jsonl", "reference.jsonl")})
    dump_json(train_dir / "manifest.json", train_manifest)
    dump_json(eval_dir / "manifest.json", eval_manifest)
    # Full cross-split, trajectory, label and image verification before completion.
    from .validate_dataset import validate_dataset
    report = validate_dataset(train_dir, eval_dir, verify_images=True, allow_building=True)
    dump_json(train_dir / "validation_report.json", report)
    dump_json(eval_dir / "validation_report.json", report)
    for path in (train_dir, eval_dir):
        (path / "BUILDING").unlink()
    print(json.dumps({"event": "complete", "train_records": len(records), "test_games": len(test_levels),
                      "questions": sum(task_counts.values()), "seconds": round(time.monotonic() - started, 1)}), flush=True)
    return train_manifest, eval_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data/train_sokoban")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "data/eval_sokoban")
    parser.add_argument("--train-count", type=int, default=30000)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-states", type=int, default=20000)
    parser.add_argument("--tile-size", type=int, default=44)
    parser.add_argument("--max-attempts", type=int, default=200000)
    args = parser.parse_args()
    build(args.train_dir, args.eval_dir, args.train_count, args.test_count, args.seed,
          args.workers, args.max_states, args.tile_size, args.max_attempts)


if __name__ == "__main__":
    main()
