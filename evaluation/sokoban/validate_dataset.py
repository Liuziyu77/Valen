"""Audit generated data, including exact optimal sets and replayable provenance."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from PIL import Image
from valen.data.schema import validate_record
from .dataset import sha256_file
from .env import ACTIONS, Board, State, canonical_hash, state_hash, transition
from .generator import SOURCES, difficulty
from .solver import ReverseIndex


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def replay(board, state, actions, must_move=True):
    for action in actions:
        state, moved, _ = transition(board, state, action)
        require(moved or not must_move, "Invalid action in certified replay")
    return state


def validate_dataset(train_dir, eval_dir, verify_images=True, allow_building=False):
    started = time.monotonic()
    train_dir, eval_dir = Path(train_dir), Path(eval_dir)
    manifests = []
    for directory in (train_dir, eval_dir):
        require(allow_building or not (directory / "BUILDING").exists(), "Dataset is incomplete")
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            require(sha256_file(directory / name) == digest, "File hash mismatch: " + name)
        manifests.append(manifest)
    train_manifest, eval_manifest = manifests
    train_levels = read_jsonl(train_dir / "levels.jsonl")
    test_levels = read_jsonl(eval_dir / "levels.jsonl")
    references = read_jsonl(eval_dir / "reference.jsonl")
    records = read_jsonl(train_dir / "train.jsonl")
    require(len(records) == train_manifest["records"], "Training record count mismatch")
    require(len(test_levels) == len(references) == eval_manifest["games"], "Test game count mismatch")
    require(len(train_levels) == train_manifest["levels"], "Training level count mismatch")
    require(train_manifest["split"] == "train" and eval_manifest["split"] == "test", "Unexpected split")
    grouped, layout_ids, seen_states, seen_record_ids, asset_paths = defaultdict(list), set(), set(), set(), set()
    sources, tasks, languages, distances, boxes = (Counter() for _ in range(5))
    for record in records:
        validate_record(record)
        meta = record["meta"]
        require(meta["split"] == "train", "Non-training record in training file")
        require(meta["record_id"] not in seen_record_ids, "Duplicate record ID")
        seen_record_ids.add(meta["record_id"])
        require(meta["sampling_source"] in SOURCES, "Unknown source")
        sources[meta["sampling_source"]] += 1
        languages[meta["language_bucket"]] += 1
        distances[difficulty(meta["optimal_distance"])] += 1
        boxes[str(len(meta["state"]["boxes"]))] += 1
        grouped[record["group_id"]].append(record)
        require(set(record["request"]["questions"]) == set(record["targets"])
                == set(meta["question_annotations"]) == {"next_action"},
                "Each record must contain only next_action in questions, targets and annotations")
        require(record["request"]["questions"]["next_action"]["type"] == "choice",
                "next_action must be a choice question")
        tasks["next_action"] += 1
        require(len(record["assets"]) == 1, "Expected one screenshot")
        asset = record["assets"][0]
        require(asset["path"] not in asset_paths, "Duplicate asset path")
        asset_paths.add(asset["path"])
        image_path = (train_dir / asset["path"]).resolve()
        require(train_dir.resolve() in image_path.parents, "Asset path outside dataset")
        content = record["request"]["state"]["messages"][0]["content"]
        require(len(content) == 2 and content[0]["type"] == "text" and content[1]["type"] == "image_url",
                "Unexpected model-visible content")
        require(content[1]["image_url"]["url"] == asset["path"], "Image request/asset mismatch")
        if verify_images:
            require(sha256_file(image_path) == asset["sha256"], "Image hash mismatch")
            with Image.open(image_path) as image:
                image.verify()

    for number, level in enumerate(train_levels):
        board, initial = Board.from_dict(level["board"]), State.from_dict(level["initial_state"])
        board.validate(initial)
        identity = canonical_hash(board)
        require(identity not in layout_ids and level["group_id"] == "sokoban:" + identity, "Duplicate or invalid layout group")
        layout_ids.add(identity)
        index = ReverseIndex(board, max_states=train_manifest["max_search_states"], max_depth=train_manifest["max_distance"])
        require(initial in index.distances and index.distances[initial] == len(level["solution_actions"]), "Unproven initial distance")
        require(board.solved(replay(board, initial, level["solution_actions"])), "Reference does not solve level")
        require(level["group_id"] in grouped, "Unused training layout")
        group = grouped.pop(level["group_id"])
        require(len(group) <= 20, "Too many records from one layout")
        for record in group:
            meta = record["meta"]
            state = board.validate(State.from_dict(meta["state"]))
            require(not board.solved(state) and state in index.distances, "Training state must be solvable and nonterminal")
            identity = canonical_hash(board, state)
            require(identity not in seen_states and identity == meta["canonical_state_id"], "Duplicate or invalid canonical state")
            seen_states.add(identity)
            require(meta["state_id"] == state_hash(board, state), "Invalid exact state hash")
            require(meta["optimal_distance"] == index.distances[state], "Incorrect distance")
            optimal = index.optimal_actions(state)
            labels = record["targets"]["next_action"]["probabilities"]
            require(set(labels) == set(ACTIONS), "Missing action candidate")
            require(set(meta["question_annotations"]["next_action"]["optimal_actions"]) == set(optimal), "Wrong optimal action set")
            require(all(abs(labels[a] - (1 / len(optimal) if a in optimal else 0)) < 1e-9 for a in ACTIONS), "Wrong policy distribution")
            require(len(meta["solution_actions"]) == index.distances[state]
                    and board.solved(replay(board, state, meta["solution_actions"])), "Invalid record solution")
            proof = meta["provenance"]
            require(replay(board, initial, proof["prefix_actions"]) == state, "Provenance does not reach sample")
            if meta["sampling_source"] == "optimal_trajectory":
                require(len(proof["prefix_actions"]) + index.distances[state] == index.distances[initial], "Nonoptimal reference prefix")
            elif meta["sampling_source"] == "solvable_deviation":
                parent = State.from_dict(proof["parent_state"])
                action = proof["deviation_action"]
                require(proof["prefix_actions"][-1] == action and transition(board, parent, action)[0] == state,
                        "Wrong deviation transition")
                require(replay(board, initial, proof["prefix_actions"][:-1]) == parent
                        and proof["parent_distance"] == index.distances[parent], "Wrong deviation parent provenance")
                require(action not in index.optimal_actions(parent), "Deviation action was optimal")
            else:
                failure = State.from_dict(proof["failure_state"])
                require(transition(board, state, proof["failed_action"])[0] == failure, "Wrong failed transition")
                require(index.status(failure) == "proven_unsolvable", "Failure is not certified")
                if proof["certificate"] == "static_dead_square":
                    require(any(b in index.dead_squares for b in failure.boxes), "Invalid dead-square certificate")
                else:
                    require(proof["certificate"] == "exhausted_reverse_graph" and index.complete, "Incomplete deadlock proof")
            if verify_images:
                with Image.open(train_dir / record["assets"][0]["path"]) as image:
                    require(image.size == (board.width * meta["tile_size"], board.height * meta["tile_size"]), "Incorrect image dimensions")
        if (number + 1) % 500 == 0:
            print(json.dumps({"event": "audit_training", "levels": number + 1, "seconds": round(time.monotonic() - started, 1)}), flush=True)
    require(not grouped, "Record references missing layout")
    ref_map = {r["level_id"]: r for r in references}
    require(len(ref_map) == len(references), "Duplicate reference ID")
    seen_test_ids = set()
    for level in test_levels:
        board, initial = Board.from_dict(level["board"]), State.from_dict(level["initial_state"])
        board.validate(initial)
        identity = canonical_hash(board)
        require(identity not in layout_ids, "Train/test leakage or duplicate test layout")
        layout_ids.add(identity)
        require(level["group_id"] == "sokoban:" + identity and level["split"] == "test", "Invalid test group")
        require(level["level_id"] not in seen_test_ids, "Duplicate test ID")
        seen_test_ids.add(level["level_id"])
        reference = ref_map[level["level_id"]]
        require(not board.solved(initial), "Test starts completed")
        require(board.solved(replay(board, initial, reference["solution_actions"])), "Test reference fails")
        index = ReverseIndex(board, max_states=eval_manifest["max_search_states"], max_depth=eval_manifest["max_distance"])
        require(index.distances.get(initial) == reference["optimal_distance"] == len(reference["solution_actions"]), "Test distance not exact")
        require(reference["difficulty"] == difficulty(reference["optimal_distance"]), "Wrong test difficulty")
        require(reference["optimal_distance"] <= level["max_steps"], "Test budget below reference length")
    require(seen_test_ids == set(ref_map), "Extra or missing test references")
    require(dict(sources) == train_manifest["source_counts"], "Source counts mismatch")
    require(dict(sources) == dict(zip(SOURCES, (len(records)*80//100, len(records)*15//100, len(records)*5//100))), "Wrong 80/15/5 source quotas")
    require(len(seen_states) == len(records) == train_manifest["unique_states"], "Unique state count mismatch")
    require(dict(tasks) == train_manifest["task_counts"] and sum(tasks.values()) == train_manifest["questions"], "Question counts mismatch")
    require(dict(languages) == train_manifest["languages"] and dict(distances) == train_manifest["distance_buckets"]
            and dict(boxes) == train_manifest["box_counts"], "Training distribution mismatch")
    return {"passed": True, "train_records": len(records), "train_questions": sum(tasks.values()),
            "test_games": len(test_levels), "train_levels": len(train_levels), "source_counts": dict(sources),
            "images_verified": verify_images, "optimal_sets_recomputed": len(records),
            "cross_split_layout_overlap": 0, "canonical_duplicate_states": 0,
            "seconds": round(time.monotonic() - started, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path.cwd()
    parser.add_argument("--train-dir", type=Path, default=root / "data/train_sokoban")
    parser.add_argument("--eval-dir", type=Path, default=root / "data/eval_sokoban")
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.train_dir, args.eval_dir, not args.skip_images), indent=2))


if __name__ == "__main__":
    main()
