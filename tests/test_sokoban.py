"""Rules, exact-search certificates, dataset leakage, and complete episodes."""
import json
import random
import pytest
from evaluation.sokoban.env import ACTIONS, Board, SokobanEnv, State, canonical_hash, parse_ascii, transition
from evaluation.sokoban.solver import ReverseIndex, reverse_predecessors, solve_bfs, static_dead_squares
from evaluation.sokoban.generator import SOURCES, generate_candidate, sample_training
from evaluation.sokoban.render import render
from evaluation.sokoban.build_dataset import build
from evaluation.sokoban.evaluate import ReferencePolicy, run_episode, evaluate
from evaluation.sokoban.evaluate import MemoizedPolicy
from evaluation.sokoban.dataset import action_request
from evaluation.sokoban.compare import compare, load_run, merge, normalized_media
from evaluation.sokoban.validate_dataset import read_jsonl, validate_dataset


def test_push_goal_overlay_terminal_and_invalid_step_budget():
    board, initial = parse_ascii("#####\n#@$.#\n#####")
    env = SokobanEnv(board, initial, max_steps=2)
    _, event = env.step("up")
    assert not event["moved"] and env.steps == 1 and env.state == initial
    child = env.clone()
    state, event = env.step("right")
    assert event["pushed"] and event["terminated"] and not event["truncated"]
    assert "*" in board.ascii(state)[1]
    assert child.state == initial and child.steps == 1
    with pytest.raises(RuntimeError):
        env.step("left")
    with pytest.raises(ValueError):
        transition(board, initial, "undo")


def test_no_chain_push_no_pull_and_goal_preservation():
    board, state = parse_ascii("#######\n#@$$..#\n#######")
    assert transition(board, state, "right") == (state, False, False)
    board, state = parse_ascii("######\n# +$ #\n######")
    moved, valid, pushed = transition(board, state, "left")
    assert valid and not pushed and moved.boxes == state.boxes
    assert board.ascii(moved)[1][2] == "."


def test_reverse_edges_really_are_inverse_and_distances_match_forward_bfs():
    board, _ = parse_ascii("######\n#    #\n# @$ #\n#  . #\n######")
    index = ReverseIndex(board)
    assert index.complete
    for state, distance in index.distances.items():
        for previous, action in reverse_predecessors(board, state):
            assert transition(board, previous, action)[0] == state
        reference = solve_bfs(board, state, max_nodes=5000)
        assert reference.status == "optimal_proven" and reference.distance == distance
        if distance:
            expected = []
            for action in ACTIONS:
                child, valid, _ = transition(board, state, action)
                if valid:
                    result = solve_bfs(board, child, max_nodes=5000)
                    if result.distance == distance - 1:
                        expected.append(action)
            assert index.optimal_actions(state) == expected


def test_truncated_reverse_index_preserves_exact_sets_and_unknown_status():
    board, _ = parse_ascii("#######\n#     #\n# @$  #\n#   . #\n#     #\n#######")
    full = ReverseIndex(board)
    limited = ReverseIndex(board, max_states=35)
    assert not limited.complete
    for state, distance in limited.distances.items():
        assert full.distances[state] == distance
        assert full.optimal_actions(state) == limited.optimal_actions(state)
    missing = next(s for s in full.distances if s not in limited.distances)
    assert limited.status(missing) == "unknown"
    far = max(full.distances, key=full.distances.get)
    assert solve_bfs(board, far, max_nodes=1).status == "unknown"


def test_static_dead_square_and_unsolvable_proof():
    board, state = parse_ascii("######\n#$ @ #\n#  . #\n######")
    assert state.boxes[0] in static_dead_squares(board)
    assert solve_bfs(board, state).status == "proven_unsolvable"
    assert ReverseIndex(board).status(state) == "proven_unsolvable"


def test_two_box_search_and_depth_limited_certificates():
    board, initial = parse_ascii("#######\n# @   #\n# $$  #\n# ..  #\n#######")
    full = ReverseIndex(board)
    assert full.distances[initial] == solve_bfs(board, initial).distance
    limited = ReverseIndex(board, max_depth=2)
    assert not limited.complete
    for state, distance in limited.distances.items():
        assert distance == full.distances[state]
        assert limited.optimal_actions(state) == full.optimal_actions(state)


def test_canonical_identity_handles_symmetry_and_wall_margin():
    board, state = parse_ascii("######\n#@ $ #\n#  . #\n######")
    rotated = ["".join(row) for row in zip(*board.ascii(state)[::-1])]
    other, other_state = parse_ascii(rotated)
    assert canonical_hash(board) == canonical_hash(other)
    assert canonical_hash(board, state) == canonical_hash(other, other_state)
    child = transition(board, state, "right")[0]
    assert canonical_hash(board, state) != canonical_hash(board, child)
    assert render(board, state).size == (264, 176)


def test_multiple_optimal_actions_exist():
    board, _ = parse_ascii("#######\n#     #\n# @$  #\n#   . #\n#     #\n#######")
    index = ReverseIndex(board)
    assert any(len(index.optimal_actions(s)) > 1 for s in index.distances)


def test_failed_rollout_corrections_and_deviations_have_provenance():
    board, index = generate_candidate(0)
    initial = max(index.distances, key=index.distances.get)
    samples = sample_training(index, initial, random.Random(123), dict(zip(SOURCES, (8, 2, 1))))
    assert {s[0] for s in samples} == set(SOURCES)
    for source, state, proof in samples:
        at = initial
        for action in proof["prefix_actions"]:
            at = transition(board, at, action)[0]
        assert at == state
        if source == "failure_correction":
            child = transition(board, state, proof["failed_action"])[0]
            assert index.status(child) == "proven_unsolvable"
        elif source == "solvable_deviation":
            assert proof["deviation_action"] not in index.optimal_actions(State.from_dict(proof["parent_state"]))


def level_fixture():
    board, initial = parse_ascii("#####\n#@$.#\n#####")
    return {"level_id": "test-0", "group_id": "test-group", "board": board.to_dict(),
            "initial_state": initial.to_dict(), "max_steps": 3,
            "render": {"theme": "classic", "tile_size": 32}, "language": "en"}


def test_episode_calls_policy_with_only_pixels_rules_and_action_question(tmp_path):
    class Policy:
        def choose(self, request):
            assert set(request) == {"state", "questions"}
            assert set(request["questions"]) == {"next_action"}
            assert len(request["state"]["messages"][0]["content"]) == 2
            return {"action": "right"}
    result = run_episode(Policy(), level_fixture(), tmp_path / "episode", {"optimal_distance": 1})
    assert result["solved"] and result["steps"] == 1 and result["steps_over_optimal"] == 1


def test_episode_no_auto_correction_and_policy_errors_count_as_failures(tmp_path):
    class WallPolicy:
        def choose(self, request):
            return {"action": "up"}
    result = run_episode(WallPolicy(), level_fixture(), tmp_path / "wall")
    assert result["steps"] == result["invalid_moves"] == 3
    assert not result["solved"] and result["termination"] == "step_limit"
    class ErrorPolicy:
        def choose(self, request):
            raise RuntimeError("expected test failure")
    result = run_episode(ErrorPolicy(), level_fixture(), tmp_path / "error")
    assert not result["solved"] and result["termination"] == "policy_error" and result["steps"] == 0


def test_memoization_uses_pixels_prompt_and_candidate_order(tmp_path):
    from PIL import Image
    path = tmp_path / "current.png"
    Image.new("RGB", (32,32), "white").save(path)
    class Policy:
        calls = 0
        def choose(self, request):
            self.calls += 1
            return {"action": "up", "probabilities": {a: float(a == "up") for a in ACTIONS}}
    base = Policy()
    policy = MemoizedPolicy(base)
    request = action_request(path)
    assert not policy.choose(request)["cache_hit"]
    assert policy.choose(request)["cache_hit"] and policy.verified
    assert base.calls == 2  # First reuse independently verified.
    assert policy.choose(request)["model_calls"] == 0 and base.calls == 2
    Image.new("RGB", (32,32), "black").save(path)
    assert not policy.choose(request)["cache_hit"] and base.calls == 3
    reordered = action_request(path, action_order=tuple(reversed(ACTIONS)))
    assert not policy.choose(reordered)["cache_hit"] and base.calls == 4
    assert not policy.choose(action_request(path, "zh"))["cache_hit"] and base.calls == 5


def test_media_comparison_ignores_only_tensor_container_bookkeeping():
    original = {"images_kwargs": {"size": {"shortest_edge": 65536}}}
    mutated = dict(original, return_tensors="pt")
    assert normalized_media(original) == normalized_media(mutated)
    assert "return_tensors" in mutated  # Do not rewrite raw run provenance.
    assert normalized_media({"images_kwargs": {"size": {"shortest_edge": 1024}}}) != normalized_media(original)
    with pytest.raises(ValueError, match="output type"):
        normalized_media(dict(original, return_tensors="np"))


def test_small_dataset_build_audit_and_full_oracle_games(tmp_path):
    train, test = tmp_path / "train", tmp_path / "eval"
    train_manifest, test_manifest = build(train, test, train_count=40, test_count=4, seed=123, workers=1)
    assert train_manifest["records"] == 40 and train_manifest["questions"] == 40
    assert train_manifest["task_counts"] == {"next_action": 40}
    for record in read_jsonl(train / "train.jsonl"):
        assert set(record["request"]["questions"]) == set(record["targets"]) == {"next_action"}
        assert set(record["meta"]["question_annotations"]) == {"next_action"}
    assert train_manifest["source_counts"] == dict(zip(SOURCES, (32, 6, 2)))
    assert test_manifest["games"] == 4
    assert not (train / "BUILDING").exists()
    report = evaluate(test, tmp_path / "oracle", policy_name="oracle")
    assert report["overall"]["games"] == report["overall"]["solved"] == 4
    assert report["overall"]["mean_success_steps_over_optimal"] == 1
    assert report["config"]["privileged_reference"]
    for shard in range(2):
        evaluate(test, tmp_path / "sharded" / ("shard_" + str(shard)), policy_name="oracle", num_shards=2, shard_index=shard)
    rows, merged = merge(tmp_path / "sharded", test)
    assert len(rows) == merged["overall"]["solved"] == 4
    comparison = compare(tmp_path / "oracle", tmp_path / "sharded", test, tmp_path / "comparison")
    assert comparison["paired_outcomes"] == dict(both_solved=4, first_only=0, second_only=0, both_failed=0)
    assert comparison["success_rate_difference_first_minus_second"] == 0
    assert comparison["mcnemar_exact_two_sided_p"] == 1
    report = (tmp_path / "comparison/report.md").read_text()
    assert "4/4" in report and "2B" not in report and "/200" not in report
    with pytest.raises(FileExistsError):
        compare(tmp_path / "oracle", tmp_path / "sharded", test, tmp_path / "comparison")
    metrics_path = tmp_path / "oracle/metrics.json"
    original = metrics_path.read_text()
    metrics = json.loads(original)
    metrics["config"]["smoke_subset"] = True
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="smoke subset"):
        load_run(tmp_path / "oracle", test)
    metrics["config"]["smoke_subset"] = False
    metrics["config"]["dataset_manifest_sha256"] = "wrong dataset"
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="Dataset version"):
        load_run(tmp_path / "oracle", test)
    metrics_path.write_text(original)
    (tmp_path / "sharded" / "shard_1" / "metrics.json").unlink()
    with pytest.raises(ValueError, match="Missing shards"):
        merge(tmp_path / "sharded", test)
    with pytest.raises(FileExistsError):
        build(train, test, train_count=40, test_count=4, seed=123, workers=1)
    image = next((train / "assets").glob("*.png"))
    image.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Image hash"):
        validate_dataset(train, test)
