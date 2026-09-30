"""JevBench mapping, pinned scoring, failure accounting and resume behavior."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from evaluation.jevbench._upstream.tasks import Task
from evaluation.jevbench._upstream.scoring import score_task
from evaluation.jevbench.dataset import to_record, probabilities
from evaluation.jevbench.metrics import summarize_run


def task(kind="choice", expected="b", state="Assess this input."):
    criteria = {"b": "Second option", "a": "First option"}
    labels = ["a", "b"]
    if kind == "noul":
        criteria, labels = {"true": "All requirements proven", "false": "A requirement is unproven"}, ["no", "yes"]
    if kind == "score":
        criteria, labels = ["low", "middle", "high"], ["0", "1", "2"]
    return Task("task-1", "family", state, {"type": kind, "instructions": "Apply the rubric.", "criteria": criteria},
                labels, expected, "public", provenance={"rationale": "SECRET GOLD EXPLANATION"})


def test_mapping_preserves_option_order_and_excludes_gold_metadata():
    original = task()
    before = deepcopy(original)
    record = to_record(original)
    assert list(record["request"]["questions"]["decision"]["criteria"]) == ["b", "a"]
    assert original == before
    assert "targets" not in record
    assert "SECRET GOLD" not in json.dumps(record)
    assert "expected" not in record["request"]


def test_noul_retains_custom_rubric_and_maps_yes_to_true():
    original = task("noul", "yes")
    question = to_record(original)["request"]["questions"]["decision"]
    assert "All requirements proven" in question["instructions"]
    assert "A requirement is unproven" in question["instructions"]
    probs = probabilities(original, {"answers": {"decision": {"type": "noul", "noul": .8}}})
    assert probs == pytest.approx({"no": .2, "yes": .8})
    assert score_task(probs, original)["correct"]


def test_structured_state_is_serialized_without_gold_fields():
    original = task(state={"z": "事实", "a": [1, 2]})
    assert to_record(original)["request"]["state"] == '{"a": [1, 2], "z": "事实"}'


def test_score_uses_argmax_and_reports_expected_value_separately():
    original = task("score", 0)
    scored = score_task({"0": .45, "1": .15, "2": .4}, original)
    assert scored["correct"] and scored["ordinal_ev"] == pytest.approx(.95)
    assert score_task({"0": .5, "1": 0., "2": .5}, original)["predicted"] == "0"


@pytest.mark.parametrize("probs", [{"a": .5}, {"a": float("nan"), "b": .5},
                                  {"a": True, "b": 0}, {"a": .1, "b": .1}])
def test_invalid_distributions_are_failures(probs):
    result = score_task(probs, task())
    assert not result["valid"] and not result["correct"]


def row(original, probs=None, ok=True):
    result = score_task(probs, original) if ok else dict(valid=False, strict_valid=False, correct=False, probs=None)
    return {**result, "task_id": original.id, "model": "test", "ok": ok, "latency_s": .1,
            "cost_usd": None, "cost_basis": "unmeasured", "probs_source": "native",
            "timing": {"compile_seconds": .01, "forward_seconds": .08, "answer_seconds": .01}}


def test_failed_items_remain_in_accuracy_and_calibration_uses_top_probability():
    first, second = task(), task()
    second.id = "task-2"
    result = summarize_run([first, second], [row(first, {"a": .2, "b": .8}), row(second, ok=False)],
                           {first.id: "standard", second.id: "standard"})
    assert result["accuracy"] == .5
    assert result["brier_mean"] == pytest.approx(.08)
    assert result["ece"]["ece"] == pytest.approx(.2)
    assert result["calibration_n"] == 1 and result["n_scorable"] == 2
    assert result["official_leaderboard_score"] is None


def test_exact_gold_distribution_is_separate_from_hard_label_brier():
    original = task()
    original.provenance["gold_probs"] = {"a": .3, "b": .7}
    result = summarize_run([original], [row(original, {"a": .4, "b": .6})], {original.id: "hard"})
    assert result["gold_distribution"]["mean_tvd"] == pytest.approx(.1)
    assert result["gold_distribution"]["brier_mean"] == pytest.approx(.02)
    assert result["brier_mean"] == pytest.approx(.32)


def test_resume_recovers_only_incomplete_final_jsonl(tmp_path):
    from evaluation.jevbench.evaluate import read_predictions
    path = tmp_path / "predictions.jsonl"
    path.write_bytes(b'{"task_id":"a"}\n{"task')
    assert read_predictions(path, {"a", "b"}) == [{"task_id": "a"}]
    assert path.read_bytes() == b'{"task_id":"a"}\n'
    path.write_bytes(b'{"task_id":"a"}\ninvalid\n')
    with pytest.raises(ValueError, match="Corrupt"):
        read_predictions(path, {"a"})


def test_end_to_end_runner_coverage_and_resume_guard(tmp_path, monkeypatch):
    from evaluation.jevbench import evaluate
    from valen.data.schema import candidates
    original = task()
    monkeypatch.setattr(evaluate, "load_dataset", lambda *args: ([original],
        {"tasks": 1, "tiers": {original.id: "standard"}, "canonical_sha256": "test"}))
    calls = []

    class Model(torch.nn.Module):
        architecture = "qwen"
        model_name = "test"

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.))

        def forward(self, question):
            calls.append(question.qid)
            return torch.tensor([.8 if k == "b" else .2 for k in question.keys]).log()

    class Compiler:
        def compile(self, record):
            q = record["request"]["questions"]["decision"]
            pairs = candidates(q)
            question = SimpleNamespace(qid="decision", kind=q["type"], keys=[k for k, _ in pairs])
            return SimpleNamespace(questions=[question], logical_tokens=12, compute_tokens=12)

    monkeypatch.setattr(evaluate, "build_model", lambda config: Model())
    monkeypatch.setattr(evaluate, "build_compiler", lambda *args: Compiler())
    monkeypatch.setattr(evaluate, "load_checkpoint", lambda *args: {"progress": {"step": 10}})
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.pt").write_bytes(b"test")
    (checkpoint / "config.json").write_text(json.dumps({"architecture": "qwen", "model_path": "test", "stage": "joint"}))
    output = tmp_path / "evaluation"
    report = evaluate.run(checkpoint, "test", output, "cpu", warmup=0)
    assert report["complete"] and report["accuracy"] == 1.
    assert report["n_attempted"] == 1 and calls == ["decision"]
    evaluate.run(checkpoint, "test", output, "cpu", warmup=0, resume=True)
    assert calls == ["decision"]
    with pytest.raises(ValueError, match="changed"):
        evaluate.run(checkpoint, "test", output, "cpu", warmup=1, resume=True)


def test_loader_rejects_modified_snapshot_before_model_loading(tmp_path):
    from evaluation.jevbench.dataset import load_dataset
    (tmp_path / "public").mkdir()
    (tmp_path / "public/easy.jsonl").write_text(task().to_json() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(tmp_path, ["easy"])


def test_comparison_checks_matching_conditions_and_allows_equivalent_model_paths(tmp_path):
    from evaluation.jevbench.compare import compare
    original = task()
    records = [dict(row(original, {"a": .2, "b": .8}), expected="b")]
    report = summarize_run([original], records, {original.id: "standard"})
    report.update(smoke_subset=False, model="first", head="bilinear", benchmark_revision="test", hardware="cpu")
    manifest = dict(benchmark_revision="test", mapping_version=1, dataset={"tasks": 1},
                    task_ids_sha256="test", warmup_per_type=0, temperature=1., hardware="cpu",
                    code_sha256={}, versions={}, planned_tasks=1,
                    runtime_config={"model_path": "models/test", "architecture": "qwen", "stage": "joint"})
    paths = []
    for index in range(2):
        path = tmp_path / str(index)
        path.mkdir()
        current_report, current_manifest = deepcopy(report), deepcopy(manifest)
        if index:
            current_report.update(model="second", head="mlp")
            from pathlib import Path
            current_manifest["runtime_config"]["model_path"] = str(Path("models/test").resolve())
        (path / "metrics.json").write_text(json.dumps(current_report))
        (path / "run_manifest.json").write_text(json.dumps(current_manifest))
        (path / "predictions.jsonl").write_text(json.dumps(records[0]) + "\n")
        paths.append(path)
    result = compare(paths, tmp_path / "comparison")
    assert len(result["models"]) == 2
    current_manifest["hardware"] = "different gpu"
    (paths[1] / "run_manifest.json").write_text(json.dumps(current_manifest))
    with pytest.raises(ValueError, match="different"):
        compare(paths, tmp_path / "bad")
