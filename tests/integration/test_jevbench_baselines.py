"""Native baseline contracts, integrity checks and end-to-end journal auditing."""
from copy import deepcopy
import hashlib
import json

import pytest

from evaluation.jevbench import baselines
from evaluation.jevbench._upstream.tasks import Task
from evaluation.jevbench._upstream.scoring import score_task
from evaluation.jevbench.metrics import summarize_run


def make_task(kind="choice", identifier="a"):
    criteria, labels, expected = {"b": "Second", "a": "First"}, ["a", "b"], "b"
    if kind == "noul":
        criteria, labels, expected = {"true": "Confirmed", "false": "Unconfirmed"}, ["no", "yes"], "yes"
    elif kind == "score":
        criteria, labels, expected = ["Low", "Middle", "High"], ["0", "1", "2"], 2
    return Task(identifier, "family", {"z": "Evidence", "a": [1]},
                {"type": kind, "instructions": "Apply rubric", "criteria": criteria,
                 "answer": {"label": "SECRET"}}, labels, expected, "public",
                provenance={"rationale": "SECRET GOLD"})


def test_native_input_excludes_gold_and_preserves_state_and_option_order():
    task = make_task()
    original = deepcopy(task)
    request = baselines.native_request(task)
    assert list(request["state"]) == ["z", "a"]
    assert list(request["questions"]["decision"]["criteria"]) == ["b", "a"]
    assert "SECRET" not in json.dumps(request)
    request["state"]["a"].append(2)
    assert task == original
    assert baselines.native_request(make_task("noul"))["questions"]["decision"]["criteria"] == {
        "true": "Confirmed", "false": "Unconfirmed"}


def test_native_noul_and_score_use_probability_distributions():
    task = make_task("noul")
    response = {"answers": {"decision": {"type": "noul", "noul": .7}}}
    assert baselines.native_probabilities(task, response) == pytest.approx({"no": .3, "yes": .7})
    response["answers"]["decision"]["noul"] = float("nan")
    with pytest.raises(ValueError, match="outside"):
        baselines.native_probabilities(task, response)
    task = make_task("score")
    response = {"answers": {"decision": {"type": "score", "score": 1.4,
                                          "probabilities": {"0": .2, "1": .2, "2": .6}}}}
    assert score_task(baselines.native_probabilities(task, response), task)["predicted"] == "2"
    response["answers"]["decision"]["type"] = "choice"
    with pytest.raises(ValueError, match="type differs"):
        baselines.native_probabilities(task, response)


def test_model_fingerprint_rejects_wrong_weight_hash(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"correct downloaded content")
    manifest = {"repo_id": "test/model", "revision": "fixed", "files": [
        {"path": weights.name, "size": weights.stat().st_size,
         "lfs": {"sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}}]}
    (tmp_path / "download_manifest.json").write_text(json.dumps(manifest))
    assert baselines.fingerprint_model(tmp_path)["download"]["revision"] == "fixed"
    weights.write_bytes(b"damaged downloaded content")
    with pytest.raises(ValueError, match="hash mismatch|size mismatch"):
        baselines.fingerprint_model(tmp_path)


def test_baseline_runner_failures_resume_and_source_guard(tmp_path, monkeypatch):
    tasks = [make_task(identifier="a"), make_task(identifier="b")]
    data = {"tasks": 2, "tiers": {t.id: "standard" for t in tasks}, "canonical_sha256": "test"}
    monkeypatch.setattr(baselines, "load_dataset", lambda *args: (tasks, data))
    monkeypatch.setattr(baselines, "fingerprint_model", lambda *args: {"download": {"repo_id": "fake/laya"}})
    monkeypatch.setattr(baselines, "fingerprint_source", lambda *args: {"revision": "fixed"})
    calls = []
    def predict(request):
        calls.append(request)
        if len(calls) == 2:
            raise RuntimeError("Simulated native failure")
        return {"answers": {"decision": {"type": "choice", "probabilities": {"a": .2, "b": .8}}},
                "usage": {"truncated": True}}
    monkeypatch.setattr(baselines, "load_adapter", lambda *args: (predict, {"temperature": [1, 1, 1]}))
    output = tmp_path / "run"
    report = baselines.run("laya", tmp_path, tmp_path, "fake", output, device="cpu", warmup=0)
    assert report["complete"] and report["accuracy"] == .5 and report["n_valid"] == 1
    assert report["truncation"]["state_truncated"] == 1
    assert report["forward_seconds"]["mean_s"] is None
    baselines.run("laya", tmp_path, tmp_path, "fake", output, device="cpu", warmup=0, resume=True)
    assert len(calls) == 2
    monkeypatch.setattr(baselines, "fingerprint_source", lambda *args: {"revision": "changed"})
    with pytest.raises(ValueError, match="identical manifest"):
        baselines.run("laya", tmp_path, tmp_path, "fake", output, device="cpu", warmup=0, resume=True)


def test_cross_model_comparison_audits_each_response_and_library_conditions(tmp_path, monkeypatch):
    from evaluation.jevbench import compare_models
    tasks = [make_task(kind, kind) for kind in ("choice", "noul", "score")]
    tiers = dict(zip((t.id for t in tasks), ("easy", "standard", "hard")))
    data = {"tasks": 3, "tiers": tiers, "canonical_sha256": "test"}
    monkeypatch.setattr(compare_models, "load_dataset", lambda *args: (tasks, data))
    paths = []
    for index in range(2):
        records = []
        for task in tasks:
            probs = {k: .1 for k in task.labels}
            probs[str(task.expected)] = 1 - .1*(len(probs)-1)
            response = {"answers": {"decision": {"type": task.question["type"], "probabilities": probs}}}
            if task.question["type"] == "noul":
                response["answers"]["decision"]["noul"] = probs["yes"]
                probs = baselines.native_probabilities(task, response)
            records.append({**score_task(probs, task), "ok": True, "task_id": task.id,
                            "expected": task.expected, "model": str(index), "latency_s": .1,
                            "cost_usd": None, "cost_basis": "local", "probs_source": "native", "timing": {},
                            "probs_as_returned": probs, "response": response, "request": baselines.native_request(task)})
        report = summarize_run(tasks, records, tiers)
        report.update(smoke_subset=False, model=str(index), benchmark_revision="fixed", hardware="cpu")
        manifest = dict(benchmark_revision="fixed", mapping_version=baselines.BASELINE_MAPPING_VERSION,
                        dataset=data, planned_tasks=3, task_ids_sha256="same", warmup_per_type=1,
                        hardware="cpu", code_sha256={}, versions={"torch": "same"})
        path = tmp_path / str(index)
        path.mkdir()
        for name, content in (("metrics", report), ("run_manifest", manifest), ("complete", {"tasks": 3})):
            (path / (name+".json")).write_text(json.dumps(content))
        (path / "predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
        paths.append(path)
    result = compare_models.compare_models(paths, "fake", tmp_path / "comparison")
    assert len(result["audit"]) == 2 and result["paired_outcomes"][0]["first_only_correct"] == 0
    records[0]["request"]["state"]["a"].append("WRONG EVIDENCE")
    (paths[1] / "predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    with pytest.raises(ValueError, match="input evidence"):
        compare_models.compare_models(paths, "fake", tmp_path / "corrupt")
