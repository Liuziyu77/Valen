import math
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from visualjev.evaluation.metrics import question_metrics, summarize


def test_metrics_match_known_probabilities():
    question = SimpleNamespace(kind="score", keys=["0", "1", "2"], target=[0, 0, 1])
    metrics = question_metrics(question, torch.tensor([.2, .3, .5]).log())
    assert metrics["nll"] == pytest.approx(math.log(2))
    assert metrics["brier"] == pytest.approx(.38)
    assert metrics["expected_score_mae"] == pytest.approx(.7)
    assert metrics["rps"] == pytest.approx(.145)
    assert metrics["accuracy"] == 1


def test_binary_f1_uses_candidate_keys_and_task_specific_denominator():
    rows = []
    for keys, target, probs in [(["false", "true"], [0, 1], [.1, .9]),
                                (["true", "false"], [0, 1], [.8, .2])]:
        question = SimpleNamespace(kind="noul", keys=keys, target=target)
        rows.append(dict(task="noul", modality="text", language="zh",
                         metrics=question_metrics(question, torch.tensor(probs).log())))
    rows.append(dict(task="choice", modality="image", language="en", metrics={"accuracy": 1., "nll": 0.}))
    result = summarize(rows)
    assert result["task/noul"]["macro_f1"] == pytest.approx(1 / 3)
    assert result["overall"]["accuracy"] == pytest.approx(2 / 3)
    assert result["overall"]["metric_counts"]["brier"] == 2
    assert result["task/noul"]["confusion"]["false_positive"] == 1


def test_soft_targets_do_not_get_hard_accuracy():
    question = SimpleNamespace(kind="choice", keys=["a", "b"], target=[.4, .6])
    assert "accuracy" not in question_metrics(question, torch.zeros(2))
    with pytest.raises(ValueError, match="Non-finite"):
        question_metrics(question, torch.tensor([float("nan"), 1.]))


def test_cli_uses_explicit_dataset_despite_legacy_request_file(tmp_path, monkeypatch):
    from visualjev.evaluation import evaluate
    (tmp_path / "evaluation_request.json").write_text(json.dumps({"data": "only_visual_200.jsonl"}))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", "checkpoint", "--data", "old_test.jsonl", "--output", str(tmp_path)])
    calls = []
    monkeypatch.setattr(evaluate, "run", lambda checkpoint, data, output, device: calls.append(data))
    monkeypatch.setattr(evaluate, "close", lambda: None)
    evaluate.main()
    assert calls == ["old_test.jsonl"]
