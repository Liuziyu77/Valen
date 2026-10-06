"""Check released-model evaluation without downloading weights."""
import json
from types import SimpleNamespace

import pytest
import torch

from evaluation.visualdecisionbench.evaluate import evaluate


class FakeModel:
    def __init__(self):
        self.seen = []

    def eval(self):
        return self

    def make_compiler(self, media_root, execution):
        def compile(record):
            self.seen.append(record)
            questions = []
            for qid, question in record["request"]["questions"].items():
                if question["type"] == "noul":
                    keys = ["true", "false"]
                elif question["type"] == "score":
                    keys = [str(i) for i in range(len(question["criteria"]))]
                else:
                    keys = list(question["criteria"])
                questions.append(SimpleNamespace(qid=qid, kind=question["type"], keys=keys,
                                                  descriptions=keys))
            return SimpleNamespace(questions=questions, inputs={} if execution == "shared_state" else None)

        return SimpleNamespace(compile=compile, media_kwargs={"videos_kwargs": {"num_frames": 16}})

    def __call__(self, question):
        probabilities = ([1 / len(question.keys)] * len(question.keys) if question.kind == "score"
                         else [.1, .9] if question.kind == "noul" else [.8, .2])
        return torch.tensor(probabilities).log()

    def forward_state(self, compiled):
        return [self(question) for question in compiled.questions]


@pytest.mark.parametrize("execution", ["question", "shared_state"])
def test_subsets_hard_accuracy_soft_score_and_target_separation(tmp_path, execution):
    choice = {"type": "choice", "instructions": "Select A", "criteria": {"a": "A", "b": "B"}}
    score = {"type": "score", "instructions": "Rate", "criteria": [str(i) for i in range(12)]}
    soft = {str(i): .25 if i == 0 else .75 if i == 11 else 0.0 for i in range(12)}
    records = [
        {"request": {"state": "Image state", "questions": {"same_id": choice, "truth": {"type": "noul", "instructions": "True"}, "rating": score}},
         "targets": {"same_id": {"probabilities": {"a": 1, "b": 0}}, "truth": {"probabilities": {"true": 1, "false": 0}}, "rating": {"probabilities": soft}},
         "meta": {"benchmark_subset": "visualdecisionbench_image", "modality": "image"}},
        {"request": {"state": "Video state", "questions": {"same_id": choice}},
         "targets": {"same_id": {"probabilities": {"a": 1, "b": 0}}},
         "meta": {"benchmark_subset": "visualdecisionbench_video", "modality": "video"}},
    ]
    (tmp_path / "eval.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    model = FakeModel()
    output = tmp_path / "results"
    report = evaluate(model, tmp_path, output, device="cpu", execution=execution)
    assert report["records"] == 2 and report["questions"] == 4
    assert report["metrics"]["overall"]["accuracy"] == pytest.approx(2 / 3)
    assert report["metrics"]["overall"]["metric_counts"]["accuracy"] == 3
    assert report["subsets"]["visualdecisionbench_image"]["overall"]["accuracy"] == .5
    assert report["subsets"]["visualdecisionbench_video"]["overall"]["accuracy"] == 1
    assert "accuracy" not in report["metrics"]["task/score"]
    assert report["metrics"]["task/score"]["expected_score_mae"] == pytest.approx(2.75)
    assert report["metrics"]["task/noul"]["confusion"]["false_negative"] == 1
    assert all(set(row) == {"request", "assets"} for row in model.seen)
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert len({(row["record_index"], row["qid"]) for row in predictions}) == 4
    assert predictions[2]["target"] == soft
    assert report["evaluation_seconds"] > 0


def test_missing_label_does_not_silently_shrink_denominator(tmp_path):
    row = {"request": {"state": "State", "questions": {"q": {"type": "noul", "instructions": "True"}}}}
    (tmp_path / "eval.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="every question requires a target"):
        evaluate(FakeModel(), tmp_path, tmp_path / "results", device="cpu")
