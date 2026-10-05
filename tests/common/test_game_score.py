import json
import pytest

from evaluation.game.score import score


def test_tied_optimal_actions_and_revision_buckets(tmp_path):
    data, predictions = tmp_path/"eval.jsonl", tmp_path/"predictions.jsonl"
    target = {"up": .5, "down": 0., "left": .5, "right": 0.}
    record = {"targets": {"next_action": {"probabilities": target}}, "meta": {"environment": "sokoban"}}
    prediction = {"record_index": 0, "qid": "next_action", "target": target,
                  "probabilities": {"up": .2, "down": .1, "left": .6, "right": .1}}
    data.write_text(json.dumps(record)+"\n")
    predictions.write_text(json.dumps(prediction)+"\n")
    report = score(data, predictions, "game-v4")
    assert report["revision"] == "game-v4"
    assert report["metrics"]["sokoban"]["optimal_action_accuracy"] == 1.
    assert report["metrics"]["overall"]["optimal_probability_mass"] == pytest.approx(.8)
    predictions.write_text(json.dumps(prediction)+"\n"+json.dumps(prediction)+"\n")
    with pytest.raises(ValueError, match="duplicated"):
        score(data, predictions, "game-v4")
