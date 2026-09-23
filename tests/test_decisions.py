import math
from types import SimpleNamespace
import pytest
import torch
from visualjev.evaluation.inference import answer
from visualjev.training.sft import question_loss
from visualjev.modeling.model import DecisionHead
from visualjev.data.schema import target_distribution, candidates


def test_soft_cross_entropy_and_rps():
    logits = torch.tensor([0.2, -0.4, 1.3], requires_grad=True)
    target = [0.1, 0.3, 0.6]
    loss, terms = question_loss(logits, target, "score", rps_weight=0.2)
    p = logits.softmax(0)
    expected_ce = -sum(y * torch.log(x) for y, x in zip(target, p))
    expected_rps = ((p[0] - .1)**2 + (p[0] + p[1] - .4)**2) / 2
    torch.testing.assert_close(loss, expected_ce + .2 * expected_rps)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["choice", "noul", "score"])
def test_default_sft_matches_label_cross_entropy(kind):
    logits = torch.tensor([1., -1.], requires_grad=True)
    loss, terms = question_loss(logits, [0., 1.], kind)
    expected = torch.nn.functional.cross_entropy(logits[None], torch.tensor([1]))
    torch.testing.assert_close(loss, expected)
    gradient, = torch.autograd.grad(loss, logits, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(gradient, expected_gradient)
    assert set(terms) == {"ce", "rps"}


def test_rps_respects_order():
    near, _ = question_loss(torch.tensor([-20., 20., -20.]), [1, 0, 0], "score", 1)
    far, _ = question_loss(torch.tensor([-20., -20., 20.]), [1, 0, 0], "score", 1)
    assert far > near


def test_targets_follow_candidate_order():
    assert target_distribution({"probabilities": {"a": .8, "b": .2}}, ["b", "a"]) == [.2, .8]
    assert target_distribution({"probabilities": None}, ["a"]) is None


@pytest.mark.parametrize("probs", [{"a": .3}, {"a": float("nan")}, {"a": -1}, {"b": 1}, {"a": True}])
def test_invalid_targets(probs):
    with pytest.raises(ValueError):
        target_distribution({"probabilities": probs}, ["a"])


def test_score_bounds_and_choice_limits():
    with pytest.raises(ValueError):
        candidates({"type": "score", "instructions": "x", "criteria": ["only"]})
    with pytest.raises(ValueError):
        candidates({"type": "choice", "instructions": "x", "criteria": {str(i): "x" for i in range(256)}})


def test_responses():
    q = SimpleNamespace(kind="score", keys=["0", "1", "2"], descriptions=["low", "mid", "high"])
    out = answer(q, torch.tensor([.1, .3, .6]).log())
    assert out["score"] == pytest.approx(1.5)
    assert out["confidence"] == pytest.approx(.25)
    q.kind = "choice"
    assert answer(q, torch.zeros(3))["confidence"] == pytest.approx(0)
    q.keys, q.kind = ["false", "true"], "noul"
    assert answer(q, torch.tensor([.2, .8]).log())["noul"] == pytest.approx(.8)
    with pytest.raises(ValueError):
        answer(q, torch.zeros(2), temperature=0)


def test_head_gradient_reaches_both_projections_and_backbone():
    head = DecisionHead(8, 4)
    hidden = torch.randn(1, 10, 8, requires_grad=True)
    logits = head(hidden, [3, 6], 9)
    torch.nn.functional.cross_entropy(logits[None], torch.tensor([1])).backward()
    assert head.decision.weight.grad.abs().sum() > 0
    assert head.candidate.weight.grad.abs().sum() > 0
    assert hidden.grad[0, 9].abs().sum() > 0
    assert hidden.grad[0, 3].abs().sum() > 0
