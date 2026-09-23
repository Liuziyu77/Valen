from types import SimpleNamespace

import pytest
import torch

from visionjev.training.rlcd import (
    RLCDObjective, clipped_policy_loss, decision_reward, group_advantages, validate_options,
)
from visionjev.training.runner import run


def test_reward_penalizes_confident_errors_and_rewards_confident_correct_answers():
    target = [1., 0.]
    actions = torch.tensor([0, 1])
    confident, *_ = decision_reward(torch.tensor([.9, .1]), target, actions)
    uncertain, *_ = decision_reward(torch.tensor([.6, .4]), target, actions)
    wrong, *_ = decision_reward(torch.tensor([.1, .9]), target, actions)
    assert confident.tolist() == pytest.approx([.99, -.01])
    assert confident[0] > uncertain[0]
    assert wrong[1] < uncertain[1]
    reward, _, _, error = decision_reward(torch.tensor([.2, .8]), [.3, .7], actions)
    torch.testing.assert_close(error, torch.tensor([.3 * .8**2 + .7 * .2**2, .7 * .2**2 + .3 * .8**2]))
    torch.testing.assert_close(reward, torch.tensor([.3, .7]) - error)


def test_advantages_are_detached_and_constant_groups_exactly_zero():
    reward = torch.tensor([1., -1., 0.], requires_grad=True)
    advantage = group_advantages(reward)
    assert not advantage.requires_grad
    assert advantage.mean().item() == pytest.approx(0., abs=1e-7)
    for constant in (0., .3, -.9):
        assert not group_advantages(torch.full((16,), constant)).any()


def test_grpo_clips_both_advantage_signs_and_detaches_old_policy():
    # Both positive/high-ratio and negative/low-ratio samples hit their bounds.
    logs = torch.tensor([.8, .2]).log().requires_grad_()
    old = torch.tensor([.5, .5]).log().requires_grad_()
    advantages = torch.tensor([1., -1.], requires_grad=True)
    loss, clipped = clipped_policy_loss(logs, torch.tensor([0, 1]), old, advantages, .2)
    assert loss.item() == pytest.approx(-.2)
    assert clipped.item() == 1.
    loss.backward()
    assert not logs.grad.any()
    assert old.grad is None and advantages.grad is None


def test_grpo_gradient_increases_rewarded_candidate_probability():
    logits = torch.zeros(2, requires_grad=True)
    logs = logits.log_softmax(0)
    loss, _ = clipped_policy_loss(logs, torch.tensor([0, 1]), logs.detach(), torch.tensor([1., -1.]), .2)
    loss.backward()
    assert logits.grad[0] < 0 < logits.grad[1]


class Decision(torch.nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(probabilities).log())

    def forward(self, question):
        return self.logits


@pytest.mark.parametrize("kind,probs,target", [
    ("choice", [.1, .2, .7], [1., 0., 0.]),
    ("noul", [.1, .9], [1., 0.]),
    ("score", [.1, .2, .7], [.2, .6, .2]),
])
def test_all_outputs_have_finite_loss_and_frozen_rollouts(kind, probs, target):
    model = Decision(probs)
    q = SimpleNamespace(qid="q", kind=kind, target=target)
    objective = RLCDObjective(model, {})
    rollout = objective.prepare(model, [SimpleNamespace(questions=[q])])[0][0]
    original = rollout.old_log_probs.clone()
    loss, terms = objective.loss(model, q, rollout)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(model.logits.grad).all()
    assert all(not value.requires_grad for value in vars(rollout).values() if isinstance(value, torch.Tensor))
    assert all(p.grad is None and not p.requires_grad for p in objective.reference.parameters())
    with torch.no_grad():
        model.logits[0] += .5
    _, terms = objective.loss(model, q, rollout)
    assert terms["kl"] > 0
    torch.testing.assert_close(original, rollout.old_log_probs)


def test_brier_keeps_signal_when_group_has_only_one_sampled_answer(monkeypatch):
    model = Decision([.2, .8])
    q = SimpleNamespace(qid="q", kind="noul", target=[1., 0.])
    monkeypatch.setattr(torch, "multinomial", lambda p, n, replacement: torch.ones(n, dtype=torch.long))
    objective = RLCDObjective(model, {"beta": 0})
    rollout = objective.prepare(model, [SimpleNamespace(questions=[q])])[0][0]
    assert not rollout.advantages.any()
    loss, terms = objective.loss(model, q, rollout)
    assert terms["policy_loss"] == 0 and terms["zero_advantage_group"] == 1
    loss.backward()
    assert model.logits.grad[0] < 0 < model.logits.grad[1]
    pure_grpo = RLCDObjective(model, {"beta": 0, "brier_weight": 0})
    model.zero_grad()
    pure_grpo.loss(model, q, rollout)[0].backward()
    assert not model.logits.grad.any()


@pytest.mark.parametrize("options", [{"group_size": 1}, {"num_iterations": 0}, {"beta": -1},
                                     {"clip_epsilon": 1}, {"advantage_epsilon": 0},
                                     {"confidence_weight": float("nan")}, {"typo": 1}])
def test_bad_options_fail_early(options):
    with pytest.raises(ValueError):
        validate_options(options)


def test_rlcd_requires_a_checkpoint_before_loading_device():
    with pytest.raises(ValueError, match="requires --initialize"):
        run({"method": "rlcd", "device": "not_a_device"})
