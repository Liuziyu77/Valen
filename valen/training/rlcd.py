"""Valen RLCD experiment: GRPO over categorical decisions, with calibration.

This is an explicit local design, not TypeSafe's unpublished RLCD implementation.
Rollouts and rewards are frozen for all policy updates in one group batch.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import torch


DEFAULTS = {
    "group_size": 16,
    "num_iterations": 2,
    "correctness_weight": 1.0,
    "confidence_weight": 1.0,
    "brier_weight": 1.0,
    "beta": 0.02,
    "clip_epsilon": 0.2,
    "advantage_epsilon": 1e-6,
}


def validate_options(options):
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown RLCD options: {sorted(unknown)}")
    result = dict(DEFAULTS, **options)
    for name, minimum in (("group_size", 2), ("num_iterations", 1)):
        if type(result[name]) is not int or result[name] < minimum:
            raise ValueError(f"rlcd.{name} must be an integer >= {minimum}")
    for name in set(DEFAULTS) - {"group_size", "num_iterations"}:
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"rlcd.{name} must be finite and nonnegative")
    if not 0 < result["clip_epsilon"] < 1 or result["advantage_epsilon"] <= 0:
        raise ValueError("RLCD requires 0 < clip_epsilon < 1 and advantage_epsilon > 0")
    if result["correctness_weight"] == result["confidence_weight"] == 0:
        raise ValueError("At least one RLCD reward weight must be positive")
    return result


def decision_reward(probabilities, target, actions, correctness_weight=1.0, confidence_weight=1.0):
    """Return detached rewards; soft labels use expected binary squared error."""
    confidence = probabilities.detach()[actions]
    correctness = torch.as_tensor(target, device=confidence.device, dtype=confidence.dtype).detach()[actions]
    error = correctness * (1 - confidence).square() + (1 - correctness) * confidence.square()
    reward = correctness_weight * correctness - confidence_weight * error
    return reward, correctness, confidence, error


def group_advantages(rewards, epsilon=1e-6):
    rewards = rewards.detach()
    centered = rewards - rewards.mean()
    std = rewards.std(unbiased=False)
    # Constant-reward groups must be exactly zero (including FP rounding).
    if torch.all(rewards == rewards[0]):
        return torch.zeros_like(rewards)
    return centered / (std + epsilon)


def clipped_policy_loss(log_probs, actions, old_log_probs, advantages, epsilon):
    ratio = (log_probs[actions] - old_log_probs.detach()).exp()
    advantages = advantages.detach()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    # Fraction for which the clipped branch actually determines the objective.
    clip_fraction = (clipped < unclipped).float().mean().detach()
    return loss, clip_fraction


@dataclass
class Rollout:
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    correctness: torch.Tensor
    confidence: torch.Tensor
    confidence_error: torch.Tensor
    reference_log_probs: torch.Tensor | None
    features: list | None = None


class RLCDObjective:
    metric_names = ("policy_loss", "brier", "kl", "reward", "sample_correctness",
                    "sample_confidence", "confidence_error", "reward_std",
                    "zero_advantage_group", "clip_fraction")

    def __init__(self, model, options, state=None, resuming=False, cache_frozen_features=False):
        self.options = validate_options(options)
        self.num_iterations = self.options["num_iterations"]
        self.reference = None
        self.reference_weights = None
        self.cache_features = cache_frozen_features
        self.cache_verified = False
        if self.cache_features and (not hasattr(model, "extract_features") or
                                    any(p.requires_grad for p in model.backbone.parameters())):
            raise ValueError("Feature reuse requires a fully frozen Valen backbone")
        if self.options["beta"]:
            names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
            if resuming:
                if not state or set(state.get("reference_weights", {})) != names:
                    raise ValueError("RLCD resume requires the original frozen reference weights")
                self.reference_weights = state["reference_weights"]
            else:
                self.reference_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}
            if self.cache_features:
                if any(not name.startswith("head.") for name in names):
                    raise ValueError("Feature reuse requires only head parameters to be trainable")
                self.reference = deepcopy(model.head)
                self.reference.load_state_dict({k.removeprefix("head."): v for k, v in self.reference_weights.items()})
            else:
                self.reference = deepcopy(model)
                self.reference.load_state_dict(self.reference_weights, strict=False)
            self.reference.requires_grad_(False).eval()

    @torch.no_grad()
    def prepare(self, model, pack):
        # Deterministic forwards: dropout is disabled by the runner, while train
        # mode remains enabled for gradient checkpointing in subsequent updates.
        result = []
        for state in pack:
            group = []
            for question in state.questions:
                features = model.extract_features(question) if self.cache_features else None
                logits = (model.score_features(features) if features is not None else model(question)).float()
                if self.cache_features and not self.cache_verified:
                    torch.testing.assert_close(logits, model(question).float(), rtol=0, atol=0)
                    self.cache_verified = True
                if logits.ndim != 1 or not torch.isfinite(logits).all():
                    raise ValueError(f"Expected finite candidate logits for {question.qid}")
                target = torch.as_tensor(question.target, dtype=logits.dtype, device=logits.device)
                if target.shape != logits.shape or not torch.isfinite(target).all() or (target < 0).any() or not torch.isclose(target.sum(), logits.new_tensor(1.), atol=1e-5):
                    raise ValueError(f"Expected a label distribution for {question.qid}")
                log_probs = logits.log_softmax(-1)
                actions = torch.multinomial(log_probs.exp(), self.options["group_size"], replacement=True)
                reward, correctness, confidence, error = decision_reward(
                    log_probs.exp(), target, actions, self.options["correctness_weight"], self.options["confidence_weight"])
                reference = None
                if self.reference is not None:
                    reference = (model.score_features(features, self.reference) if features is not None
                                 else self.reference(question)).float().log_softmax(-1)
                group.append(Rollout(actions, log_probs[actions], group_advantages(reward, self.options["advantage_epsilon"]),
                                     reward, correctness, confidence, error, reference, features))
            result.append(group)
        return result

    def loss(self, model, question, rollout):
        logits = model.score_features(rollout.features) if rollout.features is not None else model(question)
        log_probs = logits.float().log_softmax(-1)
        policy_loss, clip_fraction = clipped_policy_loss(
            log_probs, rollout.actions, rollout.old_log_probs, rollout.advantages, self.options["clip_epsilon"])
        probs = log_probs.exp()
        target = torch.as_tensor(question.target, device=probs.device, dtype=probs.dtype)
        brier = (probs - target).square().sum()
        # Exact categorical KL, rather than a noisy sampled token estimator.
        kl = (probs * (log_probs - rollout.reference_log_probs)).sum() if self.reference is not None else probs.new_zeros(())
        loss = policy_loss + self.options["beta"] * kl + self.options["brier_weight"] * brier
        terms = {"policy_loss": policy_loss, "brier": brier, "kl": kl,
                 "reward": rollout.rewards.mean(), "sample_correctness": rollout.correctness.mean(),
                 "sample_confidence": rollout.confidence.mean(), "confidence_error": rollout.confidence_error.mean(),
                 "reward_std": rollout.rewards.std(unbiased=False),
                 "zero_advantage_group": (rollout.advantages == 0).all().float(), "clip_fraction": clip_fraction}
        return loss, {key: value.detach() for key, value in terms.items()}

    def state_dict(self):
        return {"reference_weights": self.reference_weights} if self.reference is not None else {}
