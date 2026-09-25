"""Architecture contracts; adapters are plain Python objects, never model wrappers."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

import torch

from valen.data.types import QuestionSpec


class CompilerProtocol(Protocol):
    def compile(self, record, rng=None, labeled_only=False): ...


@dataclass(frozen=True)
class Capabilities:
    video: bool = False
    structured_descriptions: bool = False
    custom_noul_criteria: bool = False
    frozen_feature_cache: bool = False
    max_images: int | None = None


@dataclass
class Decision:
    question: QuestionSpec
    logits: torch.Tensor
    rollout: object = None


@dataclass
class PolicyInput:
    question: QuestionSpec
    logits: torch.Tensor
    reference_logits: torch.Tensor | None = None
    features: object = None


class ArchitectureBackend:
    """A training unit owns exactly one graph lifetime and one backward call."""
    name: str
    unit_scope: str
    capabilities: Capabilities

    def build_model(self, config):
        raise NotImplementedError

    def build_compiler(self, config, media_root) -> CompilerProtocol:
        raise NotImplementedError

    def candidates(self, question):
        raise NotImplementedError

    def optimizer_groups(self, model, config):
        raise NotImplementedError

    def inference_units(self, model, state):
        raise NotImplementedError

    def training_units(self, model, state, rollouts):
        raise NotImplementedError

    def question_logits(self, model, question, rollout=None):
        raise ValueError(f"{self.name} requires a complete state for forward")

    def policy_inputs(self, model, state, reference, cache_features=False):
        raise NotImplementedError

    def validate_state(self, config, state):
        pass

    def validate_feature_cache(self, model):
        raise ValueError(f"Frozen feature reuse is not supported by {self.name}")

    def make_reference(self, model, weights, cache_features=False):
        if cache_features:
            self.validate_feature_cache(model)
        reference = deepcopy(model)
        reference.load_state_dict(weights, strict=False)
        return reference.requires_grad_(False).eval()

    def validate_initialization(self, previous, config):
        if previous["architecture"] != config["architecture"] or self.base_paths(previous) != self.base_paths(config):
            raise ValueError("Initialization backbone/head mismatch")

    def base_paths(self, config):
        raise NotImplementedError

    def adaptation(self, model, config):
        raise NotImplementedError
