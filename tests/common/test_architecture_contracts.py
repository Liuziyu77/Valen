"""Behavioral boundaries that must survive architecture refactors."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from valen.configuration import flatten_config
from valen.modeling.factory import get_backend, normalize_model_config
from valen.training.runner import normalize_config
from valen.training.sft import SFTObjective


def test_new_config_layout_preserves_every_legacy_training_recipe():
    root = Path(__file__).resolve().parents[2] / "configs/train"
    for old in root.glob("*.json"):
        if old.name.startswith("dual_encoder_"):
            new = root / "dual_encoder/modernbert_dinov3b16" / old.name.replace("dual_encoder_", "").replace(".json", "_warmup.json")
        else:
            new = root / "qwen" / old.name
        assert normalize_config(json.loads(old.read_text())) == normalize_config(json.loads(new.read_text()))


@pytest.mark.parametrize("config", [
    {"model": {"architecture": "dual_encoder"}, "architecture": "qwen"},
    {"training": {"seed": 1}, "seed": 2},
    {"data": {"path": "x", "max_length": 20}, "max_length": 10},
    {"config_version": 3}, {"data": {}}, {"model": "qwen"},
])
def test_ambiguous_or_invalid_sectioned_config_is_rejected(config):
    with pytest.raises(ValueError):
        flatten_config(config)


def test_sectioned_config_can_be_normalized_repeatedly_without_mutation():
    config = {"config_version": 2, "model": {"architecture": "dual_encoder"},
              "data": {"path": "data.jsonl", "image_size": 384},
              "training": {"stage": "warmup"}, "objective": {"method": "sft", "brier_weight": .1}}
    before = json.dumps(config)
    flat = normalize_config(config)
    assert normalize_config(flat) == flat
    assert json.dumps(config) == before
    assert flat["data"] == "data.jsonl" and flat["image_size"] == 384


def test_legacy_model_and_compiler_imports_refer_to_canonical_classes():
    from valen.modeling.model import Valen, ValenQwen
    from valen.modeling.qwen.model import ValenQwen as CanonicalQwen
    from valen.modeling.dual_encoder import ValenDualEncoder
    from valen.modeling.dual_encoder.model import ValenDualEncoder as CanonicalDual
    from valen.data.compiler import Compiler, Question
    from valen.data.compilers.qwen import Compiler as CanonicalCompiler
    from valen.data.parallel_compiler import ParallelQuestion
    from valen.data.types import QuestionSpec
    assert Valen is ValenQwen is CanonicalQwen
    assert ValenDualEncoder is CanonicalDual
    assert Compiler is CanonicalCompiler
    assert ParallelQuestion is QuestionSpec
    question = Question("q", "noul", ["true", "false"], ["yes", "no"], ["branch"], [1., 0.], True)
    assert isinstance(question, QuestionSpec) and question.branches == ["branch"]


def test_qwen_next_forward_waits_for_previous_backward():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([.2, -.2]))
            self.live_graph = False
            self.forwards = 0

        def forward(self, question):
            assert not self.live_graph, "The previous question graph is still waiting for backward"
            self.live_graph = True
            self.forwards += 1
            logits = self.weight * self.forwards
            logits.register_hook(lambda grad: setattr(self, "live_graph", False))
            return logits

    model = Model()
    q = SimpleNamespace(kind="noul", target=[1., 0.])
    state = SimpleNamespace(questions=[q] * 4)
    objective = SFTObjective({})
    for unit in get_backend("qwen").training_units(model, state, [None] * 4):
        assert len(unit) == 1
        d = unit[0]
        loss, _ = objective.loss_from_logits(d.logits, d.question)
        loss.backward()
    assert model.forwards == 4 and not model.live_graph


def test_dual_state_has_one_forward_and_one_shared_backward():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([.2, -.2]))
            self.forwards = 0
            self.backwards = 0

        def forward_state(self, state):
            self.forwards += 1
            shared = self.weight.square()
            shared.register_hook(lambda grad: setattr(self, "backwards", self.backwards + 1))
            return [shared * (i + 1) for i in range(len(state.questions))]

    q = SimpleNamespace(kind="noul", target=[1., 0.])
    state, model = SimpleNamespace(questions=[q] * 4), Model()
    objective = SFTObjective({})
    units = get_backend("dual_encoder").training_units(model, state, [None] * 4)
    for unit in units:
        assert len(unit) == 4
        torch.stack([objective.loss_from_logits(d.logits, d.question)[0] for d in unit]).mean().backward()
    assert model.forwards == model.backwards == 1


def test_capabilities_preserve_distinct_noul_contracts():
    qwen, dual = get_backend("qwen"), get_backend("dual_encoder")
    question = {"type": "noul", "instructions": "Is it red?", "criteria": {"true": "red", "false": "blue"}}
    assert qwen.capabilities.video and not dual.capabilities.video
    assert dual.capabilities.custom_noul_criteria and not qwen.capabilities.custom_noul_criteria
    assert dual.candidates(question) == [("true", "red"), ("false", "blue")]
    assert qwen.candidates(question) != dual.candidates(question)
    with pytest.raises(ValueError, match="not supported"):
        dual.validate_feature_cache(None)


@pytest.mark.parametrize("architecture", ["qwen", "dual_encoder"])
def test_sokoban_policy_uses_architecture_factories(tmp_path, monkeypatch, architecture):
    from valen.modeling import factory
    from valen.training import checkpoint
    from evaluation.sokoban.evaluate import ValenPolicy
    config = {"config_version": 2, "model": {"architecture": architecture}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    calls = []
    monkeypatch.setattr(factory, "build_model", lambda c: torch.nn.Linear(1, 1))
    compiler = object()
    def build(c, media_root):
        calls.append((c["architecture"], c["device"], media_root))
        return compiler
    monkeypatch.setattr(factory, "build_compiler", build)
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda *args: None)
    policy = ValenPolicy(tmp_path, device="cpu")
    assert policy.compiler is compiler
    assert calls == [(architecture, "cpu", ".")]
