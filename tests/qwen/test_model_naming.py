"""Public model identity and compatibility with existing decision-head weights."""
from types import SimpleNamespace

import pytest
import torch

from valen import MODEL_NAME
from valen.evaluation.inference import answer, predict
from valen.modeling.model import Valen, ValenQwen, build_model
from valen.training.checkpoint import load_checkpoint


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
        self.embedding = torch.nn.Embedding(10, 8)
        self.requires_grad_(False)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


@pytest.mark.parametrize("kind", ["choice", "noul", "score"])
def test_response_uses_valen_qwen_without_changing_decisions_or_usage(kind):
    model = ValenQwen(Backbone(), projection_dim=4)
    keys = ["false", "true"] if kind == "noul" else ["0", "1"]
    question = SimpleNamespace(
        qid="decision", kind=kind, keys=keys, descriptions=["low", "high"],
        branches=[SimpleNamespace(inputs={"input_ids": torch.tensor([[1, 2, 3]])},
                                  candidate_positions=[0, 1], decision_position=2)])
    compiled = SimpleNamespace(questions=[question], logical_tokens=3, compute_tokens=3)
    expected_answer = answer(question, model(question), temperature=1.3)
    response = predict(model, compiled, temperature=1.3)
    assert response == {"model": "Valen", "answers": {"decision": expected_answer},
                        "usage": {"input_tokens": 3, "output_tokens": 0},
                        "internal_usage": {"compute_tokens": 3}}
    assert model.model_name == MODEL_NAME == "Valen"
    assert model.architecture == "qwen"

    # A model-specific identity takes precedence over the Qwen default.
    model.model_name = "custom-checkpoint"
    assert predict(model, compiled)["model"] == "custom-checkpoint"


def test_checkpoint_parameter_keys_remain_compatible(tmp_path):
    model = ValenQwen(Backbone(), projection_dim=4)
    frozen = model.backbone.embedding.weight.detach().clone()
    # The pre-rename checkpoint stores named tensors, not the model class.
    weights = {"head.decision.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8),
               "head.candidate.weight": torch.full((4, 8), 0.25)}
    torch.save({"weights": weights, "progress": {"epoch": 1, "step": 147},
                "base_manifest": None}, tmp_path / "checkpoint.pt")
    payload = load_checkpoint(tmp_path, model)
    assert type(model).__name__ == "ValenQwen"
    assert payload["progress"] == {"epoch": 1, "step": 147}
    for key, value in weights.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(model.backbone.embedding.weight, frozen, rtol=0, atol=0)


def test_legacy_import_is_the_same_model_class():
    assert Valen is ValenQwen


@pytest.mark.parametrize("architecture", [None, "qwen"])
def test_build_model_defaults_legacy_configs_to_qwen(tmp_path, monkeypatch, architecture):
    from transformers import Qwen3_5ForConditionalGeneration

    monkeypatch.setattr(Qwen3_5ForConditionalGeneration, "from_pretrained",
                        lambda *args, **kwargs: (SimpleNamespace(model=Backbone()), {}))
    config = {"model_path": str(tmp_path), "stage": "warmup", "device": "cpu"}
    if architecture is not None:
        config["architecture"] = architecture
    model = build_model(config)
    assert type(model) is ValenQwen
    assert model.model_name == "Valen"


@pytest.mark.parametrize("architecture", ["unknown_encoder", "qwne", "", None])
def test_unsupported_architecture_fails_before_loading_weights(architecture):
    from valen.training.runner import normalize_config

    config = {"architecture": architecture}
    with pytest.raises(ValueError, match="Unsupported architecture"):
        build_model(config)
    with pytest.raises(ValueError, match="Unsupported architecture"):
        normalize_config(config)
