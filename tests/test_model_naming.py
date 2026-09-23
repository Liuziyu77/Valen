"""Public model identity and compatibility with existing decision-head weights."""
from types import SimpleNamespace

import pytest
import torch

from valen import MODEL_NAME
from valen.evaluation.inference import answer, predict
from valen.modeling.model import Valen
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
def test_response_uses_valen_without_changing_decisions_or_usage(kind):
    model = Valen(Backbone(), projection_dim=4)
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


def test_checkpoint_parameter_keys_remain_compatible(tmp_path):
    model = Valen(Backbone(), projection_dim=4)
    frozen = model.backbone.embedding.weight.detach().clone()
    # The pre-rename checkpoint stores named tensors, not the model class.
    weights = {"head.decision.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8),
               "head.candidate.weight": torch.full((4, 8), 0.25)}
    torch.save({"weights": weights, "progress": {"epoch": 1, "step": 147},
                "base_manifest": None}, tmp_path / "checkpoint.pt")
    payload = load_checkpoint(tmp_path, model)
    assert type(model).__name__ == "Valen"
    assert payload["progress"] == {"epoch": 1, "step": 147}
    for key, value in weights.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(model.backbone.embedding.weight, frozen, rtol=0, atol=0)
