"""决策头的数值与接口约束。 / Numerical and interface contracts for new heads."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from valen.modeling.qwen.heads import build_head, head_signature
from valen.modeling.qwen.model import ValenQwen
from valen.modeling.qwen.backend import QwenBackend
from valen.training.rlcd import RLCDObjective


SETTINGS = {"head_hidden_dim": 12, "head_bottleneck_dim": 6, "head_width": 8,
            "head_layers": 2, "head_token_hidden_dim": 5, "head_channel_hidden_dim": 16,
            "projection_dim": 4}


def branch():
    return SimpleNamespace(inputs={"input_ids": torch.arange(12)[None]}, candidate_positions=[7, 9],
                           decision_position=11, context_span=(0, 3), instruction_span=(4, 6),
                           candidate_spans=[(6, 8), (8, 10)])


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
        self.embedding = torch.nn.Embedding(16, 8)
        self.requires_grad_(False)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


@pytest.mark.parametrize("kind", ["bilinear", "mlp", "role_mlp", "mixer"])
def test_candidate_scoring_is_independent_and_permutation_equivariant(kind):
    head = build_head(8, dict(SETTINGS, head_type=kind))
    hidden = torch.randn(1, 12, 8, requires_grad=True)
    b = branch()
    logits = head.score_features(head.extract_features(hidden, b))
    permuted = deepcopy(b)
    permuted.candidate_positions.reverse()
    permuted.candidate_spans.reverse()
    actual = head.score_features(head.extract_features(hidden, permuted))
    torch.testing.assert_close(actual, logits.flip(0))
    single = deepcopy(b)
    single.candidate_positions = b.candidate_positions[:1]
    single.candidate_spans = b.candidate_spans[:1]
    torch.testing.assert_close(head.score_features(head.extract_features(hidden, single)), logits[:1])
    torch.nn.functional.cross_entropy(logits[None], torch.tensor([1])).backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


@pytest.mark.parametrize("kind", ["mlp", "role_mlp", "mixer"])
def test_cached_rlcd_matches_uncached_logits_gradients_and_reference(kind):
    model = ValenQwen(Backbone(), head_config=dict(SETTINGS, head_type=kind))
    copied = deepcopy(model)
    q = SimpleNamespace(qid="q", kind="choice", target=[0., 1.], branches=[branch()])
    pack = [SimpleNamespace(questions=[q])]
    regular = RLCDObjective(model, {})
    cached = RLCDObjective(copied, {}, cache_frozen_features=True)
    torch.manual_seed(4)
    first = regular.prepare(model, pack)[0][0]
    torch.manual_seed(4)
    second = cached.prepare(copied, pack)[0][0]
    torch.testing.assert_close(first.reference_log_probs, second.reference_log_probs, rtol=0, atol=0)
    a, _ = regular.loss(model, q, first)
    b, _ = cached.loss(copied, q, second)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.backward(); b.backward()
    for pa, pb in zip(model.head.parameters(), copied.head.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)


def test_role_pooling_reads_only_declared_spans():
    head = build_head(8, dict(SETTINGS, head_type="mixer"))
    hidden = torch.arange(96).reshape(1, 12, 8).float()
    result = head.extract_features(hidden, branch())
    assert result.shape == (2, 4, 8)
    torch.testing.assert_close(result[0, 0], hidden[0, :3].mean(0))
    torch.testing.assert_close(result[0, 1], hidden[0, 4:6].mean(0))
    torch.testing.assert_close(result[1, 2], hidden[0, 8:10].mean(0))
    changed = hidden.clone(); changed[0, 3] += 10000; changed[0, 10] -= 10000
    torch.testing.assert_close(result, head.extract_features(changed, branch()), rtol=0, atol=0)


def test_head_identity_is_checked_when_initializing():
    old = {"architecture": "qwen", "model_path": "base", "head_type": "mixer", "head_layers": 2}
    with pytest.raises(ValueError, match="head mismatch"):
        QwenBackend().validate_initialization(old, dict(old, head_layers=3))
    with pytest.raises(ValueError, match="Unknown"):
        head_signature({"head_type": "mixre"})
    with pytest.raises(ValueError, match="positive"):
        head_signature({"head_type": "mixer", "head_layers": 0})
