from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from visualjev.modeling.model import VisualJev
from visualjev.training.rlcd import RLCDObjective


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
        self.embedding = torch.nn.Embedding(20, 8)
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


@pytest.mark.parametrize("branches", [1, 3])
def test_cached_rollout_matches_full_forward_losses_gradients_and_updates(branches):
    torch.manual_seed(8)
    a = VisualJev(Backbone(), projection_dim=4)
    a.backbone.requires_grad_(False)
    b = deepcopy(a)
    q = SimpleNamespace(qid="q", target=[1.] + [0.] * (2 * branches - 1),
                        branches=[SimpleNamespace(inputs={"input_ids": torch.tensor([[i, 3, 5, 7]])},
                                                  candidate_positions=[0, 2], decision_position=3)
                                  for i in range(branches)])
    pack = [SimpleNamespace(questions=[q])]
    regular = RLCDObjective(a, {})
    cached = RLCDObjective(b, {}, cache_frozen_features=True)
    torch.manual_seed(13)
    ra = regular.prepare(a, pack)[0][0]
    torch.manual_seed(13)
    rb = cached.prepare(b, pack)[0][0]
    assert torch.equal(ra.actions, rb.actions)
    assert all(not feature.requires_grad for feature in rb.features)
    for name in ('old_log_probs', 'advantages', 'rewards', 'reference_log_probs'):
        torch.testing.assert_close(getattr(ra, name), getattr(rb, name), rtol=0, atol=0)
    oa = torch.optim.AdamW(a.head.parameters(), lr=.02)
    ob = torch.optim.AdamW(b.head.parameters(), lr=.02)
    for _ in range(2):
        oa.zero_grad(); ob.zero_grad()
        la, _ = regular.loss(a, q, ra)
        lb, _ = cached.loss(b, q, rb)
        torch.testing.assert_close(la, lb, rtol=0, atol=0)
        la.backward(); lb.backward()
        for pa, pb in zip(a.head.parameters(), b.head.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)
        oa.step(); ob.step()
    assert b.backbone.calls == 2 * branches  # one-time exact-forward verification
    assert a.backbone.calls == 3 * branches
    for name, value in regular.state_dict()['reference_weights'].items():
        torch.testing.assert_close(value, cached.state_dict()['reference_weights'][name], rtol=0, atol=0)
    restored = RLCDObjective(b, {}, state=cached.state_dict(), resuming=True, cache_frozen_features=True)
    assert all(not p.requires_grad for p in restored.reference.parameters())


def test_cache_rejects_trainable_backbone():
    with pytest.raises(ValueError, match="fully frozen"):
        RLCDObjective(VisualJev(Backbone(), 4), {}, cache_frozen_features=True)
