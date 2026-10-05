from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from valen.data.compilers.qwen import Branch, Question
from valen.modeling.factory import get_backend
from valen.modeling.qwen.batching import collate_branches
from valen.modeling.qwen.model import ValenQwen
from valen.training.batching import training_batches
from valen.training.runner import normalize_config
from valen.training.sft import question_loss


class CausalBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 8)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8, pad_token_id=0))
        self.batch_shapes = []

    def forward(self, input_ids, attention_mask, **kwargs):
        self.batch_shapes.append(tuple(input_ids.shape))
        hidden = self.embedding(input_ids) * attention_mask[..., None]
        return SimpleNamespace(last_hidden_state=hidden.cumsum(1))


def question(length, kind="choice"):
    def branch(offset, count):
        ids = (torch.arange(length) + offset + 1).remainder(63)[None]
        spans = [(length - count - 1 + i, length - count + i) for i in range(count)]
        return Branch({"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                      [end - 1 for _, end in spans], length - 1,
                      (0, 2), (2, 3), spans)
    count = 4 if kind == "score" else 3
    branches = [branch(i, 1) for i in range(count)] if kind == "score" else [branch(0, count)]
    return Question(str(length), kind, [str(i) for i in range(count)], ["x"] * count,
                    branches, [1.] + [0.] * (count - 1), True)


@pytest.mark.parametrize("head", ["bilinear", "mlp", "role_mlp", "mixer"])
def test_batched_logits_and_gradients_match_serial_with_score_branches(head):
    torch.manual_seed(7)
    serial = ValenQwen(CausalBackbone(), head_config={"head_type": head})
    batched = deepcopy(serial)
    questions = [question(7), question(10, "score"), question(8)]
    reference = [serial(q) for q in questions]
    outputs = batched.forward_batch(questions, max_tokens=30)
    for expected, actual in zip(reference, outputs):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    sum(question_loss(logits, q.target, q.kind)[0] for q, logits in zip(questions, reference)).backward()
    sum(question_loss(logits, q.target, q.kind)[0] for q, logits in zip(questions, outputs)).backward()
    for (_, expected), (_, actual) in zip(serial.named_parameters(), batched.named_parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=2e-5)
    assert any(batch > 1 for batch, _ in batched.backbone.batch_shapes)
    assert all(batch * length <= 30 for batch, length in batched.backbone.batch_shapes)


def test_mixed_media_collation_preserves_patch_and_grid_order():
    qs = [question(7), question(9), question(8), question(10)]
    branches = [q.branches[0] for q in qs]
    for index, branch in enumerate(branches):
        branch.inputs["mm_token_type_ids"] = torch.full_like(branch.inputs["input_ids"], index % 3)
    for index in (0, 2):
        branches[index].inputs.update(pixel_values=torch.full((4, 6), float(index)),
                                      image_grid_thw=torch.tensor([[1, 2, 2]]))
    branches[1].inputs.update(pixel_values_videos=torch.full((8, 6), 10.),
                             video_grid_thw=torch.tensor([[2, 2, 2]]))
    inputs = collate_branches(branches)
    assert inputs["input_ids"].shape == (4, 10)
    assert inputs["attention_mask"].sum(1).tolist() == [7, 9, 8, 10]
    assert inputs["mm_token_type_ids"][0, 7:].tolist() == [0, 0, 0]
    assert inputs["pixel_values"][:, 0].tolist() == [0.] * 4 + [2.] * 4
    assert inputs["image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 2]]
    assert inputs["pixel_values_videos"].shape == (8, 6)
    assert inputs["video_grid_thw"].tolist() == [[2, 2, 2]]


def test_parallel_qa_loss_preserves_unequal_state_question_weights():
    torch.manual_seed(8)
    serial = ValenQwen(CausalBackbone(), head_config={"head_type": "mixer"})
    batched = deepcopy(serial)
    pack = [SimpleNamespace(questions=[question(7)]),
            SimpleNamespace(questions=[question(8), question(10, "score"), question(9)])]
    rollouts = [[None] * len(state.questions) for state in pack]
    losses = []
    for state in pack:
        loss = sum(question_loss(serial(q), q.target, q.kind, .2, .1)[0] for q in state.questions)
        losses.append(loss / len(state.questions))
    (sum(losses) / len(pack)).backward()
    batches = 0
    for unit in training_batches(batched, get_backend("qwen"), pack, rollouts,
                                 {"microbatch_size": 4, "microbatch_max_tokens": 100}):
        batches += 1
        sum(question_loss(d.logits, d.question.target, d.question.kind, .2, .1)[0] * weight / len(pack)
            for d, weight in unit).backward()
    assert batches == 1
    for (_, expected), (_, actual) in zip(serial.named_parameters(), batched.named_parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=2e-5)


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_invalid_microbatch_size_is_rejected(value):
    with pytest.raises(ValueError, match="microbatch_size"):
        normalize_config({"microbatch_size": value})


def test_parallel_qa_microbatches_reject_rlcd():
    with pytest.raises(ValueError, match="Qwen SFT"):
        normalize_config({"method": "rlcd", "microbatch_size": 2})
