"""Shared causal sequence: input alignment, one forward/backward and cache parity."""
from copy import deepcopy
import json
import random
from types import SimpleNamespace

import pytest
import torch

from valen.data.compilers.qwen import Compiler
from valen.evaluation.inference import predict
from valen.modeling.factory import normalize_model_config
from valen.modeling.qwen.backend import QwenBackend
from valen.modeling.qwen.model import ValenQwen
from valen.training.rlcd import RLCDObjective
from valen.training.sft import SFTObjective
from valen.training.batching import training_batches
from valen.training.sft import question_loss


class Tokenizer:
    all_special_tokens = ["<|im_start|>", "<|im_end|>"]

    def encode(self, text, **kwargs):
        return list(text.encode("ascii"))

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class Processor:
    tokenizer = Tokenizer()

    def __init__(self):
        self.calls = 0

    def apply_chat_template(self, messages, **kwargs):
        self.calls += 1
        ids = torch.tensor([self.tokenizer.encode("State evidence\n")])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                "mm_token_type_ids": torch.zeros_like(ids)}


def record():
    return {"group_id": "hidden-group", "request": {"state": "example", "questions": {
        "private-choice-id": {"type": "choice", "instructions": "Select the color.",
                              "criteria": {"red": "red object", "blue": "blue object"}},
        "private-noul-id": {"type": "noul", "instructions": "Is the object red?"},
        "private-score-id": {"type": "score", "instructions": "Count objects.",
                             "criteria": ["zero objects", "one object", "two objects", "three objects", "four objects"]}}},
        "targets": {"private-choice-id": {"probabilities": {"red": 1., "blue": 0.}},
                    "private-noul-id": {"probabilities": {"true": 1., "false": 0.}},
                    "private-score-id": {"probabilities": {str(i): float(i == 2) for i in range(5)}}}}


def compiler(**kwargs):
    # Schema Noul descriptions include Chinese; this tokenizer maps characters directly.
    class UnicodeTokenizer(Tokenizer):
        def encode(self, text, **kwargs):
            return [ord(c) for c in text]
    processor = Processor()
    processor.tokenizer = UnicodeTokenizer()
    return Compiler(processor, execution="shared_state", **kwargs)


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
        self.embedding = torch.nn.Embedding(256, 8)
        self.calls = self.backwards = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        # A causal prefix-dependent oracle; repeated per-question execution must
        # reproduce the same shared-sequence results, unlike the legacy prompt.
        hidden = self.embedding(input_ids % 256).cumsum(1) / 100
        if hidden.requires_grad:
            hidden.register_hook(self._backward)
        return SimpleNamespace(last_hidden_state=hidden)

    def _backward(self, grad):
        self.backwards += 1
        return grad


def model(kind="bilinear"):
    return ValenQwen(Backbone(), head_config={"head_type": kind, "projection_dim": 4,
        "head_hidden_dim": 12, "head_bottleneck_dim": 6, "head_width": 8,
        "head_layers": 2, "head_token_hidden_dim": 5, "head_channel_hidden_dim": 16})


def test_compiler_shared_readouts_and_all_task_labels():
    c = compiler()
    r = record()
    state = c.compile(r, random.Random(1))
    assert c.processor.calls == 1
    assert len(state.questions) == 3
    inputs = state.inputs
    ids = inputs["input_ids"][0].tolist()
    text = ''.join(map(chr, ids))
    assert text.count("State evidence") == 1
    assert "private-" not in text and "hidden-group" not in text
    assert state.compute_tokens == state.logical_tokens == len(ids)
    assert inputs["attention_mask"].shape == inputs["mm_token_type_ids"].shape == inputs["input_ids"].shape
    first_decision = state.questions[0].branches[0].decision_position
    for q in state.questions:
        assert len(q.branches) == 1
        b = q.branches[0]
        assert b.inputs is inputs
        assert max(b.candidate_positions) < first_decision
        assert text[:b.decision_position + 1].endswith("Decision:")
        start, end = b.instruction_span
        assert ''.join(map(chr, ids[start:end])) == r["request"]["questions"][q.qid]["instructions"]
        for description, (a, z), position in zip(q.descriptions, b.candidate_spans, b.candidate_positions):
            assert ''.join(map(chr, ids[a:z])) == description
            assert position == z - 1
        assert dict(zip(q.keys, q.target)) == r["targets"][q.qid]["probabilities"]
    # Changing labels and external identifiers cannot change model inputs.
    changed = deepcopy(r)
    changed["targets"]["private-choice-id"]["probabilities"] = {"red": 0., "blue": 1.}
    changed["request"]["questions"] = {str(i): q for i, q in enumerate(changed["request"]["questions"].values())}
    changed["targets"] = {str(i): t for i, t in enumerate(changed["targets"].values())}
    torch.testing.assert_close(c.compile(changed, random.Random(1)).inputs["input_ids"], inputs["input_ids"])


def test_shared_media_tensors_reused_without_decoding(tmp_path):
    class MediaProcessor(Processor):
        def apply_chat_template(self, messages, **kwargs):
            result = super().apply_chat_template(messages, **kwargs)
            result.update(pixel_values_videos=torch.ones(4, 8), video_grid_thw=torch.tensor([[1, 2, 2]]),
                          video_metadata=[{"frames_indices": [0, 2], "fps": 2}])
            return result
    p = tmp_path / "mock.video"
    p.write_bytes(b"synthetic media, never decoded")
    processor = MediaProcessor()
    processor.tokenizer = compiler().tokenizer
    c = Compiler(processor, execution="shared_state")
    r = record()
    r["request"]["state"] = {"messages": [{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": str(p)}}]}]}
    state = c.compile(r)
    assert processor.calls == 1
    assert "video_metadata" not in state.inputs
    assert state.media[0]["sampling"]["timestamps_seconds"] == [0., 1.]
    assert all(not q.is_text and q.branches[0].inputs is state.inputs for q in state.questions)
    assert state.inputs["pixel_values_videos"].shape == (4, 8)


def test_partial_labels_keep_identical_training_and_inference_context():
    torch.manual_seed(19)
    c, r = compiler(), record()
    r["targets"] = {"private-choice-id": r["targets"]["private-choice-id"]}
    train, infer = c.compile(r, labeled_only=True), c.compile(r)
    assert len(train.questions) == 1 and len(infer.questions) == 3
    for key in train.inputs:
        torch.testing.assert_close(train.inputs[key], infer.inputs[key], rtol=0, atol=0)
    assert train.questions[0].branches[0].decision_position == infer.questions[0].branches[0].decision_position
    m = model("mixer").eval()
    torch.testing.assert_close(m.forward_state(train)[0], m.forward_state(infer)[0], rtol=0, atol=0)
    # Adding a label changes supervision alone, even with shuffled Choice options.
    torch.testing.assert_close(c.compile(r, random.Random(5), True).inputs["input_ids"],
                               c.compile(record(), random.Random(5), True).inputs["input_ids"], rtol=0, atol=0)


@pytest.mark.parametrize('bins', [5, 11])
def test_single_question_is_supported_by_shared_path_including_score(bins):
    r = record()
    r["request"]["questions"] = {"private-score-id": r["request"]["questions"]["private-score-id"]}
    r["targets"] = {"private-score-id": r["targets"]["private-score-id"]}
    r["request"]["questions"]["private-score-id"]["criteria"] = [f'Count {i}' for i in range(bins)]
    r["targets"]["private-score-id"]["probabilities"] = {str(i): float(i == 2) for i in range(bins)}
    state = compiler().compile(r)
    m = model("mixer")
    logits = m.forward_state(state)
    assert len(logits) == 1 and logits[0].shape == (bins,)
    assert m.backbone.calls == 1


def test_execution_defaults_and_spelling_alias_are_canonical():
    assert normalize_model_config({}) == normalize_model_config({"qwen_execution": "question"})
    assert normalize_model_config({"qwen_execution": "share_state"})["qwen_execution"] == "shared_state"


@pytest.mark.parametrize("kind", ["bilinear", "mlp", "role_mlp", "mixer"])
@pytest.mark.parametrize("reduction", ["state_mean", "question_mean"])
def test_shared_state_microbatch_gradients_match_serial_and_never_repeat_media(kind, reduction):
    torch.manual_seed(23)
    serial, c = model(kind), compiler()
    batched = deepcopy(serial)
    short = record()
    short["request"]["questions"].pop("private-score-id")
    short["targets"].pop("private-score-id")
    pack = [c.compile(short), c.compile(record())]
    normalizer = len(pack) if reduction == "state_mean" else sum(len(s.questions) for s in pack)
    config = {"microbatch_size": 2, "microbatch_max_tokens": 2000, "loss_reduction": reduction}
    expected_logits = [serial.forward_state(s) for s in pack]
    for state, outputs in zip(pack, expected_logits):
        weight = 1 / len(state.questions) if reduction == "state_mean" else 1
        loss = sum(question_loss(z, q.target, q.kind, .2, .1)[0] * weight / normalizer
                   for q, z in zip(state.questions, outputs))
        loss.backward()
    units = list(training_batches(batched, QwenBackend(), pack,
                                 [[None] * len(s.questions) for s in pack], config))
    assert len(units) == 1 and len(units[0]) == 5
    loss = sum(question_loss(d.logits, d.question.target, d.question.kind, .2, .1)[0] * weight / normalizer
               for d, weight in units[0])
    loss.backward()
    assert batched.backbone.calls == batched.backbone.backwards == 1
    assert serial.backbone.calls == serial.backbone.backwards == 2
    for p, other in zip(serial.parameters(), batched.parameters()):
        torch.testing.assert_close(other.grad, p.grad, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize("kind", ["bilinear", "mlp", "role_mlp", "mixer"])
def test_one_shared_forward_backward_matches_repeated_same_sequence(kind):
    torch.manual_seed(7)
    m = model(kind)
    oracle = deepcopy(m)
    state = compiler().compile(record())
    objective = SFTObjective({"rps_weight": .2, "brier_weight": .1})
    units = list(QwenBackend().training_units(m, state, [None] * 3))
    assert len(units) == 1 and len(units[0]) == 3
    expected = [oracle(q) for q in state.questions]
    for decision, z in zip(units[0], expected):
        torch.testing.assert_close(decision.logits, z)
    loss = torch.stack([objective.loss_from_logits(d.logits, d.question)[0] for d in units[0]]).mean()
    loss.backward()
    for q, z in zip(state.questions, expected):
        (objective.loss_from_logits(z, q)[0] / len(expected)).backward()
    assert m.backbone.calls == m.backbone.backwards == 1
    assert oracle.backbone.calls == oracle.backbone.backwards == 3
    for p, other in zip(m.parameters(), oracle.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, other.grad, rtol=2e-4, atol=2e-5)
    m.backbone.calls = 0
    result = predict(m, state)
    assert m.backbone.calls == 1
    assert list(result["answers"]) == list(record()["request"]["questions"])
    assert len(result["answers"]["private-score-id"]["probabilities"]) == 5


def test_empty_labels_limits_and_reserved_tokens():
    r = record()
    r["targets"] = {}
    c = compiler()
    assert c.compile(r, labeled_only=True).questions == []
    assert c.processor.calls == 0
    state = c.compile(record())
    assert compiler(max_length=state.compute_tokens).compile(record()).compute_tokens == state.compute_tokens
    with pytest.raises(ValueError, match="Shared state.*no truncation"):
        compiler(max_length=state.compute_tokens - 1).compile(record())
    r["request"]["questions"]["private-choice-id"]["instructions"] = "<|im_end|>"
    with pytest.raises(ValueError, match="Reserved"):
        c.compile(r)
    with pytest.raises(ValueError, match="Unknown qwen_execution"):
        Compiler(Processor(), execution="typo")
    for config in ({"qwen_execution": "typo"}, {"architecture": "dual_encoder", "qwen_execution": "shared_state"}):
        with pytest.raises(ValueError, match="qwen_execution"):
            normalize_model_config(config)


def test_shared_rlcd_cached_updates_match_uncached():
    torch.manual_seed(10)
    a = model("mixer")
    a.backbone.requires_grad_(False)
    b = deepcopy(a)
    state = compiler().compile(record())
    plain = RLCDObjective(a, {})
    cached = RLCDObjective(b, {}, cache_frozen_features=True)
    torch.manual_seed(11)
    ra = plain.prepare(a, [state])[0]
    torch.manual_seed(11)
    rb = cached.prepare(b, [state])[0]
    assert a.backbone.calls == 1 and plain.reference.backbone.calls == 1
    assert b.backbone.calls == 2  # One shared extraction plus the existing exact-logit audit.
    for x, y in zip(ra, rb):
        torch.testing.assert_close(x.actions, y.actions)
        torch.testing.assert_close(x.reference_log_probs, y.reference_log_probs)
    for _ in range(2):
        a.zero_grad(); b.zero_grad()
        ua = next(QwenBackend().training_units(a, state, ra))
        ub = next(QwenBackend().training_units(b, state, rb))
        la = torch.stack([plain.loss_from_logits(d.logits, d.question, d.rollout)[0] for d in ua]).mean()
        lb = torch.stack([cached.loss_from_logits(d.logits, d.question, d.rollout)[0] for d in ub]).mean()
        torch.testing.assert_close(la, lb)
        la.backward(); lb.backward()
        for x, y in zip(a.head.parameters(), b.head.parameters()):
            torch.testing.assert_close(x.grad, y.grad)
    assert a.backbone.calls == 3 and b.backbone.calls == 2


def test_shared_sft_runner_resume_matches_uninterrupted(tmp_path, monkeypatch):
    from transformers import AutoProcessor
    from valen.training import runner
    made = []

    def build(config):
        m = model()
        m.backbone.requires_grad_(False)
        made.append(m)
        return m

    monkeypatch.setattr(runner, "build_model", build)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: compiler().processor)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps(record()) + '\n')
    config = {"model_path": str(tmp_path / "base"), "data": str(data), "output": str(tmp_path / "resumed"),
              "qwen_execution": "shared_state", "stage": "warmup", "device": "cpu", "epochs": 3,
              "max_steps": 1, "save_every": 1, "rps_weight": .2, "brier_weight": .1}
    runner.run(config)
    assert made[-1].backbone.calls == 1
    checkpoint = tmp_path / "resumed/latest"
    assert json.loads((checkpoint / "config.json").read_text())["qwen_execution"] == "shared_state"
    runner.run(dict(config, max_steps=2), resume=checkpoint)
    assert made[-1].backbone.calls == 1
    runner.run(dict(config, max_steps=2, output=str(tmp_path / "full")))
    assert made[-1].backbone.calls == 2
    a = torch.load(checkpoint / "checkpoint.pt", weights_only=False)
    b = torch.load(tmp_path / "full/latest/checkpoint.pt", weights_only=False)
    for key in a["weights"]:
        torch.testing.assert_close(a["weights"][key], b["weights"][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="Resume config/data changed"):
        runner.run(dict(config, max_steps=3, qwen_execution="question"), resume=checkpoint)


def tiny_real_model():
    from transformers import Qwen3_5Config, Qwen3_5Model
    config = Qwen3_5Config(
        text_config={"vocab_size": 65536, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                     "head_dim": 16, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
                     "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                     "layer_types": ["linear_attention", "full_attention"],
                     "rope_parameters": {"rope_type": "default", "rope_theta": 10000.,
                                         "partial_rotary_factor": 1., "mrope_section": [2, 3, 3]}},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                       "out_hidden_size": 32, "patch_size": 2, "temporal_patch_size": 2,
                       "spatial_merge_size": 2, "num_position_embeddings": 16},
        image_token_id=60000, video_token_id=60001, vision_start_token_id=60002, vision_end_token_id=60003,
    )
    config._attn_implementation = "eager"
    return ValenQwen(Qwen3_5Model(config), projection_dim=4)


def synthetic_video(state):
    # Two temporal groups exercise expanded video RoPE, not just an image-like frame.
    state.inputs["input_ids"][0, :6] = torch.tensor([60002, 60001, 60003, 60002, 60001, 60003])
    state.inputs["mm_token_type_ids"][0, [1, 4]] = 2
    state.inputs["video_grid_thw"] = torch.tensor([[2, 2, 2]])
    state.inputs["pixel_values_videos"] = torch.randn(8, 24)


@pytest.mark.parametrize("video", [False, True])
def test_tiny_real_qwen_shared_training(video):
    """Random tiny Qwen with DeltaNet/full attention and synthetic video patches."""
    m = tiny_real_model()
    m.train()
    m.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    state = compiler().compile(record())
    if video:
        synthetic_video(state)
    calls = []
    handle = m.backbone.register_forward_pre_hook(lambda *_: calls.append(1))
    unit = next(QwenBackend().training_units(m, state, [None] * 3))
    objective = SFTObjective({"rps_weight": .2})
    loss = torch.stack([objective.loss_from_logits(d.logits, d.question)[0] for d in unit]).mean()
    assert torch.isfinite(loss)
    loss.backward()
    handle.remove()
    assert len(calls) == 1
    assert m.head.decision.weight.grad is not None
    assert m.backbone.language_model.embed_tokens.weight.grad is not None
    if video:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.backbone.visual.parameters())
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_tiny_real_qwen_state_batch_matches_serial_with_video_and_padding():
    torch.manual_seed(31)
    serial = tiny_real_model()
    batched = deepcopy(serial)
    r = record()
    r["request"]["questions"].pop("private-score-id")
    r["targets"].pop("private-score-id")
    states = [compiler().compile(record()), compiler().compile(r)]
    synthetic_video(states[0])
    expected = [serial.forward_state(s) for s in states]
    actual = batched.forward_state_batch(states, max_tokens=2000)
    for state, zs, others in zip(states, expected, actual):
        for q, z, other in zip(state.questions, zs, others):
            torch.testing.assert_close(other, z, rtol=2e-4, atol=2e-5)
    sum(question_loss(z, q.target, q.kind)[0]
        for s, zs in zip(states, expected) for q, z in zip(s.questions, zs)).backward()
    sum(question_loss(z, q.target, q.kind)[0]
        for s, zs in zip(states, actual) for q, z in zip(s.questions, zs)).backward()
    for a, b in zip(serial.parameters(), batched.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(b.grad, a.grad, rtol=1e-3, atol=3e-5)


@pytest.mark.parametrize('reduction', ['state_mean', 'question_mean'])
def test_batched_shared_rlcd_keeps_rollout_identity_and_matches_serial_updates(reduction):
    torch.manual_seed(73)
    serial = model('mixer')
    batched = deepcopy(serial)
    c = compiler()
    short = record()
    short['request']['questions'].pop('private-score-id')
    short['targets'].pop('private-score-id')
    # Reverse length order so sorting must retain rollout/state identities.
    pack = [c.compile(record()), c.compile(short)]
    cfg = {'method': 'rlcd', 'microbatch_size': 2, 'microbatch_max_tokens': 4096,
           'loss_reduction': reduction, 'qwen_execution': 'shared_state', 'data_prefetch': True}
    from valen.training.runner import normalize_config
    normalize_config(cfg)
    one = RLCDObjective(serial, {})
    many = RLCDObjective(batched, {}, batch_config=cfg)
    torch.manual_seed(74)
    a = one.prepare(serial, pack)
    torch.manual_seed(74)
    b = many.prepare(batched, pack)
    assert serial.backbone.calls == one.reference.backbone.calls == 2
    assert batched.backbone.calls == many.reference.backbone.calls == 1
    for ga, gb in zip(a, b):
        for ra, rb in zip(ga, gb):
            torch.testing.assert_close(ra.actions, rb.actions)
            torch.testing.assert_close(ra.reference_log_probs, rb.reference_log_probs, rtol=1e-5, atol=1e-5)
    normalizer = len(pack) if reduction == 'state_mean' else sum(len(s.questions) for s in pack)
    for net, objective, rollouts, size in [(serial, one, a, 1), (batched, many, b, 2)]:
        for unit in training_batches(net, QwenBackend(), pack, rollouts, dict(cfg, microbatch_size=size)):
            sum(objective.loss_from_logits(d.logits, d.question, d.rollout)[0] * weight / normalizer
                for d, weight in unit).backward()
    for left, right in zip(serial.parameters(), batched.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=4e-4, atol=4e-5)
    assert all(p.grad is None for p in many.reference.parameters())
    assert many.prepare(batched, []) == []
