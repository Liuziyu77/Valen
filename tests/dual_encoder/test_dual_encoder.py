"""Isolation, shared gradients and real HF encoder/checkpoint integration on CPU."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from PIL import Image
import torch

from valen.data.parallel_compiler import build_parallel_compiler, parallel_candidates
from valen.evaluation.inference import predict
from valen.modeling.model import build_model
from valen.training.runner import run
from valen.training.checkpoint import load_checkpoint
from valen.training.rlcd import RLCDObjective
from valen.training.sft import SFTObjective


def tiny_encoders(root):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import (PreTrainedTokenizerFast, ModernBertConfig, ModernBertModel,
                              DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessor)
    root = Path(root)
    text, vision = root / "text", root / "vision"
    words = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "user", "red", "blue", "green", "true", "false",
             "color", "image", "low", "high", "medium", "Choose", "Is", "it", "Rate", "alpha", "beta"]
    tokenizer = Tokenizer(models.WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", unk_token="[UNK]").save_pretrained(text)
    tc = ModernBertConfig(vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, max_position_embeddings=256, local_attention=16,
                         global_attn_every_n_layers=2, pad_token_id=0, cls_token_id=1, sep_token_id=2)
    ModernBertModel(tc).save_pretrained(text)
    vc = DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                        image_size=32, patch_size=16, num_register_tokens=4,
                        pos_embed_shift=None, pos_embed_jitter=None, pos_embed_rescale=None)
    DINOv3ViTModel(vc).save_pretrained(vision)
    DINOv3ViTImageProcessor().save_pretrained(vision)
    return {"architecture": "dual_encoder", "text_model_path": str(text), "vision_model_path": str(vision),
            "model_name": "Valen-DE-tiny-test", "hidden_size": 32, "num_heads": 4, "intermediate_size": 64,
            "fusion_layers": 1, "question_layers": 1, "candidate_layers": 1, "question_queries": 2,
            "image_size": 32, "state_max_length": 128, "question_max_length": 128,
            "candidate_max_length": 128, "text_batch_size": 8, "candidate_chunk_size": 2,
            "stage": "warmup", "device": "cpu", "dtype": "fp32", "seed": 7,
            "gradient_checkpointing": False, "text_unfreeze_layers": 1, "vision_unfreeze_layers": 1}


def record(image=None):
    state = "red blue"
    assets = []
    if image:
        state = {"messages": [{"role": "user", "content": [{"type": "text", "text": "red blue"},
                    {"type": "image_url", "image_url": {"url": str(image)}}]}]}
    return {"group_id": "tiny", "request": {"state": state, "questions": {
        "choice": {"type": "choice", "instructions": "Choose color", "criteria": {"red": "red", "blue": "blue"}},
        "noul": {"type": "noul", "instructions": "Is it red", "criteria": {"true": "red", "false": "blue"}},
        "score": {"type": "score", "instructions": "Rate color", "criteria": ["low", "medium", "high"]}}},
        "targets": {"choice": {"probabilities": {"red": 1., "blue": 0.}},
                    "noul": {"probabilities": {"true": 1., "false": 0.}},
                    "score": {"probabilities": {"0": 0., "1": .2, "2": .8}}}, "assets": assets}


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    config = tiny_encoders(tmp_path / "encoders")
    compiler = build_parallel_compiler(config, tmp_path)
    model = build_model(config).eval()
    return config, compiler, model


def test_256_questions_share_state_and_remain_independent(setup):
    _, compiler, model = setup
    row = record()
    single = deepcopy(row)
    single["request"]["questions"] = {"original": row["request"]["questions"]["noul"]}
    single["targets"] = {}
    many = deepcopy(single)
    many["request"]["questions"] = {f"q{i}": deepcopy(single["request"]["questions"]["original"]) for i in range(256)}
    many["request"]["questions"]["q200"]["instructions"] = "green blue"
    calls = []
    handle = model.fusion[0].register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        expected = model(compiler.compile(single))[0]
        result = model(compiler.compile(many))
    handle.remove()
    assert len(calls) == 2
    assert len(result) == 256
    for i in (0, 10, 255):
        torch.testing.assert_close(result[i], expected, atol=2e-6, rtol=1e-5)


def test_score_independence_choice_permutation_and_chunk_equivalence(setup):
    _, compiler, model = setup
    row = record()
    with torch.no_grad():
        original = model(compiler.compile(row))
        changed = deepcopy(row)
        changed["request"]["questions"]["score"]["criteria"][2] = "blue green alpha"
        changed["request"]["questions"]["choice"]["criteria"] = {"blue": "blue", "red": "red"}
        modified = model(compiler.compile(changed))
        torch.testing.assert_close(original[2][:2], modified[2][:2], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(original[0], modified[0].flip(0), atol=1e-6, rtol=1e-5)
        model.candidate_chunk_size, model.text_batch_size = 100, 100
        for a, b in zip(original, model(compiler.compile(row))):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_image_tokens_mask_and_shared_backward(setup, tmp_path):
    config, compiler, model = setup
    path = tmp_path / "image.png"
    Image.new("RGB", (32, 8), "red").save(path)
    state = compiler.compile(record(path))
    assert state.visual_mask.shape == (1, 5)
    calls = []
    handle = model.vision_encoder.register_forward_hook(lambda *args: calls.append(1))
    objective = SFTObjective({"brier_weight": .1, "rps_weight": .2})
    losses = objective.state_losses(model, state, [None] * 3)
    torch.stack([loss for loss, _ in losses]).mean().backward()
    handle.remove()
    assert calls == [1]
    for name in ("text_projection", "vision_projection", "fusion", "question_reader", "candidate_reader", "head"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()), name
    assert all(p.grad is None for p in model.text_encoder.parameters())
    assert all(p.grad is None for p in model.vision_encoder.parameters())


def test_preprocessing_and_contract_reject_invalid_inputs(setup, tmp_path):
    _, compiler, _ = setup
    path = tmp_path / "image.png"
    Image.new("RGB", (256, 8), "red").save(path)
    row = record(path)
    assert compiler.compile(row).visual_mask.sum() < 5
    row["request"]["state"]["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": str(path)}})
    with pytest.raises(ValueError, match="one image"):
        compiler.compile(row)
    row = record()
    compiler.limits = (128, 3, 128)
    with pytest.raises(ValueError, match="no truncation"):
        compiler.compile(row)
    assert parallel_candidates({"type": "noul", "instructions": {"rule": "red"},
                                "criteria": {"true": "red", "false": "blue"}}) == [("true", "red"), ("false", "blue")]


@pytest.mark.parametrize("stage", ["text", "vision_top"])
def test_unfreezing_and_checkpointed_gradients(setup, tmp_path, stage):
    config, compiler, _ = setup
    model = build_model(dict(config, stage=stage, gradient_checkpointing=True)).train()
    path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), "red").save(path)
    sum(z.square().sum() for z in model(compiler.compile(record(path)))).backward()
    assert any(p.grad is not None for p in model.text_encoder.layers[-1].parameters())
    assert all(p.grad is None for p in model.text_encoder.layers[0].parameters())
    if stage == "vision_top":
        assert any(p.grad is not None for p in model.vision_encoder.model.layer[-1].parameters())


def test_rlcd_rollouts_share_state_and_reference_stays_frozen(setup):
    _, compiler, model = setup
    state = compiler.compile(record())
    objective = RLCDObjective(model, {"group_size": 8, "beta": .1})
    reference = {k: v.clone() for k, v in objective.reference.state_dict().items()}
    calls = []
    handle = model.fusion[0].register_forward_hook(lambda *args: calls.append(1))
    rolls = objective.prepare(model, [state])[0]
    assert len(calls) == 1
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        losses = objective.state_losses(model, state, rolls)
        loss = torch.stack([value for value, _ in losses]).mean()
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert len(calls) == 3
    handle.remove()
    for k, v in reference.items():
        torch.testing.assert_close(v, objective.reference.state_dict()[k], rtol=0, atol=0)


def test_sft_rlcd_resume_and_inference_with_real_encoder_files(setup, tmp_path, monkeypatch):
    config, compiler, _ = setup
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps(record()) + "\n")
    config = dict(config, data=str(data), output=str(tmp_path / "sft"), epochs=3, max_steps=1,
                  tokens_per_step=1024, save_every=1, brier_weight=.1, rps_weight=.2)
    model, _, progress = run(config)
    assert progress["step"] == 1
    model, _, progress = run(dict(config, max_steps=2), resume=tmp_path / "sft/latest")
    assert progress["step"] == 2
    continuous, _, _ = run(dict(config, max_steps=2, output=str(tmp_path / "sft-continuous")))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, continuous.state_dict()[key], rtol=0, atol=0)
    restored = build_model(config)
    load_checkpoint(tmp_path / "sft/latest", restored)
    state = compiler.compile(record())
    assert predict(model, state) == predict(restored, state)
    rl = dict(config, output=str(tmp_path / "rl"), method="rlcd", rps_weight=0., brier_weight=0.,
              rlcd={"group_size": 4, "num_iterations": 2, "beta": .1})
    _, _, progress = run(rl, initialize=tmp_path / "sft/latest")
    assert progress["optimizer_steps"] == 2
    resumed, _, progress = run(dict(rl, max_steps=2), resume=tmp_path / "rl/latest")
    assert progress["optimizer_steps"] == 4
    continuous, _, _ = run(dict(rl, max_steps=2, output=str(tmp_path / "rl-continuous")), initialize=tmp_path / "sft/latest")
    for key, value in resumed.state_dict().items():
        torch.testing.assert_close(value, continuous.state_dict()[key], rtol=0, atol=0)
    payload = torch.load(tmp_path / "rl/latest/checkpoint.pt", weights_only=False)
    assert payload["base_manifest"]["architecture"] == "dual_encoder"
    assert payload["training_state"]["reference_weights"]
    with pytest.raises(ValueError, match="architecture mismatch"):
        run(dict(config, num_heads=8, output=str(tmp_path / "wrong-heads")), initialize=tmp_path / "sft/latest")


@pytest.mark.parametrize("resolution", [224, 384])
def test_dino_resolution_and_256_image_questions(setup, tmp_path, resolution):
    _, compiler, model = setup
    compiler.image_size = resolution
    path = tmp_path / "square.png"
    Image.new("RGB", (128, 128), "blue").save(path)
    row = record(path)
    row["targets"] = {}
    question = row["request"]["questions"]["noul"]
    row["request"]["questions"] = {"single": question}
    state = compiler.compile(row)
    with torch.no_grad():
        memory, mask = model.encode_state(state)
        assert memory.shape[1] == len(state.state_tokens) + (resolution // 16) ** 2 + 1
        assert mask.all()
        expected = model(state)[0]
        row["request"]["questions"] = {f"q{i}": deepcopy(question) for i in range(256)}
        row["request"]["questions"]["q42"]["instructions"] = "green blue"
        result = model(compiler.compile(row))
        assert len(result) == 256
        torch.testing.assert_close(result[255], expected, rtol=1e-5, atol=2e-6)


def test_wrong_encoder_identity_rejected(setup, tmp_path, monkeypatch):
    config, _, model = setup
    from valen.training.checkpoint import save_checkpoint
    import random
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    save_checkpoint(tmp_path / "checkpoint", model, optimizer, config, {"step": 1}, random.Random(1))
    model.base_manifest = deepcopy(model.base_manifest)
    model.base_manifest["vision"]["revision"] = "different"
    with pytest.raises(ValueError, match="Base model revision"):
        load_checkpoint(tmp_path / "checkpoint", model)
