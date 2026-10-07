"""Attention selection, failure diagnostics and real Qwen SDPA gradients."""
from types import SimpleNamespace

import pytest
import torch

from valen.modeling.qwen.attention import attention_implementation, with_attention_implementation


@pytest.mark.parametrize("kind", [None, "", "flash_attention", {}, False])
def test_invalid_attention_fails_before_loading_weights(kind):
    from valen.modeling.factory import build_model
    with pytest.raises(ValueError, match="attn_implementation"):
        build_model({"model_path": "missing", "attn_implementation": kind})


def test_legacy_default_and_checkpoint_override_do_not_mutate_config():
    config = {"architecture": "qwen", "stage": "joint"}
    assert attention_implementation(config) == "eager"
    changed = with_attention_implementation(config, "sdpa")
    assert changed["attn_implementation"] == "sdpa" and "attn_implementation" not in config
    with pytest.raises(ValueError, match="only"):
        with_attention_implementation({"architecture": "dual_encoder"}, "sdpa")


def test_flash_rejects_fp32_and_cpu_without_loading_weights():
    with pytest.raises(ValueError, match="bf16"):
        attention_implementation({"attn_implementation": "flash_attention_2", "dtype": "fp32"})
    with pytest.raises(ValueError, match="CUDA"):
        attention_implementation({"attn_implementation": "flash_attention_2", "device": "cpu"})


def test_flash_checks_gpu_and_binary_compatibility(monkeypatch):
    from valen.modeling.qwen import attention
    config = {"attn_implementation": "flash_attention_2", "device": "cuda:3"}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        attention_implementation(config)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    with pytest.raises(ValueError, match="Ampere"):
        attention_implementation(config)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    def unavailable(name):
        raise OSError("undefined symbol in CUDA extension")
    monkeypatch.setattr(attention.importlib, "import_module", unavailable)
    with pytest.raises(ImportError, match="matching PyTorch/CUDA"):
        attention_implementation(config)
    monkeypatch.setattr(attention.importlib, "import_module", lambda name: SimpleNamespace(__version__="2.2.0"))
    with pytest.raises(ImportError):
        attention_implementation(config)
    monkeypatch.setattr(attention.importlib, "import_module", lambda name: SimpleNamespace(__version__="2.7.4.post1"))
    assert attention_implementation(config) == "flash_attention_2"


@pytest.fixture(scope="module")
def tiny_qwen(tmp_path_factory):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    config = Qwen3_5Config(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=16, layer_types=["linear_attention", "full_attention"],
                         linear_num_key_heads=2, linear_num_value_heads=2,
                         linear_key_head_dim=8, linear_value_head_dim=8,
                         rope_parameters=dict(rope_type="default", rope_theta=10000,
                                              partial_rotary_factor=1.0, mrope_section=[2, 3, 3])),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=32, patch_size=2, spatial_merge_size=2,
                           temporal_patch_size=1, num_position_embeddings=16),
        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
    config._attn_implementation = "eager"
    path = tmp_path_factory.mktemp("attention-base")
    torch.manual_seed(3)
    Qwen3_5ForConditionalGeneration(config).save_pretrained(path)
    return path


@pytest.mark.parametrize("head", ["bilinear", "mlp", "role_mlp", "mixer"])
def test_real_qwen_sdpa_matches_eager_image_logits_and_trainable_gradients(tiny_qwen, head):
    from valen.data.compilers.qwen import Branch, Question
    from valen.modeling.factory import build_model
    from valen.training.sft import question_loss
    torch.manual_seed(5)
    ids = torch.tensor([[1, 62, 60, 60, 60, 60, 63, 2, 3, 4, 5, 6]])
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
              "mm_token_type_ids": (ids == 60).long(),
              "pixel_values": torch.randn(16, 12), "image_grid_thw": torch.tensor([[1, 4, 4]])}
    branch = Branch(inputs, [9, 10], 11, (0, 8), (8, 9), [(9, 10), (10, 11)])
    question = Question("q", "choice", ["a", "b"], ["a", "b"], [branch], [1., 0.], False)
    results = {}
    for backend in ("eager", "sdpa"):
        torch.manual_seed(7)
        model = build_model(dict(model_path=str(tiny_qwen), device="cpu", dtype="fp32",
                                 stage="joint", head_type=head, attn_implementation=backend,
                                 lora_rank=2, lora_alpha=4, gradient_checkpointing=True))
        assert model.attention_backends == {"text": backend, "vision": backend}
        model.train()
        logits = model(question)
        loss, _ = question_loss(logits, question.target, question.kind)
        loss.backward()
        gradients = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                     if p.requires_grad and p.grad is not None}
        for prefix in ("head.", "backbone.visual.merger."):
            assert any(g.abs().sum()>0 for n,g in gradients.items() if n.startswith(prefix))
        assert any(g.abs().sum()>0 for n,g in gradients.items() if "lora_B" in n)
        assert all(torch.isfinite(g).all() for g in gradients.values())
        model.eval()
        with torch.no_grad():
            inference_logits = model(question)
        torch.testing.assert_close(inference_logits, logits, rtol=1e-4, atol=1e-5)
        results[backend] = (logits.detach(), gradients, set(model.state_dict()))
    eager, sdpa = results["eager"], results["sdpa"]
    assert eager[2] == sdpa[2] and eager[1].keys() == sdpa[1].keys()
    torch.testing.assert_close(eager[0], sdpa[0], rtol=1e-4, atol=1e-5)
    for name in eager[1]:
        torch.testing.assert_close(eager[1][name], sdpa[1][name], rtol=2e-3, atol=2e-5)


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_full_shared_training_updates_language_vision_and_restores_checkpoint(tiny_qwen, tmp_path, dtype):
    import random
    from valen.data.compilers.qwen import Branch, Question
    from valen.modeling.factory import build_model, get_backend, optimizer_groups
    from valen.training.checkpoint import load_checkpoint, save_checkpoint
    from valen.training.sft import question_loss

    config = dict(model_path=str(tiny_qwen), device="cpu", dtype=dtype, stage="joint",
                  finetuning_type="full", qwen_execution="shared_state", head_type="mixer",
                  head_width=8, head_channel_hidden_dim=16, attn_implementation="sdpa",
                  gradient_checkpointing=True)
    model = build_model(config).train()
    assert all(p.requires_grad for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert model.backbone_autocast_dtype == (torch.bfloat16 if dtype == "bf16" else None)
    assert not any("lora_" in name for name, _ in model.named_parameters())
    groups = optimizer_groups(model, config)
    assert {g["name"] for g in groups} == {"head", "language", "merger", "vision"}
    grouped = [id(p) for group in groups for p in group["params"]]
    assert len(grouped) == len(set(grouped)) == len(list(model.parameters()))
    ids = torch.tensor([[1, 62, 60, 60, 60, 60, 63, 2, 3, 4, 5, 6]])
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
              "mm_token_type_ids": (ids == 60).long(),
              "pixel_values": torch.randn(16, 12), "image_grid_thw": torch.tensor([[1, 4, 4]])}
    branch = Branch(inputs, [9, 10], 11, (0, 8), (8, 9), [(9, 10), (10, 11)])
    question = Question("q", "choice", ["a", "b"], ["a", "b"], [branch], [1., 0.], False)
    state = SimpleNamespace(inputs=inputs, questions=[question, question])
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(groups)
    logits = model.forward_state(state)
    torch.testing.assert_close(logits[0], logits[1])
    sum(question_loss(z, question.target, question.kind)[0] for z in logits).backward()
    for prefix in ("head.", "backbone.language_model.", "backbone.visual.patch_embed.",
                   "backbone.visual.blocks.", "backbone.visual.merger."):
        assert any(n.startswith(prefix) and p.grad is not None and p.grad.abs().sum() > 0
                   for n, p in model.named_parameters()), prefix
    optimizer.step()
    for prefix in ("backbone.language_model.", "backbone.visual.patch_embed.", "backbone.visual.blocks."):
        assert any(n.startswith(prefix) and not torch.equal(p, before[n])
                   for n, p in model.named_parameters()), prefix
    model.eval()
    with torch.no_grad():
        expected = model.forward_state(state)
    save_checkpoint(tmp_path, model, optimizer, config, {"step": 1}, random.Random(42))
    restored = build_model(config).eval()
    restored_optimizer = torch.optim.AdamW(optimizer_groups(restored, config))
    payload = load_checkpoint(tmp_path, restored, restored_optimizer)
    assert set(payload["weights"]) == set(dict(model.named_parameters()))
    assert len(restored_optimizer.state) == len(optimizer.state)
    with torch.no_grad():
        for actual, target in zip(restored.forward_state(state), expected):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
    adaptation = get_backend("qwen").adaptation(model, config)
    assert adaptation["text"] == adaptation["vision"] == "full"
    assert adaptation["lora_targets"] == []


def test_full_joint_initializes_from_head_warmup_without_lora(tiny_qwen, tmp_path):
    import random
    from valen.modeling.factory import build_model, optimizer_groups
    from valen.training.checkpoint import load_checkpoint, save_checkpoint
    config = dict(model_path=str(tiny_qwen), device="cpu", dtype="fp32", stage="warmup",
                  finetuning_type="full", head_type="mixer", attn_implementation="sdpa")
    warmup = build_model(config)
    assert all(p.requires_grad == n.startswith("head.") for n, p in warmup.named_parameters())
    save_checkpoint(tmp_path, warmup, torch.optim.AdamW(optimizer_groups(warmup, config)),
                    config, {"step": 1}, random.Random(42))
    joint = build_model(dict(config, stage="joint"))
    load_checkpoint(tmp_path, joint, strict=False)
    for name, tensor in warmup.head.state_dict().items():
        torch.testing.assert_close(joint.head.state_dict()[name], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("options", [{"finetuning_type": "invalid"},
                                     {"finetuning_type": "full", "stage": "vision_top"}])
def test_invalid_full_training_config_fails_before_loading(options):
    from valen.modeling.factory import build_model
    with pytest.raises(ValueError):
        build_model(dict(model_path="missing", **options))


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_real_qwen_parallel_text_image_video_matches_serial_gradients(tiny_qwen, attention):
    from copy import deepcopy
    from valen.data.compilers.qwen import Branch, Question
    from valen.modeling.factory import build_model
    from valen.training.sft import question_loss
    torch.manual_seed(11)
    serial = build_model(dict(model_path=str(tiny_qwen), device="cpu", dtype="fp32", stage="joint",
                              head_type="mixer", head_width=8, head_channel_hidden_dim=16,
                              attn_implementation=attention, lora_rank=2, lora_alpha=4,
                              gradient_checkpointing=True))
    batched = deepcopy(serial)
    questions = []
    for kind, length in [("image", 12), ("video", 14), ("text", 9), ("image", 13)]:
        token = 60 if kind == "image" else 61
        ids = ([1, 62] + [token] * 4 + [63] + list(range(2, length - 5))
               if kind != "text" else list(range(1, length + 1)))
        ids = torch.tensor([ids])
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                  "mm_token_type_ids": (ids == token).long() * (1 if kind == "image" else 2)}
        if kind != "text":
            key = "pixel_values" if kind == "image" else "pixel_values_videos"
            grid = "image_grid_thw" if kind == "image" else "video_grid_thw"
            inputs.update({key: torch.randn(16, 12), grid: torch.tensor([[1, 4, 4]])})
        n = ids.shape[1]
        branch = Branch(inputs, [n - 3, n - 2], n - 1, (0, n - 5),
                        (n - 5, n - 3), [(n - 3, n - 2), (n - 2, n - 1)])
        questions.append(Question(kind, "choice", ["a", "b"], ["a", "b"],
                                  [branch], [1., 0.], kind == "text"))
    expected = [serial(q) for q in questions]
    actual = batched.forward_batch(questions, max_tokens=64)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(right, left, rtol=1e-4, atol=1e-5)
    sum(question_loss(logits, q.target, q.kind)[0] for q, logits in zip(questions, expected)).backward()
    sum(question_loss(logits, q.target, q.kind)[0] for q, logits in zip(questions, actual)).backward()
    for (name, left), (_, right) in zip(serial.named_parameters(), batched.named_parameters()):
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, rtol=3e-3, atol=3e-5, msg=name)


@pytest.mark.parametrize("full", [False, True])
def test_standalone_hf_preserves_backbone_precision_and_both_paths(tiny_qwen, tmp_path, monkeypatch, full):
    """Portable releases retain native math. / 独立模型保持原始精度和两种前向路径。"""
    import importlib
    from pathlib import Path
    from transformers import AutoModel
    import transformers.dynamic_module_utils as dynamic
    from valen.data.compilers.qwen import Branch, Question
    from valen.modeling.factory import build_model
    from valen.modeling.qwen.hf_export import bundle_runtime

    monkeypatch.setattr(dynamic, "HF_MODULES_CACHE", str(tmp_path / "module_cache"))
    package = tmp_path / "portable_runtime"
    bundle_runtime(Path(__file__).resolve().parents[2], package)
    monkeypatch.syspath_prepend(str(tmp_path))
    configuration = importlib.import_module("portable_runtime.configuration_valen")
    modeling = importlib.import_module("portable_runtime.modeling_valen")
    configuration.ValenQwenConfig.register_for_auto_class()
    modeling.ValenQwenForDecisionMaking.register_for_auto_class("AutoModel")
    config = dict(model_path=str(tiny_qwen), device="cpu", dtype="bf16", stage="joint" if full else "warmup",
                  finetuning_type="full" if full else "lora", head_type="mixer", head_width=8,
                  head_channel_hidden_dim=16, attn_implementation="sdpa", gradient_checkpointing=False)
    native = build_model(config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    branch = Branch(inputs, [3, 4], 5, (0, 2), (2, 3), [(3, 4), (4, 5)])
    question = Question("q", "choice", ["a", "b"], ["a", "b"], [branch], None, False)
    state = SimpleNamespace(inputs=inputs, questions=[question, question])
    with torch.inference_mode():
        expected_question = native(question)
        expected_state = native.forward_state(state)
    settings = native.backbone.config.to_dict()
    settings.pop("model_type", None)
    dtype = "float32" if full else "bfloat16"
    settings["dtype"] = dtype
    for field in ("text_config", "vision_config"):
        settings[field]["dtype"] = dtype
    settings.update(valen_head={key: value for key, value in config.items() if key.startswith("head_")},
                    valen_backbone_autocast_dtype="bfloat16" if full else None)
    exported = modeling.ValenQwenForDecisionMaking(configuration.ValenQwenConfig(**settings),
                                                   backbone=native.backbone, head=native.head).eval()
    destination = tmp_path / "release"
    exported.save_pretrained(destination)
    reloaded, info = AutoModel.from_pretrained(destination, trust_remote_code=True, dtype="auto",
                                              attn_implementation="sdpa", output_loading_info=True)
    assert not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    assert {p.dtype for p in reloaded.head.parameters()} == {torch.float32}
    assert {p.dtype for p in reloaded.backbone.parameters()} == {torch.float32 if full else torch.bfloat16}
    assert reloaded.backbone_autocast_dtype == (torch.bfloat16 if full else None)
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(question), expected_question, rtol=0, atol=0)
        for actual, expected in zip(reloaded.forward_state(state), expected_state):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
