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
