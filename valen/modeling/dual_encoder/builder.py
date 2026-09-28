import torch
from .model import ValenDualEncoder


def build_dual_encoder(config):
    from ..manifest import read_model_manifest
    from .encoders.modernbert import ModernBertAdapter
    from .encoders.dinov3 import DINOv3Adapter
    encoders = [ModernBertAdapter.load(config["text_model_path"]),
                DINOv3Adapter.load(config["vision_model_path"])]
    model = ValenDualEncoder(*encoders, config)
    model.text_encoder.requires_grad_(False)
    model.vision_encoder.requires_grad_(False)
    stage = config.get("stage", "warmup")
    if stage not in {"warmup", "text", "joint", "vision_top"}:
        raise ValueError(f"Unknown dual-encoder stage: {stage}")
    if stage != "warmup":
        ModernBertAdapter.unfreeze(model.text_encoder, config.get("text_unfreeze_layers", 6))
    if stage == "vision_top":
        DINOv3Adapter.unfreeze(model.vision_encoder, config.get("vision_unfreeze_layers", 2))
    for encoder in encoders:
        if config.get("gradient_checkpointing", True) and any(p.requires_grad for p in encoder.parameters()):
            encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.base_manifest = read_model_manifest(config)
    if config.get("pretrained_shared_path"):
        from .transfer import load_shared
        load_shared(model, config["pretrained_shared_path"])
    return model.to(config.get("device", "cuda"))


def dual_optimizer_groups(model, config):
    from .transfer import is_shared
    groups = {"decision": [], "text": [], "vision": []}
    # 可单独降低共享融合模块的学习率。 / Optionally use a lower LR for shared fusion modules.
    if "fusion_lr" in config:
        groups["fusion"] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            key = "text" if name.startswith("text_encoder.") else "vision" if name.startswith("vision_encoder.") else "decision"
            if key == "decision" and "fusion" in groups and is_shared(name):
                key = "fusion"
            groups[key].append(parameter)
    rates = {"decision": 2e-4, "text": 1e-5, "vision": 2e-6, "fusion": 2e-5}
    return [{"name": k, "params": v, "lr": config.get(f"{k}_lr", rates[k])} for k, v in groups.items() if v]
