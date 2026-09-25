import torch
from torch import nn
from .model import ValenQwen
from ..manifest import read_base_manifest


def build_qwen(config):
    from transformers import Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[config.get("dtype", "bf16")]
    # Load the exact published architecture before extracting its backbone.
    # Loading Qwen3_5Model directly can trigger a language-prefix remapping in
    # Transformers 5.4 and silently initialize the language tower from scratch.
    container, loading_info = Qwen3_5ForConditionalGeneration.from_pretrained(
        config["model_path"], dtype=dtype, attn_implementation="eager",
        local_files_only=True, output_loading_info=True)
    if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete pretrained model load: {loading_info}")
    backbone = container.model
    del container  # Never execute or retain the generation/vocabulary output layer.
    backbone.requires_grad_(False)
    stage = config.get("stage", "joint")
    if stage not in {"warmup", "text", "joint", "vision_top"}:
        raise ValueError(f"Unknown stage: {stage}")
    targets = []
    if stage != "warmup":
        # Enumerate exact language Linear paths. This covers DeltaNet, full attention
        # and FFN, without accidentally adapting the visual tower or decision head.
        targets = [name for name, module in backbone.language_model.named_modules() if isinstance(module, nn.Linear)]
        backbone.language_model = get_peft_model(backbone.language_model, LoraConfig(
            r=config.get("lora_rank", 32), lora_alpha=config.get("lora_alpha", 64),
            lora_dropout=0.0, target_modules=targets, bias="none"))
    if stage in {"joint", "vision_top"}:
        backbone.visual.merger.requires_grad_(True)
    if stage == "vision_top":
        for block in backbone.visual.blocks[-4:]:
            block.requires_grad_(True)
    if config.get("gradient_checkpointing", True) and stage != "warmup":
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = ValenQwen(backbone, config.get("projection_dim", 256))
    model.base_manifest = read_base_manifest(config["model_path"])
    model.to(config.get("device", "cuda"))
    model.lora_targets = targets
    return model


def qwen_optimizer_groups(model, config):
    groups = {"head": [], "lora": [], "merger": [], "vision": []}
    rates = {"head": 2e-4, "lora": 5e-5, "merger": 1e-5, "vision": 2e-6}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("head."):
            group = "head"
        elif "lora_" in name:
            group = "lora"
        elif ".visual.merger." in name:
            group = "merger"
        elif ".visual.blocks." in name:
            group = "vision"
        else:
            raise ValueError(f"Unclassified trainable parameter: {name}")
        groups[group].append(parameter)
    return [{"params": parameters, "lr": config.get(f"{name}_lr", rates[name]), "name": name}
            for name, parameters in groups.items() if parameters]
