import torch


def load_encoder(path, config, unused_prefixes=()):
    from transformers import AutoModel
    encoder, info = AutoModel.from_pretrained(path, config=config, local_files_only=True,
                                             dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True)
    unexpected = [k for k in info.get("unexpected_keys", []) if not k.startswith(unused_prefixes)]
    if unexpected or any(info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete encoder load: {info}")
    return encoder


def unfreeze_last(layers, norm, count, option):
    if type(count) is not int or not 1 <= count <= len(layers):
        raise ValueError(f"{option} out of range")
    for layer in layers[-count:]:
        layer.requires_grad_(True)
    norm.requires_grad_(True)
