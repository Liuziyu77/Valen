from .common import load_encoder, unfreeze_last


class DINOv3Adapter:
    @staticmethod
    def load(path):
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        if config.model_type != "dinov3_vit":
            raise ValueError("Vision encoder requires DINOv3 ViT")
        # Old/current/reference policies must receive deterministic features.
        for name in ("pos_embed_shift", "pos_embed_jitter", "pos_embed_rescale"):
            setattr(config, name, None)
        config.attention_dropout = config.drop_path_rate = 0.0
        return load_encoder(path, config)

    @staticmethod
    def unfreeze(encoder, count):
        unfreeze_last(encoder.model.layer, encoder.norm, count, "vision_unfreeze_layers")
