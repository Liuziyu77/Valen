from .common import load_encoder, unfreeze_last


class ModernBertAdapter:
    @staticmethod
    def load(path):
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        if config.model_type != "modernbert":
            raise ValueError("Text encoder requires ModernBERT")
        for name in ("attention_dropout", "embedding_dropout", "mlp_dropout", "classifier_dropout"):
            setattr(config, name, 0.0)
        # Published weights include the MLM vocabulary head, which is not used.
        return load_encoder(path, config, ("head.", "decoder."))

    @staticmethod
    def unfreeze(encoder, count):
        unfreeze_last(encoder.layers, encoder.final_norm, count, "text_unfreeze_layers")
