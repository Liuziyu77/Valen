from copy import deepcopy

from ..interfaces import ArchitectureBackend, Capabilities, Decision, PolicyInput


class QwenBackend(ArchitectureBackend):
    name = "qwen"
    unit_scope = "question"
    capabilities = Capabilities(video=True, frozen_feature_cache=True)

    def build_model(self, config):
        from .builder import build_qwen
        return build_qwen(config)

    def build_compiler(self, config, media_root):
        from transformers import AutoProcessor
        from valen.data.compilers.qwen import Compiler
        return Compiler(AutoProcessor.from_pretrained(config["model_path"], local_files_only=True),
                        media_root, config.get("max_length", 8192), config.get("media_kwargs"))

    def candidates(self, question):
        from valen.data.schema import candidates
        return candidates(question)

    def optimizer_groups(self, model, config):
        from .builder import qwen_optimizer_groups
        return qwen_optimizer_groups(model, config)

    def inference_units(self, model, state):
        for question in state.questions:
            yield [Decision(question, model(question))]

    def question_logits(self, model, question, rollout=None):
        features = getattr(rollout, "features", None)
        return model.score_features(features) if features is not None else model(question)

    def training_units(self, model, state, rollouts):
        if len(rollouts) != len(state.questions):
            raise ValueError("Rollout/question count mismatch")
        for question, rollout in zip(state.questions, rollouts):
            # Yield before the next forward, allowing immediate backward and release.
            yield [Decision(question, self.question_logits(model, question, rollout), rollout)]

    def policy_inputs(self, model, state, reference, cache_features=False):
        for question in state.questions:
            features = model.extract_features(question) if cache_features else None
            logits = model.score_features(features) if cache_features else model(question)
            ref = None
            if reference is not None:
                ref = model.score_features(features, reference) if cache_features else reference(question)
            yield PolicyInput(question, logits, ref, features)

    def validate_state(self, config, state):
        if config.get("stage") == "text" and any(not q.is_text for q in state.questions):
            raise ValueError("Text stage requires a text-only dataset")

    def validate_feature_cache(self, model):
        if not hasattr(model, "extract_features") or any(p.requires_grad for p in model.backbone.parameters()):
            raise ValueError("Feature reuse requires a fully frozen Valen backbone")
        if any(not n.startswith("head.") for n, p in model.named_parameters() if p.requires_grad):
            raise ValueError("Feature reuse requires only head parameters to be trainable")

    def make_reference(self, model, weights, cache_features=False):
        if not cache_features:
            return super().make_reference(model, weights)
        self.validate_feature_cache(model)
        reference = deepcopy(model.head)
        reference.load_state_dict({k.removeprefix("head."): v for k, v in weights.items()})
        return reference.requires_grad_(False).eval()

    def base_paths(self, config):
        return config["model_path"]

    def validate_initialization(self, previous, config):
        super().validate_initialization(previous, config)
        if previous.get("projection_dim", 256) != config.get("projection_dim", 256):
            raise ValueError("Initialization backbone/head mismatch")

    def adaptation(self, model, config):
        stage = config.get("stage", "joint")
        return {"stage": stage, "text": "lora" if stage != "warmup" else "frozen",
                "vision_merger": stage in {"joint", "vision_top"},
                "vision_unfreeze_layers": 4 if stage == "vision_top" else 0,
                "lora_targets": getattr(model, "lora_targets", [])}
