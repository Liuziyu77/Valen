from ..interfaces import ArchitectureBackend, Capabilities, Decision, PolicyInput


class DualEncoderBackend(ArchitectureBackend):
    name = "dual_encoder"
    unit_scope = "state"
    capabilities = Capabilities(structured_descriptions=True, custom_noul_criteria=True, max_images=1)

    def build_model(self, config):
        from .builder import build_dual_encoder
        return build_dual_encoder(config)

    def build_compiler(self, config, media_root):
        from valen.data.compilers.dual_encoder import build_parallel_compiler
        return build_parallel_compiler(config, media_root)

    def candidates(self, question):
        from valen.data.compilers.dual_encoder import parallel_candidates
        return parallel_candidates(question)

    def optimizer_groups(self, model, config):
        from .builder import dual_optimizer_groups
        return dual_optimizer_groups(model, config)

    def inference_units(self, model, state):
        logits = model.forward_state(state)
        if len(logits) != len(state.questions):
            raise ValueError("Logit/question count mismatch")
        if state.questions:
            yield [Decision(q, z) for q, z in zip(state.questions, logits)]

    def training_units(self, model, state, rollouts):
        if len(rollouts) != len(state.questions):
            raise ValueError("Rollout/question count mismatch")
        for unit in self.inference_units(model, state):
            for decision, rollout in zip(unit, rollouts):
                decision.rollout = rollout
            yield unit

    def policy_inputs(self, model, state, reference, cache_features=False):
        if cache_features:
            self.validate_feature_cache(model)
        logits = model.forward_state(state)
        refs = reference.forward_state(state) if reference is not None else [None] * len(state.questions)
        if len(logits) != len(state.questions) or len(refs) != len(state.questions):
            raise ValueError("Logit/question count mismatch")
        for q, z, ref in zip(state.questions, logits, refs):
            yield PolicyInput(q, z, ref)

    def base_paths(self, config):
        return {"text": config["text_model_path"], "vision": config["vision_model_path"]}

    def validate_initialization(self, previous, config):
        super().validate_initialization(previous, config)
        defaults = {"hidden_size": 512, "num_heads": 8, "intermediate_size": 2048,
                    "fusion_layers": 2, "question_layers": 3, "candidate_layers": 2, "question_queries": 8}
        if any(previous.get(k, v) != config.get(k, v) for k, v in defaults.items()):
            raise ValueError("Initialization dual-encoder architecture mismatch")

    def adaptation(self, model, config):
        stage = config.get("stage", "warmup")
        return {"stage": stage, "text_unfreeze_layers": config.get("text_unfreeze_layers", 6) if stage != "warmup" else 0,
                "vision_unfreeze_layers": config.get("vision_unfreeze_layers", 2) if stage == "vision_top" else 0}
