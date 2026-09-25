"""Single architecture dispatch point for model, compiler and execution policy."""
from .qwen.backend import QwenBackend
from .dual_encoder.backend import DualEncoderBackend
from valen.configuration import flatten_config

BACKENDS = {backend.name: backend for backend in (QwenBackend(), DualEncoderBackend())}


def get_backend(architecture="qwen"):
    if architecture not in BACKENDS:
        raise ValueError(f"Unsupported architecture: {architecture!r}")
    return BACKENDS[architecture]


def backend_for_model(model):
    # Models used by legacy callers without an architecture tag follow Qwen's
    # existing per-question contract; no method-presence probing is required.
    return get_backend(getattr(model, "architecture", "qwen"))


def normalize_model_config(config):
    config = flatten_config(config)
    get_backend(config.setdefault("architecture", "qwen"))
    return config


def build_model(config):
    config = normalize_model_config(config)
    return get_backend(config["architecture"]).build_model(config)


def build_compiler(config, media_root="."):
    config = normalize_model_config(config)
    return get_backend(config["architecture"]).build_compiler(config, media_root)


def optimizer_groups(model, config):
    config = normalize_model_config(config)
    return get_backend(config["architecture"]).optimizer_groups(model, config)
