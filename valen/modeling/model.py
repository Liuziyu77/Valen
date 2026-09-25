"""Backward-compatible imports; canonical implementations live in architecture packages."""
from .qwen.model import ValenQwen, DecisionHead
from .factory import build_model, normalize_model_config, optimizer_groups

Valen = ValenQwen
