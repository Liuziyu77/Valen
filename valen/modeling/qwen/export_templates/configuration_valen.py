"""Standalone Qwen decision configuration. / 独立的 Qwen 决策模型配置。"""
from transformers import Qwen3_5Config


class ValenQwenConfig(Qwen3_5Config):
    model_type = "valen_qwen"

    def __init__(self, valen_head=None, qwen_execution="shared_state", valen_max_length=32768,
                 valen_media_kwargs=None, **kwargs):
        if qwen_execution not in {"question", "shared_state"}:
            raise ValueError("qwen_execution must be question or shared_state")
        self.valen_head = valen_head or {"head_type": "mixer"}
        self.qwen_execution = qwen_execution
        self.valen_max_length = valen_max_length
        self.valen_media_kwargs = valen_media_kwargs or {}
        super().__init__(**kwargs)
